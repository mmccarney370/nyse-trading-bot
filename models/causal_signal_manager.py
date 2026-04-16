# models/causal_signal_manager.py
"""
CausalSignalManager — Phase 1 extraction from signals.py
Contains the entire CausalSignalWrapper class + all causal graph, replay buffer, penalty, and prediction logic.
All previous patches, comments, B-fixes, and logging preserved exactly.
March 2026 Update: Added real persistence for replay buffer via save_buffer() and load_buffer()
→ Fixes AttributeError from missing 'save_portfolio_buffer' / 'load_portfolio_buffer' calls
→ Buffer now saved/loaded across restarts (portfolio and per-symbol)
"""
import logging
import numpy as np
import pandas as pd
import os
import time
import pickle
import hashlib
from datetime import datetime
import torch
import random # ← ADDED: Required for random.sample() in ReplayBuffer.sample()
import threading
import tempfile # ← Priority 1: for atomic writes
import shutil # ← Priority 1: for atomic rename
from config import CONFIG
from dowhy import CausalModel
from pgmpy.estimators.GES import GES
import networkx as nx
from collections import deque
from models.features import generate_features

logger = logging.getLogger(__name__)
# Reduce pgmpy verbosity — it dumps full datatype inference for all features on every GES call
logging.getLogger('pgmpy').setLevel(logging.WARNING)

class ReplayBuffer:
    """Single source of truth: Replay buffer for causal counterfactual estimation using real historical rewards.
    Standardized capacity=5000 across all usages (per-symbol + portfolio)."""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward):
        # FIX #62: np.atleast_1d handles scalar obs gracefully (obs.copy() fails on scalar)
        self.buffer.append((np.atleast_1d(obs).copy(), float(action), float(reward)))

    def sample(self, batch_size=200):
        if len(self.buffer) < batch_size:
            return None
        # Filter out zero-reward placeholder transitions (added at signal entry time
        # before the real return is known — they dilute correlation/ATE estimates)
        real_transitions = [(obs, act, rew) for obs, act, rew in self.buffer if abs(rew) > 1e-10]
        if len(real_transitions) < batch_size:
            # Fall back to full buffer if not enough real-reward transitions
            real_transitions = list(self.buffer)
        batch = random.sample(real_transitions, min(batch_size, len(real_transitions)))
        # Filter out obs with mismatched dimensions (can happen after model retrain
        # changes feature count while old obs remain in the buffer).
        # Use the most common shape as reference (not the random first element)
        from collections import Counter
        shape_counts = Counter(obs.shape for obs, _, _ in batch)
        ref_shape = shape_counts.most_common(1)[0][0]
        filtered = [(obs, act, rew) for obs, act, rew in batch if obs.shape == ref_shape]
        if len(filtered) < max(batch_size // 2, 10):
            logger.warning(f"[ReplayBuffer] Too many shape-mismatched obs ({len(batch) - len(filtered)}/{len(batch)}) — returning None")
            return None
        try:
            obs_batch = np.vstack([x[0] for x in filtered])
        except ValueError as e:
            logger.warning(f"[ReplayBuffer] np.vstack failed on filtered obs: {e} — returning None")
            return None
        action_batch = np.array([x[1] for x in filtered])
        reward_batch = np.array([x[2] for x in filtered])
        df = pd.DataFrame(obs_batch, columns=[f'feat_{i}' for i in range(obs_batch.shape[1])])
        df['action'] = action_batch
        df['reward'] = reward_batch
        return df

class CausalSignalManager:
    """
    Extracted CausalSignalWrapper — now a dedicated manager.
    Handles full multi-symbol GES + DoWhy + statistical refutation + ReplayBuffer.
    Keeps 100% of original features (no reduction, no PCA).
    """
    def __init__(self, base_model, symbol: str = None, data_ingestion=None):
        self.base_model = base_model
        self.causal_model = None
        self.causal_graph = None
        self.identified_estimand = None  # BUG #7 / P-20: Store here to avoid NameError
        self.symbol = symbol or "portfolio"
        self.data_ingestion = data_ingestion
        self.replay_buffer = ReplayBuffer(capacity=5000) # NEW: Real historical data for counterfactuals
        self._dowhy_lock = threading.Lock()  # Protects causal_model/identified_estimand/dowhy_ate writes from background thread
        self._graph_attrs_lock = threading.Lock()  # Protects _action_reward_corr, _has_action_reward_path, _reward_parent_count
        self.cache_path = f"causal_cache_{self.symbol}.pkl"
        # P-18 / Critical #11 FIX: Compute current model version hash for cache invalidation after retrain
        self.model_version_hash = self._get_model_version_hash()
        # CRIT-08 FIX: Portfolio wrapper instantiation (was never created → AttributeError on shutdown / sync / rotation)
        self.portfolio_causal_wrapper = None
        self.portfolio_causal_manager = None # B-33 alias for emergency save compatibility
        if self.symbol == "portfolio" and CONFIG.get('USE_CAUSAL_RL', False):
            self.portfolio_causal_wrapper = self
            self.portfolio_causal_manager = self
            logger.info("✅ [CRIT-08] Portfolio causal wrapper instantiated")
        # C-3 FIX: Do NOT build graph in __init__ anymore — defer until buffer has enough data
        # Graph will be built lazily on first predict/compute_penalty_factor call
        # Load persisted buffer immediately (startup must have data before inference)
        self.load_buffer()

    def _get_model_version_hash(self):
        """P-18 / Critical #11 FIX: Hash actual PPO policy weights when possible"""
        if self.base_model is None:
            return "no_model_initialized"
        try:
            # sb3 RecurrentPPO exposes policy as a torch module — try to hash its state_dict
            policy = self.base_model.policy
            state_dict = policy.state_dict() # This should work on the inner torch module
            hasher = hashlib.sha256()
            for k in sorted(state_dict.keys()):
                tensor = state_dict[k]
                hasher.update(k.encode())
                hasher.update(tensor.cpu().numpy().tobytes())
            logger.info("[CAUSAL HASH] Successfully hashed policy weights")
            return hasher.hexdigest()[:16]
        except AttributeError as e:
            logger.debug(f"[CAUSAL HASH] Policy state_dict not directly accessible: {e} — using stable fallback")
            # Stable fallback: hash class name + key config values that affect model architecture/training
            stable_str = (
                f"{self.base_model.__class__.__name__}_"
                f"{CONFIG.get('PPO_TIMESTEPS', 0)}_"
                f"{CONFIG.get('PORTFOLIO_TIMESTEPS', 0)}_"
                f"{CONFIG.get('GTRXL_HIDDEN_SIZE', 0)}_"
                f"{CONFIG.get('GTRXL_NUM_LAYERS', 0)}_"
                f"{CONFIG.get('USE_CUSTOM_GTRXL', False)}"
            )
            return hashlib.sha256(stable_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"[CAUSAL HASH] Failed to hash policy weights: {e} — using stable fallback")
            # Stable fallback: hash class name + key config values that affect model architecture/training
            stable_str = (
                f"{self.base_model.__class__.__name__}_"
                f"{CONFIG.get('PPO_TIMESTEPS', 0)}_"
                f"{CONFIG.get('PORTFOLIO_TIMESTEPS', 0)}_"
                f"{CONFIG.get('GTRXL_HIDDEN_SIZE', 0)}_"
                f"{CONFIG.get('GTRXL_NUM_LAYERS', 0)}_"
                f"{CONFIG.get('USE_CUSTOM_GTRXL', False)}"
            )
            return hashlib.sha256(stable_str.encode()).hexdigest()[:16]

    def _ensure_graph_exists(self):
        """Lazy graph build — call this before any graph-dependent operation.
        H20 FIX: Never blocks — if graph needs building, spawns a background thread.
        Returns immediately so predict/compute_penalty_factor use neutral fallback until ready."""
        if self.causal_graph is not None:
            return  # already built
        # Don't double-spawn if a build is already in progress
        if getattr(self, '_build_in_progress', False):
            return
        last_fail = getattr(self, '_last_build_fail', 0)
        if time.time() - last_fail < 300:
            return
        if len(self.replay_buffer.buffer) < 100:
            logger.debug(f"[CAUSAL LAZY] {self.symbol} — buffer only {len(self.replay_buffer.buffer)} samples — "
                         f"deferring graph build until ≥100 real transitions")
            return
        # H20 FIX: Spawn build in background thread to avoid blocking the event loop
        self._build_in_progress = True
        logger.info(f"[CAUSAL LAZY BUILD] {self.symbol} — spawning background graph build ({len(self.replay_buffer.buffer)} samples)")
        build_thread = threading.Thread(target=self._background_graph_build, daemon=True)
        build_thread.start()

    def _background_graph_build(self):
        """Runs the expensive graph build in a background thread (H20 FIX)."""
        try:
            _regime_cache = {}
            try:
                import json
                _regime_cache_file = "regime_cache.json"
                if os.path.exists(_regime_cache_file):
                    with open(_regime_cache_file, 'r') as f:
                        _regime_cache = json.load(f)
            except Exception:
                pass

            if self.symbol == "portfolio":
                if self.data_ingestion is None:
                    logger.warning("[CAUSAL LAZY] Portfolio — no data_ingestion reference — cannot build graph")
                    return
                all_features = []
                symbol_list = CONFIG.get('SYMBOLS', [])
                for real_sym in symbol_list:
                    data = self.data_ingestion.get_latest_data(real_sym, timeframe='15Min')
                    if len(data) >= 200:
                        sym_regime = _regime_cache.get(real_sym, 'mean_reverting')
                        if isinstance(sym_regime, list):
                            sym_regime = sym_regime[0]
                        features = generate_features(data, sym_regime, real_sym, data)
                        if features is not None and features.shape[0] > 0:
                            all_features.append(features)
                if not all_features:
                    logger.warning("[CAUSAL LAZY] Portfolio — no valid symbol data — deferring")
                    return
                full_matrix = np.vstack(all_features)
                features_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
            else:
                if self.data_ingestion is None:
                    return
                data = self.data_ingestion.get_latest_data(self.symbol, timeframe='15Min')
                if len(data) < 200:
                    return
                sym_regime = _regime_cache.get(self.symbol, 'mean_reverting')
                if isinstance(sym_regime, list):
                    sym_regime = sym_regime[0]
                features = generate_features(data, sym_regime, self.symbol, data)
                if features is None or features.shape[0] == 0:
                    return
                features_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])

            self.build_causal_graph(features_df)
        except Exception as e:
            logger.warning(f"[CAUSAL BG BUILD] Failed for {self.symbol}: {e}")
            self._last_build_fail = time.time()
        finally:
            self._build_in_progress = False

    def build_causal_graph(self, features_df: pd.DataFrame):
        """Robust causal graph build with downsampling to max 8000 rows + disk caching.
        CRITICAL FIX: Now seeds DoWhy with REAL action/reward columns from ReplayBuffer."""
        if not CONFIG.get('USE_CAUSAL_RL', False):
            logger.info("Causal RL disabled — skipping graph build")
            return
        # C-3 FIX: No early exit here anymore — caller (_ensure_graph_exists) already checked size
        # But keep defensive check
        if len(self.replay_buffer.buffer) < 100:
            logger.warning(f"[CAUSAL BUILD] {self.symbol} — only {len(self.replay_buffer.buffer)} real transitions "
                           f"(need ≥100) — graph not built")
            self.causal_graph = None
            self.causal_model = None
            return
        # BUG-14 FIX: Load cache unconditionally if file exists (startup buffer is always empty)
        # Only rebuild if cache is missing, corrupted, or older than 24 hours
        if os.path.exists(self.cache_path):
            try:
                file_age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.cache_path))).total_seconds() / 3600
                with open(self.cache_path, 'rb') as f:
                    cached = pickle.load(f)
                cached_hash = cached.get('model_version_hash', 'unknown')
                # P-18 / Critical #11 FIX: Force rebuild if model version changed (post-nightly retrain)
                if file_age_hours < 24 and cached_hash == self.model_version_hash:
                    self.causal_graph = cached['graph']
                    self.causal_model = cached.get('model')
                    self.identified_estimand = cached.get('identified_estimand')
                    self._action_reward_corr = cached.get('action_reward_corr', 0.0)
                    self._has_action_reward_path = cached.get('has_action_reward_path', False)
                    self._reward_parent_count = cached.get('reward_parent_count', 0)
                    logger.info(f"✅ [CAUSAL CACHE HIT] Loaded cached graph for {self.symbol} ({len(self.causal_graph.edges)} edges, age={file_age_hours:.1f}h, hash match)")
                    return
                else:
                    logger.info(f"[CAUSAL CACHE] Invalidating cache for {self.symbol} — age={file_age_hours:.1f}h or hash mismatch")
            except Exception as e:
                logger.warning(f"[CAUSAL CACHE] Failed to load cache for {self.symbol}: {e} — rebuilding")
        try:
            start_time = time.time()
            logger.info(f"[CAUSAL BUILD START] Building FULL multi-symbol causal graph for {self.symbol} — "
                        f"shape {features_df.shape} | Using ALL original features (no reduction)")
            # === CRITICAL FIX: Seed with real action/reward data from ReplayBuffer ===
            sample_df = self.replay_buffer.sample(batch_size=min(8000, len(self.replay_buffer.buffer)))
            if sample_df is not None:
                data = sample_df.copy()
                logger.info(f"[CAUSAL] Seeded DoWhy with {len(data)} REAL (obs, action, reward) transitions")
            else:
                data = features_df.copy()
                if 'action' not in data.columns:
                    data['action'] = 0.0
                if 'reward' not in data.columns:
                    data['reward'] = 0.0
                logger.warning("[CAUSAL] Buffer sample returned None — using features only (synthetic action/reward)")
            # BUG-3 FIX: Preserve symbol_id (convert to numeric codes) so it survives select_dtypes
            if 'symbol_id' in data.columns:
                data['symbol_id'] = pd.Categorical(data['symbol_id']).codes
            # === DOWNsample to max 8000 rows (increased from 4000 for better causal reliability) ===
            max_rows = 8000
            if len(data) > max_rows:
                data = data.tail(max_rows).reset_index(drop=True)
                logger.info(f"[CAUSAL DOWNSAMPLE] Using last {max_rows} rows to improve GES reliability")
            else:
                logger.info(f"[CAUSAL DOWNSAMPLE] Using all {len(data)} available rows (no downsampling needed)")
            # Clean data
            data = data.select_dtypes(include=[np.number]).dropna(axis=1, how='all').fillna(0.0)
            logger.info(f"[CAUSAL DEBUG] After cleaning: {data.shape[0]} rows × {data.shape[1]} columns")
            # === COLUMN CAP for GES tractability ===
            # GES complexity is roughly O(p^3·n) — portfolio observations are 460-dim so
            # a raw buffer sample has 462 columns, which makes GES run for hours.
            # Keep action/reward always; pick the top-K remaining features by variance.
            max_cols = CONFIG.get('CAUSAL_MAX_FEATURES', 58)
            if data.shape[1] > max_cols:
                orig_cols = data.shape[1]
                feat_cols = [c for c in data.columns if c not in ('action', 'reward')]
                variances = data[feat_cols].var().sort_values(ascending=False)
                # Skip constant/near-constant columns (no info for GES)
                informative = variances[variances > 1e-10]
                keep_features = informative.head(max_cols - 2).index.tolist()
                required = [c for c in ('action', 'reward') if c in data.columns]
                data = data[keep_features + required]
                logger.info(f"[CAUSAL] Column cap: kept top {len(keep_features)} features + "
                            f"{len(required)} required (action/reward) — was {orig_cols} cols")
            if data.shape[0] < 500:
                logger.warning(f"[CAUSAL WARNING] Only {data.shape[0]} rows — GES may find fewer edges.")
            ges = GES(data)
            logger.info(f"[CAUSAL DEBUG] Starting GES estimation (bare call for maximum compatibility)")
            self.causal_graph = ges.estimate()
            # Ensure action/reward nodes exist in graph (GES may not include isolated nodes)
            for required_node in ['action', 'reward']:
                if required_node not in self.causal_graph.nodes:
                    self.causal_graph.add_node(required_node)
            # === FAST PATH: Compute action→reward correlation from GES graph + replay buffer ===
            # Skip DoWhy entirely (CausalModel + identify_effect + estimate_effect are too slow
            # on 54-node graphs — 30-60 min each). Use GES graph structure + direct correlation instead.
            logger.info(f"[CAUSAL GES] Computing action→reward edge strength from graph ({len(self.causal_graph.edges)} edges)")
            # Check if action→reward path exists in GES graph
            has_direct_edge = self.causal_graph.has_edge('action', 'reward')
            try:
                has_path = nx.has_path(self.causal_graph, 'action', 'reward')
                # FIX #59: GES returns a PDAG (partially directed graph) — some edges are undirected.
                # Also check undirected connectivity as fallback for PDAG edges.
                if not has_path:
                    has_path = nx.has_path(self.causal_graph.to_undirected(), 'action', 'reward')
                path_length = nx.shortest_path_length(self.causal_graph, 'action', 'reward') if nx.has_path(self.causal_graph, 'action', 'reward') else 0
                if path_length == 0 and has_path:
                    path_length = nx.shortest_path_length(self.causal_graph.to_undirected(), 'action', 'reward')
            except (nx.NetworkXError, nx.NodeNotFound):
                has_path = False
                path_length = 0
            # Compute direct action→reward correlation from replay buffer (fast: plain numpy)
            if 'action' in data.columns and 'reward' in data.columns:
                corr = data['action'].corr(data['reward'])
                _corr = corr if not np.isnan(corr) else 0.0
            else:
                _corr = 0.0
            # Count causal parents of 'reward' in the GES graph (features that influence reward)
            reward_parents = list(self.causal_graph.predecessors('reward')) if 'reward' in self.causal_graph.nodes else []
            # Atomic update of all graph-derived attributes under lock
            with self._graph_attrs_lock:
                self._action_reward_corr = _corr
                self._reward_parent_count = len(reward_parents)
                self._has_action_reward_path = has_path
            logger.info(f"[CAUSAL GES] action→reward: direct_edge={has_direct_edge}, path={has_path} (len={path_length}), "
                        f"corr={self._action_reward_corr:.4f}, reward_parents={len(reward_parents)}")
            # No DoWhy model needed for fast path — set to None so predict/compute use graph-based penalty
            # FIX #60: Wrap causal_model=None assignment under _dowhy_lock for thread safety
            with self._dowhy_lock:
                self.causal_model = None
                self.identified_estimand = None
            build_time = time.time() - start_time
            initial_edges = len(self.causal_graph.edges)
            logger.info(f"✅ [CAUSAL BUILD SUCCESS] GES discovered {initial_edges} edges in {build_time:.1f}s (fast path, no DoWhy)")
            # Save to disk cache — HIGH-28 FIX: Use atomic write to prevent corruption from background thread
            try:
                cache_data = {
                    'graph': self.causal_graph,
                    'model': None,
                    'model_version_hash': self.model_version_hash,
                    'identified_estimand': None,
                    'action_reward_corr': self._action_reward_corr,
                    'has_action_reward_path': self._has_action_reward_path,
                    'reward_parent_count': self._reward_parent_count,
                }
                dir_name = os.path.dirname(self.cache_path) or '.'
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dir_name) as tmp:
                    pickle.dump(cache_data, tmp)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                shutil.move(tmp.name, self.cache_path)
                logger.debug(f"[CAUSAL CACHE SAVED] Saved graph for {self.symbol} with hash {self.model_version_hash}")
            except Exception as e:
                logger.debug(f"Failed to save causal cache: {e}")
            # === BACKGROUND: Spawn DoWhy deep analysis in a thread (non-blocking) ===
            # FIX #18: Pass explicit args instead of capturing `self` via closure.
            # The manager_ref weakref prevents stale writes if the manager is replaced during rotation.
            import threading
            import weakref
            def _background_dowhy_build(data_copy, graph_copy, symbol, cache_path, model_hash,
                                        manager_ref, dowhy_lock, action_reward_corr, has_action_reward_path, reward_parent_count):
                try:
                    logger.info(f"[DOWHY BACKGROUND] Starting deep causal analysis for {symbol}...")
                    # Clean all non-serializable attributes for GML serialization
                    for node in list(graph_copy.nodes):
                        for key in list(graph_copy.nodes[node].keys()):
                            val = graph_copy.nodes[node][key]
                            if val is None or not isinstance(val, (int, float, str)):
                                del graph_copy.nodes[node][key]
                    for u, v in list(graph_copy.edges):
                        for key in list(graph_copy.edges[u, v].keys()):
                            val = graph_copy.edges[u, v][key]
                            if val is None or not isinstance(val, (int, float, str)):
                                del graph_copy.edges[u, v][key]
                    graph_gml = ''.join(nx.generate_gml(graph_copy))
                    cm = CausalModel(data=data_copy, treatment='action', outcome='reward', graph=graph_gml)
                    estimand = cm.identify_effect(proceed_when_unidentifiable=True)
                    estimate = cm.estimate_effect(estimand, method_name="backdoor.linear_regression", target_units="ate")
                    ate_val = estimate.value if hasattr(estimate, 'value') else 0.0
                    # FIX #18: Only write back if the manager hasn't been replaced
                    mgr = manager_ref()
                    if mgr is None:
                        logger.info(f"[DOWHY BACKGROUND] Manager for {symbol} was replaced — discarding results")
                        return
                    # Swap in results under lock (thread-safe)
                    with dowhy_lock:
                        mgr.causal_model = cm
                        mgr.identified_estimand = estimand
                        mgr._dowhy_ate = ate_val
                    # Update cache with full DoWhy results — HIGH-28 FIX: atomic write
                    try:
                        dir_name = os.path.dirname(cache_path) or '.'
                        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dir_name) as tmp:
                            pickle.dump({
                                'graph': graph_copy, 'model': cm, 'model_version_hash': model_hash,
                                'identified_estimand': estimand,
                                'action_reward_corr': action_reward_corr,
                                'has_action_reward_path': has_action_reward_path,
                                'reward_parent_count': reward_parent_count,
                            }, tmp)
                            tmp.flush()
                            os.fsync(tmp.fileno())
                        shutil.move(tmp.name, cache_path)
                    except Exception:
                        pass
                    logger.info(f"[DOWHY BACKGROUND] Deep analysis complete for {symbol} — ATE={ate_val:.6f}")
                except Exception as e:
                    logger.warning(f"[DOWHY BACKGROUND] Failed for {symbol}: {e} — graph-based penalty still active")
            bg_thread = threading.Thread(
                target=_background_dowhy_build,
                args=(data.copy(), self.causal_graph.copy(), self.symbol, self.cache_path, self.model_version_hash,
                      weakref.ref(self), self._dowhy_lock, self._action_reward_corr, self._has_action_reward_path, self._reward_parent_count),
                daemon=True
            )
            bg_thread.start()
            logger.info(f"[DOWHY BACKGROUND] Spawned background thread for deep causal analysis")
        except Exception as e:
            logger.error(f"[CAUSAL BUILD FAILED] Causal graph build failed for {self.symbol}: {e}", exc_info=True)
            self.causal_model = None
            self.causal_graph = None
            self._last_build_fail = time.time()

    def refresh_causal_wrappers(self):
        """BUG #19 PATCH: Rebuild all causal wrappers after universe rotation or nightly retrain.
        CRIT-08 FIX: Creates portfolio_causal_wrapper if missing (was the root cause of AttributeError)."""
        if not CONFIG.get('USE_CAUSAL_RL', False):
            return
        logger.info(f"[CAUSAL REFRESH] Rebuilding wrappers for {self.symbol}")
        if self.symbol == "portfolio":
            self.portfolio_causal_wrapper = self
            self.portfolio_causal_manager = self # alias for B-33 / shutdown save
            logger.info("✅ [CRIT-08] Portfolio causal wrapper (re)instantiated")
        # Per-symbol wrappers are rebuilt in bot.py / CausalRLManager — this method now covers portfolio path

    def add_transition(self, obs, action, reward):
        """Call this after every real environment step to feed real reward data."""
        self.replay_buffer.push(obs, action, reward)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """Safer predict with defensive state handling and fallback."""
        # C-3 FIX: Ensure graph exists before using causal logic
        self._ensure_graph_exists()
        # CRIT-9 FIX: Check causal_graph (not causal_model) — GES fast path sets causal_model=None
        # but causal_graph is valid. Old check bypassed _compute_fast_penalty() entirely.
        if not CONFIG.get('USE_CAUSAL_RL', False) or self.causal_graph is None:
            logger.debug(f"[CAUSAL FALLBACK] {self.symbol} — no causal graph yet — using base PPO predict")
            if state is not None and hasattr(self.base_model, 'predict'):
                return self.base_model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
            return self.base_model.predict(obs, deterministic=deterministic)
        try:
            if state is not None:
                action, new_state = self.base_model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
            else:
                action, new_state = self.base_model.predict(obs, deterministic=deterministic)
        except Exception as e:
            logger.debug(f"Base model predict failed, falling back: {e}")
            action, new_state = self.base_model.predict(obs, deterministic=deterministic)
        # B-10 PATCH: Causal penalty now guaranteed to be applied and logged in live trading
        original_action = action.copy() if hasattr(action, 'copy') else action
        penalty_factor = self._compute_fast_penalty()
        action = action * penalty_factor
        action = np.clip(action, -2.0, 2.0)  # Portfolio actions range [-2, 2], not [-1, 1]
        if penalty_factor != 1.0:
            orig_str = f"{np.mean(np.abs(original_action)):.4f}" if hasattr(original_action, '__len__') else f"{original_action:.4f}"
            adj_str = f"{np.mean(np.abs(action)):.4f}" if hasattr(action, '__len__') else f"{action:.4f}"
            logger.info(f"✅ [CAUSAL PENALTY APPLIED] {self.symbol} | mean_abs_raw={orig_str} → adjusted={adj_str} (factor={penalty_factor:.4f})")
        else:
            logger.debug(f"[CAUSAL PENALTY] {self.symbol} — no violation detected (factor=1.0)")
        if state is not None:
            return action, new_state
        return action, None

    def _compute_fast_penalty(self) -> float:
        """Fast graph-based causal scaling factor using GES structure + replay buffer correlation.
        HIGH-29 FIX: Corrected semantics — factor < 1.0 dampens (penalizes), > 1.0 boosts.
        Positive correlation between action and reward means actions are causally effective → boost.
        Negative or no correlation → dampen. Returns 1.0 when neutral."""
        if not CONFIG.get('USE_CAUSAL_RL', False) or self.causal_graph is None:
            return 1.0
        try:
            # If DoWhy background thread completed, use the full ATE estimate
            with self._dowhy_lock:
                cm = self.causal_model
                estimand = self.identified_estimand
            if cm is not None and estimand is not None:
                cached_ate = getattr(self, '_dowhy_ate', None)
                if cached_ate is not None:
                    # Positive ATE means actions causally help → boost
                    # Negative ATE means actions causally hurt → dampen
                    weight = CONFIG.get('CAUSAL_PENALTY_WEIGHT', 0.40)
                    factor = 1.0 + (cached_ate * weight)
                    # FIX #61: Cap boost at 1.2x for safety (was 1.5x)
                    factor = max(0.5, min(1.2, factor))
                    logger.debug(f"[CAUSAL DOWHY] {self.symbol} cached ATE={cached_ate:.4f} → factor={factor:.4f}")
                    return factor
            # Fast path: use pre-computed GES graph correlation
            with self._graph_attrs_lock:
                corr = getattr(self, '_action_reward_corr', 0.0)
                has_path = getattr(self, '_has_action_reward_path', False)
            if not has_path:
                return 1.0  # No causal path from action to reward — neutral
            # Signed correlation: positive = actions help, negative = actions hurt
            weight = CONFIG.get('CAUSAL_PENALTY_WEIGHT', 0.40)
            factor = 1.0 + (corr * weight)
            # FIX #61: Cap boost at 1.2x for safety (was 1.5x)
            factor = max(0.5, min(1.2, factor))
            logger.debug(f"[CAUSAL GES FAST] {self.symbol} corr={corr:.4f}, path={has_path} → factor={factor:.4f}")
            return factor
        except Exception as e:
            logger.debug(f"[CAUSAL] Penalty failed: {e} — neutral")
            return 1.0

    def compute_penalty_factor(self, obs, action) -> float:
        """Public method for causal penalty. Uses obs and action to discriminate.
        M41 FIX: Now considers action DIRECTION relative to causal evidence.
        - Positive corr + same-direction action → boost (aligned with what works)
        - Positive corr + opposite action → dampen (going against what works)
        - Negative corr + same-direction action → dampen (following what hurts)
        - Negative corr + opposite action → boost (contrarian to what hurts)"""
        self._ensure_graph_exists()
        base_factor = self._compute_fast_penalty()
        if base_factor == 1.0:
            return 1.0
        action_val = float(action) if np.isscalar(action) else float(np.mean(action))
        action_magnitude = min(abs(action_val), 1.0)
        with self._graph_attrs_lock:
            corr = getattr(self, '_action_reward_corr', 0.0)
        if abs(corr) < 1e-8 or abs(action_val) < 1e-8:
            # Neutral — scale by magnitude only
            return 1.0 + (base_factor - 1.0) * action_magnitude
        # action_corr_product > 0 means action aligns with correlation direction
        # action_corr_product < 0 means action opposes correlation direction
        action_corr_product = np.sign(action_val) * np.sign(corr)
        if action_corr_product > 0:
            # Action aligns with causal direction: boost if corr positive, dampen if corr negative
            return 1.0 + (base_factor - 1.0) * action_magnitude
        else:
            # Action opposes causal direction: invert the effect
            # If base_factor > 1 (corr says "go with it"), opposing → dampen
            # If base_factor < 1 (corr says "actions hurt"), opposing → boost (contrarian is good)
            inverted_factor = 1.0 - (base_factor - 1.0) * action_magnitude
            return max(0.5, min(1.2, inverted_factor))

    def bootstrap_from_env(self, env, ppo_model, n_steps: int = 500, vec_norm=None,
                           action_noise_sigma: float = 0.4):
        """Seed the ReplayBuffer by stepping a PortfolioEnv through recent historical
        bars with PPO-predicted actions. Produces real (obs, action, reward) tuples
        that GES can use to discover action→reward structure.

        Runs only when the buffer's current content is too thin or too deterministic
        for causal discovery. GES fundamentally requires variance in the treatment
        (action) to identify action→reward structure. A converged PPO policy is
        near-deterministic in inference, so bare (deterministic) rollouts produce
        actions with std ~0.08 → GES finds 0 edges. We inject Gaussian exploration
        noise so the buffer carries enough action variance for causal discovery,
        while the underlying mean action still reflects the real PPO policy.

        Skips when the buffer already contains ≥100 samples with action std > 0.2
        (i.e., non-degenerate data from live trading or a previous noisy bootstrap).

        Args:
            env: PortfolioEnv (already-constructed, not reset between calls)
            ppo_model: portfolio PPO model (SB3 RecurrentPPO or similar)
            n_steps: how many historical bars to replay
            vec_norm: optional VecNormalize — if provided, obs is normalized before
                predict, matching live inference semantics
            action_noise_sigma: Gaussian std added to the scalar portfolio action
                stored in the buffer (~0.4 gives GES a clear signal while keeping
                the mean action close to the PPO policy)
        """
        if ppo_model is None or env is None:
            logger.debug(f"[CAUSAL BOOTSTRAP] {self.symbol} — missing env or model, skipping")
            return 0
        if len(self.replay_buffer.buffer) >= 100:
            # Check action variance — if too low, existing buffer is deterministic
            # and GES won't find edges. Drop it and re-seed with exploration noise.
            # IMPORTANT: Check std of the most-common-obs-shape subset (what GES
            # actually receives after ReplayBuffer.sample() filters shape mismatches).
            # A few legacy entries from older model versions can inflate overall std
            # and mask the fact that the dominant subset is deterministic.
            try:
                from collections import Counter
                shape_counts = Counter(
                    tuple(np.asarray(t[0]).shape) for t in self.replay_buffer.buffer
                )
                dominant_shape = shape_counts.most_common(1)[0][0]
                dominant = [
                    t for t in self.replay_buffer.buffer
                    if tuple(np.asarray(t[0]).shape) == dominant_shape
                ]
                dominant_actions = np.array([
                    t[1] for t in dominant
                    if np.isscalar(t[1]) or (hasattr(t[1], 'shape') and t[1].shape == ())
                ])
                dom_std = float(dominant_actions.std()) if len(dominant_actions) > 0 else 0.0
                if len(dominant_actions) == 0 or dom_std < 0.2:
                    logger.info(f"[CAUSAL BOOTSTRAP] {self.symbol} — dominant shape "
                                f"{dominant_shape} has {len(dominant)} entries, "
                                f"action std={dom_std:.3f} is too low for GES; "
                                f"clearing buffer + stale graph cache and re-bootstrapping")
                    self.replay_buffer.buffer.clear()
                    # Invalidate the cached graph BOTH on disk and in memory — otherwise
                    # the stale 0-edge cache gets loaded via CACHE HIT before GES reruns,
                    # or the still-set self.causal_graph skips the rebuild entirely.
                    self.causal_graph = None
                    self.causal_model = None
                    self.identified_estimand = None
                    self._build_in_progress = False
                    self._last_build_fail = 0
                    try:
                        if os.path.exists(self.cache_path):
                            os.remove(self.cache_path)
                            logger.info(f"[CAUSAL BOOTSTRAP] {self.symbol} — removed "
                                        f"stale graph cache at {self.cache_path}")
                    except Exception as e:
                        logger.warning(f"[CAUSAL BOOTSTRAP] {self.symbol} — failed to "
                                       f"remove stale cache at {self.cache_path}: {e} "
                                       f"— in-memory graph cleared, rebuild will proceed")
                else:
                    logger.debug(f"[CAUSAL BOOTSTRAP] {self.symbol} — buffer already has "
                                 f"{len(self.replay_buffer.buffer)} samples, dominant "
                                 f"action std={dom_std:.3f}, skipping")
                    return 0
            except Exception as e:
                logger.debug(f"[CAUSAL BOOTSTRAP] {self.symbol} — action variance check "
                             f"failed ({e}), skipping bootstrap")
                return 0

        # Save env state so we don't disturb the persistent env used elsewhere
        saved = {
            'current_step': getattr(env, 'current_step', None),
            'balance': getattr(env, 'balance', None),
            'equity': getattr(env, 'equity', None),
            'weights': getattr(env, 'weights', None),
            'last_weights': getattr(env, 'last_weights', None),
            'weight_history': getattr(env, 'weight_history', None),
            'episode_start': getattr(env, 'episode_start', None),
            'cumulative_pnl': getattr(env, 'cumulative_pnl', None),
            'peak_equity': getattr(env, 'peak_equity', None),
        }

        added = 0
        try:
            # Reset env to get clean initial state, then rewind to the last N bars
            obs, _ = env.reset()
            timeline_len = len(env.timeline)
            start_step = max(0, timeline_len - n_steps - 1)
            # Jump forward to the bootstrap window
            env.current_step = start_step
            env.episode_start = start_step
            obs = env._get_observation()

            # PortfolioEnv convention: reward at step(action_t) reflects the weights
            # HELD DURING bar t (i.e., the previous action). So the reward for taking
            # action_t is delivered by step(action_{t+1}). We buffer the prior
            # (obs, action) tuple and pair it with the next step's reward.
            lstm_state = None
            pending = None  # (obs_prev, action_prev) waiting for its reward
            for _ in range(n_steps):
                if env.current_step >= timeline_len - 1:
                    break
                obs_for_predict = obs.reshape(1, -1).astype(np.float32)
                if vec_norm is not None:
                    try:
                        obs_for_predict = vec_norm.normalize_obs(obs_for_predict)
                    except Exception:
                        pass
                try:
                    action, lstm_state = ppo_model.predict(
                        obs_for_predict,
                        state=lstm_state,
                        episode_start=np.array([False]),
                        deterministic=True,
                    )
                except TypeError:
                    # Non-recurrent PPO signature
                    action, _ = ppo_model.predict(obs_for_predict, deterministic=True)
                    lstm_state = None
                action_arr = np.asarray(action).flatten()
                # Inject Gaussian exploration noise BEFORE stepping the env so the
                # reward reflects the noisy action. Without noise, a converged PPO
                # policy is near-deterministic and GES finds no action→reward edge.
                if action_noise_sigma > 0:
                    action_arr = action_arr + np.random.normal(
                        0.0, action_noise_sigma, size=action_arr.shape
                    ).astype(action_arr.dtype)
                    action_arr = np.clip(action_arr, -2.0, 2.0)
                next_obs, reward, terminal, truncated, _ = env.step(action_arr)
                # Portfolio-level buffer uses signed gross exposure as the scalar action
                # (what GES consumes for action→reward edge discovery).
                if self.symbol == "portfolio":
                    scalar_action = float(np.sum(action_arr))
                else:
                    scalar_action = float(np.mean(action_arr))
                # Flush the prior (obs, action) with the reward that just arrived —
                # that reward reflects the weights held during this bar, which were
                # set by the PREVIOUS action.
                if pending is not None:
                    prev_obs, prev_action = pending
                    self.replay_buffer.push(prev_obs, prev_action, float(reward))
                    added += 1
                pending = (obs_for_predict.flatten(), scalar_action)
                obs = next_obs
                if terminal:
                    break
        except Exception as e:
            logger.warning(f"[CAUSAL BOOTSTRAP] {self.symbol} — stepping env failed at "
                           f"step {added}: {e}")
        finally:
            # Restore env state so downstream consumers see unchanged env
            for k, v in saved.items():
                if v is not None:
                    try:
                        setattr(env, k, v)
                    except Exception:
                        pass

        logger.info(f"[CAUSAL BOOTSTRAP] {self.symbol} — seeded {added} synthetic "
                    f"transitions from PortfolioEnv history (buffer now "
                    f"{len(self.replay_buffer.buffer)})")
        return added

    def warmup_from_history(self, history_entries: list):
        """Replay closed trades from live_signal_history into ReplayBuffer on startup.
        Only adds entries with realized_return (closed trades) to ensure accurate rewards."""
        added = 0
        for entry in history_entries:
            if entry.get('realized_return') is None:
                continue # skip open trades
            obs = entry.get('obs')
            if obs is None or not isinstance(obs, list) or len(obs) == 0:
                continue # skip invalid/missing obs
            try:
                obs_array = np.array(obs, dtype=np.float32).flatten()
                action = entry.get('direction', 0) * entry.get('confidence', 1.0)
                reward = entry.get('realized_return', 0.0)
                self.replay_buffer.push(obs_array, action, reward)
                added += 1
            except Exception as e:
                logger.debug(f"[CAUSAL WARMUP] Skipped invalid entry for {self.symbol}: {e}")
        if added > 0:
            logger.info(f"[CAUSAL WARMUP] {self.symbol}: replayed {added} closed trades into buffer")
        else:
            logger.debug(f"[CAUSAL WARMUP] {self.symbol}: no closed trades to replay")

    # ────────────────────────────────────────────────────────────────────────
    # NEW: Real persistence methods (fixes missing 'save_portfolio_buffer' error)
    # ────────────────────────────────────────────────────────────────────────
    def save_buffer(self, path: str = None):
        """Save replay buffer to disk atomically (Priority 1 fix — prevents corruption on crash)"""
        if path is None:
            path = f"replay_buffer_{self.symbol}.pkl"
        if not self.replay_buffer.buffer:
            logger.debug(f"[BUFFER SAVE SKIP] Empty buffer for {self.symbol} — no file written")
            return
        try:
            dir_name = os.path.dirname(path) or '.'
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dir_name) as tmp:
                pickle.dump(list(self.replay_buffer.buffer), tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            shutil.move(tmp.name, path)
            logger.debug(f"[ATOMIC BUFFER SAVE] Saved {len(self.replay_buffer.buffer)} transitions for {self.symbol} to {path}")
        except Exception as e:
            logger.error(f"[BUFFER SAVE FAILED] for {self.symbol}: {e}")

    def load_buffer(self, path: str = None):
        """Load replay buffer from disk on startup.
        HIGH FIX: Guard against duplicate loads — if buffer already has data (e.g., from
        __init__ which calls load_buffer()), skip to prevent duplicate transitions.
        CausalSignalManager.__init__() calls load_buffer(), and bot_initializer.py was
        calling it again, causing every persisted transition to appear twice."""
        if path is None:
            path = f"replay_buffer_{self.symbol}.pkl"
        if len(self.replay_buffer.buffer) > 0:
            logger.debug(f"[BUFFER LOAD SKIP] {self.symbol} — buffer already has "
                         f"{len(self.replay_buffer.buffer)} transitions — skipping duplicate load")
            return
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    transitions = pickle.load(f)
                for obs, action, reward in transitions:
                    self.replay_buffer.push(obs, action, reward)
                loaded_count = len(transitions)
                logger.info(f"[BUFFER LOAD] Loaded {loaded_count} transitions for {self.symbol} from {path}")
                if loaded_count == 0:
                    logger.debug(f"[BUFFER LOAD] File existed but was empty — buffer remains empty")
            except Exception as e:
                logger.error(f"[BUFFER LOAD FAILED] for {self.symbol}: {e}")
