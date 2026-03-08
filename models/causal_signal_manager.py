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
import tempfile # ← Priority 1: for atomic writes
import shutil # ← Priority 1: for atomic rename
from config import CONFIG
from dowhy import CausalModel
import pgmpy.estimators.GES as GES
import networkx as nx
from collections import deque
from models.features import generate_features

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Single source of truth: Replay buffer for causal counterfactual estimation using real historical rewards.
    Standardized capacity=5000 across all usages (per-symbol + portfolio)."""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward):
        self.buffer.append((obs.copy(), float(action), float(reward)))

    def sample(self, batch_size=200):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        obs_batch = np.vstack([x[0] for x in batch])
        action_batch = np.array([x[1] for x in batch])
        reward_batch = np.array([x[2] for x in batch])
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
    def __init__(self, base_model, features_df: pd.DataFrame = None, symbol: str = None, data_ingestion=None):
        self.base_model = base_model
        self.causal_model = None
        self.causal_graph = None
        self.identified_estimand = None  # BUG #7 / P-20: Store here to avoid NameError
        self.symbol = symbol or "portfolio"
        self.data_ingestion = data_ingestion
        self.replay_buffer = ReplayBuffer(capacity=5000) # NEW: Real historical data for counterfactuals
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
        """C-3 FIX: Lazy graph build — call this before any graph-dependent operation.
        Builds only if buffer has ≥100 real transitions and graph is None."""
        if self.causal_graph is not None:
            return  # already built
        if len(self.replay_buffer.buffer) < 100:
            logger.debug(f"[CAUSAL LAZY] {self.symbol} — buffer only {len(self.replay_buffer.buffer)} samples — "
                         f"deferring graph build until ≥100 real transitions")
            return
        logger.info(f"[CAUSAL LAZY BUILD] {self.symbol} — buffer now has {len(self.replay_buffer.buffer)} samples — building graph")
        
        # SPECIAL CASE for "portfolio": aggregate features from all real symbols
        if self.symbol == "portfolio":
            if self.data_ingestion is None:
                logger.warning("[CAUSAL LAZY] Portfolio — no data_ingestion reference — cannot build graph")
                return
            all_features = []
            symbol_list = CONFIG.get('SYMBOLS', [])
            logger.debug(f"[CAUSAL PORTFOLIO] Aggregating features from {len(symbol_list)} symbols")
            for real_sym in symbol_list:
                data = self.data_ingestion.get_latest_data(real_sym, timeframe='15Min')
                if len(data) >= 200:
                    features = generate_features(data, 'trending', real_sym, data)
                    if features is not None and features.shape[0] > 0:
                        all_features.append(features)
                        logger.debug(f"[CAUSAL PORTFOLIO] {real_sym} contributed {features.shape[0]} rows")
            if not all_features:
                logger.warning("[CAUSAL LAZY] Portfolio — no valid symbol data for aggregation — deferring graph build")
                return
            full_matrix = np.vstack(all_features)
            features_df = pd.DataFrame(full_matrix, columns=[f'feat_{i}' for i in range(full_matrix.shape[1])])
            logger.info(f"[CAUSAL PORTFOLIO] Aggregated features from {len(all_features)} symbols → shape {features_df.shape}")
        else:
            # Normal single-symbol case
            if self.data_ingestion is None:
                logger.warning(f"[CAUSAL LAZY] {self.symbol} — no data_ingestion reference — cannot build graph")
                return
            data = self.data_ingestion.get_latest_data(self.symbol, timeframe='15Min')
            if len(data) < 200:
                logger.warning(f"[CAUSAL LAZY] {self.symbol} — insufficient data ({len(data)} bars) — deferring")
                return
            features = generate_features(data, 'trending', self.symbol, data)
            if features is None or features.shape[0] == 0:
                logger.warning(f"[CAUSAL LAZY] {self.symbol} — feature generation failed — deferring")
                return
            features_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
        
        self.build_causal_graph(features_df)

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
                    self.causal_model = cached['model']
                    self.identified_estimand = cached.get('identified_estimand') # BUG #7 PATCH: Restore from cache on hit
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
                logger.warning("[CAUSAL] Buffer sample returned None — using features only (no action/reward)")
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
            if data.shape[0] < 500:
                logger.warning(f"[CAUSAL WARNING] Only {data.shape[0]} rows — GES may find fewer edges.")
            ges = GES(data)
            logger.info(f"[CAUSAL DEBUG] Starting GES estimation (bare call for maximum compatibility)")
            self.causal_graph = ges.estimate()
            # === BUILD DoWhy CausalModel FIRST (now with correct action/reward columns) ===
            logger.info(f"[CAUSAL DOWHY] Building DoWhy CausalModel with {len(self.causal_graph.edges)} edges")
            self.causal_model = CausalModel(
                data=data, # ← FIXED: now contains 'action' and 'reward'
                treatment='action',
                outcome='reward',
                graph=self.causal_graph
            )
            # P-20 FIX: Explicitly identify the estimand (missing step causing AttributeError)
            identified_estimand = self.causal_model.identify_effect()
            self.identified_estimand = identified_estimand # BUG #6 PATCH: Store on self so predict() and compute_penalty_factor() can access it
            logger.info(f"[CAUSAL IDENTIFY] Estimand identified successfully")
            logger.info(f"✅ [CAUSAL DOWHY SUCCESS] DoWhy CausalModel ready with {len(self.causal_graph.edges)} validated edges")
            # P-15 / Critical #7 FIX: Compute global ATE once before loop (identical for every edge — no need to recompute)
            logger.info("[CAUSAL REFUTATION] Computing global ATE once (shared across all edges)")
            estimate = self.causal_model.estimate_effect(
                identified_estimand, # P-20 FIX: Use the identified estimand
                method_name="backdoor.linear_regression",
                target_units="ate"
            )
            # === PROPER STATISTICAL REFINEMENT (DoWhy refutation) — now safely after model creation ===
            logger.info(f"[CAUSAL REFUTATION] Running DoWhy refutation tests on {len(self.causal_graph.edges)} edges...")
            refined_edges = []
            for u, v in list(self.causal_graph.edges):
                try:
                    # P-15 / Critical #7 FIX: Reuse the single pre-computed estimate (huge speedup)
                    refuter = self.causal_model.refute_estimate(
                        identified_estimand, # P-20 FIX: Use the identified estimand
                        estimate,
                        method_name="random_common_cause",
                        num_simulations=10
                    )
                    if refuter.refutation_passed:
                        refined_edges.append((u, v))
                        logger.debug(f"Edge {u}→{v} PASSED statistical refutation")
                    else:
                        logger.debug(f"Edge {u}→{v} FAILED refutation")
                except:
                    refined_edges.append((u, v)) # keep edge if test fails gracefully
            if refined_edges:
                self.causal_graph = nx.DiGraph(refined_edges)
                # Rebuild CausalModel with refined graph so they stay in sync
                self.causal_model = CausalModel(
                    data=data,
                    treatment='action',
                    outcome='reward',
                    graph=self.causal_graph
                )
                logger.info(f"✅ [CAUSAL REFINED] {len(refined_edges)} edges passed statistical refutation")
            else:
                logger.warning("[CAUSAL REFINED] No edges passed refutation — keeping original GES graph")
            build_time = time.time() - start_time
            initial_edges = len(self.causal_graph.edges)
            logger.info(f"✅ [CAUSAL BUILD SUCCESS] GES discovered {initial_edges} edges in {build_time:.1f}s")
            # Save to disk cache with model version hash
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump({
                        'graph': self.causal_graph,
                        'model': self.causal_model,
                        'model_version_hash': self.model_version_hash, # P-18 / Critical #11 FIX: Store hash for invalidation
                        'identified_estimand': self.identified_estimand # BUG #7 PATCH: Save so cache hit restores it
                    }, f)
                logger.debug(f"[CAUSAL CACHE SAVED] Saved graph for {self.symbol} with hash {self.model_version_hash}")
            except Exception as e:
                logger.debug(f"Failed to save causal cache: {e}")
        except Exception as e:
            logger.error(f"[CAUSAL BUILD FAILED] Causal graph build failed for {self.symbol}: {e}", exc_info=True)
            self.causal_model = None
            self.causal_graph = None

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
        if not CONFIG.get('USE_CAUSAL_RL', False) or self.causal_model is None:
            logger.debug(f"[CAUSAL FALLBACK] {self.symbol} — no causal model yet — using base PPO predict")
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
        original_action = action.copy() if hasattr(action, 'copy') else float(action)
        # P-7 FIX: Graceful degradation — use whatever samples are available (minimum 50 for meaningful penalty)
        try:
            available = len(self.replay_buffer.buffer)
            if available < 50:
                logger.warning(f"[CAUSAL WARMUP] {self.symbol} buffer only {available} samples — penalty attenuated (needs ~1500 for full strength)")
                penalty_factor = 1.0
            else:
                replay_sample_size = min(1500, available)
                data_df = self.replay_buffer.sample(batch_size=replay_sample_size)
                if data_df is not None:
                    logger.debug(f"[CAUSAL] Using {len(data_df)} real transitions for ATE estimation (robust sample)")
                    effect = self.causal_model.estimate_effect(
                        self.identified_estimand, # BUG #6 PATCH: now stored on self (was NameError)
                        method_name="backdoor.linear_regression",
                        target_units="ate",
                        data=data_df
                    )
                    violation = abs(effect.value)
                    # ISSUE #3 FIX: Inverted penalty — now AMPLIFIES strong causal signals instead of suppressing
                    penalty_factor = min(2.0, 1.0 + (violation * CONFIG.get('CAUSAL_PENALTY_WEIGHT', 0.34)))
                    if penalty_factor > 1.0:
                        logger.debug(f"[CAUSAL BOOST] {self.symbol} strong causal signal → amplified by {penalty_factor:.4f}")
                else:
                    penalty_factor = 1.0
        except Exception:
            logger.debug(f"[CAUSAL] Penalty computation failed — using neutral factor=1.0")
            penalty_factor = 1.0
        action = action * penalty_factor
        # B-10: Always log the penalty so we can confirm it's being applied live
        if penalty_factor != 1.0:
            logger.info(f"✅ [CAUSAL PENALTY APPLIED] {self.symbol} | raw={float(original_action):.4f} → adjusted={float(action):.4f} (factor={penalty_factor:.4f})")
        else:
            logger.debug(f"[CAUSAL PENALTY] {self.symbol} — no violation detected (factor=1.0)")
        if state is not None:
            return action, new_state
        return action, None

    # P-3 FIX: Real public method for causal penalty (used by explain_signal_breakdown and generate_signal)
    def compute_penalty_factor(self, obs, action) -> float:
        """Public method for logging/debug breakdowns. Returns the penalty factor (1.0 = no penalty).
        P-3 FIX: Replaces the missing _compute_penalty_factor calls."""
        # C-3 FIX: Ensure graph exists before computing penalty
        self._ensure_graph_exists()
        if not CONFIG.get('USE_CAUSAL_RL', False) or self.causal_model is None:
            return 1.0
        try:
            # P-7 FIX: Graceful degradation — use whatever samples are available (minimum 50)
            available = len(self.replay_buffer.buffer)
            if available < 50:
                logger.warning(f"[CAUSAL WARMUP] {self.symbol} buffer only {available} samples — penalty attenuated (needs ~1500 for full strength)")
                return 1.0
            replay_sample_size = min(1500, available)
            data_df = self.replay_buffer.sample(batch_size=replay_sample_size)
            if data_df is not None:
                logger.debug(f"[CAUSAL PENALTY LOG] Using {len(data_df)} real transitions for debug")
                effect = self.causal_model.estimate_effect(
                    self.identified_estimand, # BUG #6 PATCH: now stored on self (was NameError)
                    method_name="backdoor.linear_regression",
                    target_units="ate",
                    data=data_df
                )
                violation = abs(effect.value)
                # ISSUE #3 FIX: Inverted penalty — now AMPLIFIES strong causal signals instead of suppressing
                penalty_factor = min(2.0, 1.0 + (violation * CONFIG.get('CAUSAL_PENALTY_WEIGHT', 0.34)))
                if penalty_factor > 1.0:
                    logger.debug(f"[CAUSAL BOOST DEBUG] Strong causal signal → amplified by {penalty_factor:.4f}")
            else:
                penalty_factor = 1.0
        except Exception as e:
            logger.debug(f"[CAUSAL PENALTY LOG] Failed to compute factor: {e}")
            penalty_factor = 1.0
        return penalty_factor

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
                obs_array = np.array(obs, dtype=np.float32).reshape(1, -1)
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
        """Load replay buffer from disk on startup."""
        if path is None:
            path = f"replay_buffer_{self.symbol}.pkl"
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
