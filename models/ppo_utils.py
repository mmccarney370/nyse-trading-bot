# models/ppo_utils.py
# PPO-related utilities: training, scheduling, save/load, callback, online update
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import gymnasium as gym # Critical import for type hints
import logging
from config import CONFIG
from models.env import TradingEnv
from .policies import AuxMlpPolicy, AuxRecurrentPolicy, AuxGTrXLPolicy, CustomPPO, CustomRecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

logger = logging.getLogger(__name__)

# Directory for saved models (repeated for self-contained file)
MODEL_DIR = "ppo_checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    from sb3_contrib import RecurrentPPO
    recurrent_available = True
    logger.info("RecurrentPPO available — recurrent training enabled")
except ImportError:
    recurrent_available = False
    logger.warning("sb3-contrib not available — falling back to non-recurrent PPO")

class NaNStopCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for param in self.model.policy.parameters():
            if torch.isnan(param).any():
                logger.warning("NaN detected in model parameters — stopping training for this symbol")
                return False
        return True

class AuxVolatilityCallback(BaseCallback):
    """CRIT-12 FIX: Captures volatility_target from env infos during rollout collection
    and stores them in model._aux_vol_targets side-channel. This is necessary because
    SB3's RolloutBuffer/RecurrentRolloutBuffer do NOT store infos, making the aux
    volatility head a complete no-op without this callback."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        if not hasattr(self.model, '_aux_vol_targets'):
            self.model._aux_vol_targets = []
        self.model._aux_vol_targets.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            vol_target = info.get('volatility_target')
            if vol_target is not None:
                self.model._aux_vol_targets.append(float(vol_target))
        return True

def make_cosine_annealing_schedule() -> callable:
    """Linear warmup then constant LR schedule.
    - First PPO_LR_WARMUP_FRAC of training: ramp from min_lr to initial_lr
    - Remaining training: hold at initial_lr
    Replaces cosine annealing which decayed too fast (LR at floor by 50% through training,
    leaving no learning capacity for the second half)."""
    initial_lr = CONFIG.get('PPO_LEARNING_RATE', 2.5e-5)
    min_lr = CONFIG.get('PPO_LEARNING_RATE_MIN', 1.5e-5)
    warmup_frac = CONFIG.get('PPO_LR_WARMUP_FRAC', 0.05)
    def _schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        progress = 1.0 - progress_remaining
        if progress < warmup_frac:
            # Linear warmup: min_lr → initial_lr
            return min_lr + (initial_lr - min_lr) * (progress / warmup_frac)
        # Constant after warmup
        return initial_lr
    return _schedule

# Lazy wrapper: defers schedule creation until first call, so CONFIG is fully loaded.
class _LazyCosineSchedule:
    """Wrapper that creates the LR schedule on first call."""
    def __init__(self):
        self._schedule = None
    def __call__(self, progress_remaining: float) -> float:
        if self._schedule is None:
            self._schedule = make_cosine_annealing_schedule()
        return self._schedule(progress_remaining)

cosine_annealing_schedule = _LazyCosineSchedule()

def save_ppo_model(trainer, key: str):
    """
    Save PPO model and VecNormalize.
    - For regular symbols: ppo_checkpoints/ppo_{key}.zip + vecnorm_{key}.pkl
    - For key == "portfolio": ppo_checkpoints/portfolio/ppo_model.zip + vecnorm.pkl
    """
    if key == "portfolio":
        if trainer.portfolio_ppo_model is None or trainer.portfolio_vec_norm is None:
            logger.warning("No portfolio PPO model or VecNorm to save")
            return
        model = trainer.portfolio_ppo_model
        vec_norm = trainer.portfolio_vec_norm
        port_dir = os.path.join(MODEL_DIR, "portfolio")
        os.makedirs(port_dir, exist_ok=True)
        ppo_path = os.path.join(port_dir, "ppo_model.zip")
        vec_path = os.path.join(port_dir, "vecnorm.pkl")
    else:
        if key not in trainer.ppo_models or key not in trainer.vec_norms:
            logger.warning(f"No PPO model or VecNorm to save for {key}")
            return
        model = trainer.ppo_models[key]
        vec_norm = trainer.vec_norms[key]
        if model is None or vec_norm is None:
            logger.warning(f"PPO model or VecNorm is None for {key} — skipping save")
            return
        ppo_path = os.path.join(MODEL_DIR, f"ppo_{key}.zip")
        vec_path = os.path.join(MODEL_DIR, f"vecnorm_{key}.pkl")

    # FIX #30: Atomic save — write to temp files first, then rename both
    tmp_ppo_path = None
    tmp_vec_path = None
    try:
        import tempfile
        ppo_dir = os.path.dirname(ppo_path)
        vec_dir = os.path.dirname(vec_path)
        # Save model to temp file
        with tempfile.NamedTemporaryFile(dir=ppo_dir, suffix='.zip', delete=False) as tmp_ppo:
            tmp_ppo_path = tmp_ppo.name
        model.save(tmp_ppo_path)
        # Save vec_norm to temp file
        with tempfile.NamedTemporaryFile(dir=vec_dir, suffix='.pkl', delete=False) as tmp_vec:
            tmp_vec_path = tmp_vec.name
        joblib.dump(vec_norm, tmp_vec_path)
        # Atomic rename both
        import shutil
        shutil.move(tmp_ppo_path, ppo_path)
        shutil.move(tmp_vec_path, vec_path)
        logger.info(f"Saved PPO model and VecNormalize for {key}: {ppo_path}, {vec_path}")
    except Exception as e:
        logger.error(f"Failed to save model for {key}: {e}")
        # Clean up temp files on failure
        for tmp in [tmp_ppo_path, tmp_vec_path]:
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

def load_ppo_model(trainer, key: str):
    """
    Load PPO model and VecNormalize.
    - Returns (model, vec_norm) tuple
    - For key == "portfolio": loads from ppo_checkpoints/portfolio/
    - Sets trainer.portfolio_ppo_model / trainer.portfolio_vec_norm if applicable
    """
    if key == "portfolio":
        port_dir = os.path.join(MODEL_DIR, "portfolio")
        ppo_path = os.path.join(port_dir, "ppo_model.zip")
        vec_path = os.path.join(port_dir, "vecnorm.pkl")
    else:
        ppo_path = os.path.join(MODEL_DIR, f"ppo_{key}.zip")
        vec_path = os.path.join(MODEL_DIR, f"vecnorm_{key}.pkl")

    if not os.path.exists(ppo_path):
        logger.info(f"No saved PPO model found for {key}")
        return None, None

    try:
        # Detect whether saved model is recurrent by reading the `data` dict inside the zip
        # which contains 'policy_class' as a pickled reference
        import zipfile, io
        is_recurrent_hint = False
        with zipfile.ZipFile(ppo_path, 'r') as zf:
            if 'data' in zf.namelist():
                try:
                    import json as _json
                    with zf.open('data') as data_file:
                        raw = data_file.read()
                        # SB3 stores data as a pickle or JSON — try both
                        try:
                            data_dict = _json.loads(raw)
                        except Exception:
                            import pickle as _pkl
                            # SECURITY NOTE (FIX #52): pickle.loads can execute arbitrary code.
                            # Only load from trusted model files saved by this application.
                            data_dict = _pkl.loads(raw)
                        policy_cls_raw = data_dict.get('policy_class', '')
                        # Extract just the class name — SB3 may store a full dict with serialized metadata
                        if isinstance(policy_cls_raw, dict):
                            policy_cls_name = policy_cls_raw.get('__module__', '') + '.' + policy_cls_raw.get('__qualname__', policy_cls_raw.get('__name__', ''))
                        else:
                            policy_cls_name = str(policy_cls_raw)
                        is_recurrent_hint = any(kw in policy_cls_name.lower()
                                                for kw in ('recurrent', 'lstm', 'gtrxl', 'auxgtrxl', 'auxrecurrent'))
                        if is_recurrent_hint:
                            logger.info(f"Detected recurrent policy class: {policy_cls_name}")
                except Exception as e:
                    logger.debug(f"Could not read policy class from zip data: {e}")

        # Combine hint with config fallback
        use_recurrent = (
            is_recurrent_hint or
            CONFIG.get('PPO_RECURRENT', True)
        )

        if use_recurrent and recurrent_available:
            model_class = CustomRecurrentPPO
            logger.info(f"Loading {key} as recurrent (CustomRecurrentPPO) — detected hint or config flag")
        else:
            model_class = CustomPPO
            logger.info(f"Loading {key} as non-recurrent (CustomPPO)")

        model = model_class.load(ppo_path)
        vec_norm = joblib.load(vec_path) if os.path.exists(vec_path) else None

        # Dimension check: discard old VecNormalize if obs space dimensions don't match.
        # After feature changes, a stale VecNormalize has wrong running mean/var dimensions,
        # which causes shape mismatch errors during normalization.
        if vec_norm is not None and hasattr(model, 'observation_space'):
            expected_dim = model.observation_space.shape[0]
            try:
                vn_dim = vec_norm.observation_space.shape[0]
                if vn_dim != expected_dim:
                    logger.warning(f"VecNormalize obs dim mismatch for {key}: saved={vn_dim}, expected={expected_dim} — discarding old VecNormalize")
                    vec_norm = None
            except Exception:
                logger.warning(f"Could not verify VecNormalize dimensions for {key} — discarding to be safe")
                vec_norm = None

        logger.info(f"Loaded PPO model and VecNormalize for {key}")
        if key == "portfolio":
            trainer.portfolio_ppo_model = model
            trainer.portfolio_vec_norm = vec_norm
        else:
            trainer.ppo_models[key] = model
            trainer.vec_norms[key] = vec_norm

        return model, vec_norm

    except Exception as e:
        logger.error(f"Failed to load model for {key}: {e}")
        return None, None

def train_ppo(trainer, symbol: str, data: pd.DataFrame, incremental: bool = False):
    """
    Train or incrementally update PPO (recurrent GTrXL preferred, fallback LSTM or MLP)
    """
    if len(data) < 500:
        logger.warning(f"Insufficient data ({len(data)} bars) for {symbol} — skipping PPO training")
        return
    timesteps = CONFIG.get('PPO_ONLINE_UPDATE_TIMESTEPS' if incremental else 'PPO_TIMESTEPS', 100000)
    use_recurrent = CONFIG.get('PPO_RECURRENT', True) and recurrent_available
    use_custom_gtrxl = CONFIG.get('USE_CUSTOM_GTRXL', True) and use_recurrent
    use_aux = CONFIG.get('PPO_AUX_TASK', True)
    device = CONFIG.get('DEVICE', 'cpu')
    # Create environment (Monitor wrapper enables episode stat tracking)
    # FIX #15: Validate the data that TradingEnv will actually use (from data_ingestion),
    # not just the passed-in `data` parameter. The length check at the top validates `data`,
    # but TradingEnv.reset() calls data_ingestion.get_latest_data() independently.
    actual_data = trainer.data_ingestion.get_latest_data(symbol)
    if len(actual_data) < 500:
        logger.warning(f"data_ingestion has insufficient data ({len(actual_data)} bars) for {symbol} — skipping PPO training")
        return
    env = DummyVecEnv([lambda: Monitor(TradingEnv(trainer.data_ingestion, symbol))])
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Load existing if incremental
    if incremental:
        existing_model, existing_norm = load_ppo_model(trainer, symbol)
        if existing_model:
            model = existing_model
            # H19 FIX: Keep fresh VecNormalize (new data distribution) instead of overwriting
            # with stale loaded one. The model weights are loaded but normalization stats
            # will adapt to the current data during training (vec_env.training = True).
            vec_env.training = True
            logger.info(f"Loaded existing model for incremental training on {symbol} (fresh VecNormalize kept)")
        else:
            incremental = False # ← FIXED: Set incremental=False BEFORE setting LR schedule
    # Now set LR schedule based on final incremental decision
    online_lr = CONFIG.get('PPO_ONLINE_LEARNING_RATE', 5e-5)
    # HIGH FIX: Create a fresh schedule per training run to capture current CONFIG values
    # at training start, but remain stable throughout the run (no mid-training jumps)
    lr_schedule = make_cosine_annealing_schedule() if not incremental else lambda _: online_lr
    if use_recurrent:
        nan_callback = NaNStopCallback()
        if use_custom_gtrxl:
            policy_cls = AuxGTrXLPolicy
            ppo_cls = CustomRecurrentPPO
            logger.info(f"{'Incremental' if incremental else 'Initial'} RecurrentPPO training with custom GTrXL for {symbol}")
        else:
            policy_cls = AuxRecurrentPolicy if use_aux else RecurrentActorCriticPolicy
            ppo_cls = CustomRecurrentPPO
            logger.info(f"{'Incremental' if incremental else 'Initial'} RecurrentPPO training with LSTM for {symbol}")
        # P-2 FIX: Skip fresh instantiation when incremental and model was loaded (prevents overwriting learned weights)
        if incremental and 'existing_model' in locals() and existing_model is not None:
            model = existing_model
            # FIX #22: set_env rebinds model to fresh VecNormalize so incremental training
            # uses current normalization stats, not stale ones from the loaded checkpoint.
            if hasattr(model, 'set_env'):
                model.set_env(vec_env)
            logger.info(f"Reusing loaded RecurrentPPO model for incremental training on {symbol}")
        else:
            # CRIT-11 FIX: Pass policy_kwargs with GTrXL/LSTM config from CONFIG
            # Without this, hidden size, layers, heads, memory length all default to hardcoded values
            recurrent_policy_kwargs = dict(
                lstm_hidden_size=CONFIG.get('GTRXL_HIDDEN_SIZE', 256),
                n_lstm_layers=CONFIG.get('GTRXL_NUM_LAYERS', 2),
            )
            if not use_custom_gtrxl:
                # LSTM-specific: use standard net_arch for value function
                recurrent_policy_kwargs['net_arch'] = CONFIG.get('PPO_VF_NET_ARCH', [512, 256])
            model = ppo_cls(
                policy=policy_cls,
                env=vec_env,
                verbose=1,
                device=device,
                policy_kwargs=recurrent_policy_kwargs,
                tensorboard_log="./ppo_tensorboard/" if CONFIG.get('USE_TENSORBOARD', False) else None,
                learning_rate=lr_schedule,
                n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                n_epochs=CONFIG.get('PPO_N_EPOCHS', 4),
                gamma=CONFIG.get('PPO_GAMMA', 0.97),
                gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.93),
                clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.18),
                ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.03),
                vf_coef=CONFIG.get('VF_COEF', 1.0),
                max_grad_norm=CONFIG.get('PPO_MAX_GRAD_NORM', 0.4),
            )
        logger.info(f"Training RecurrentPPO ({'GTrXL' if use_custom_gtrxl else 'LSTM'}) for {symbol} ({timesteps} timesteps)")
        # CRIT-12 FIX: Add AuxVolatilityCallback to capture vol targets from env infos
        aux_callback = AuxVolatilityCallback() if use_aux else None
        callbacks = [nan_callback] + ([aux_callback] if aux_callback else [])
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=not incremental)
    else:
        logger.info(f"{'Incremental' if incremental else 'Initial'} standard PPO training for {symbol}")
        policy_kwargs = dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.Tanh
        )
        # Use string "MlpPolicy" for built-in fallback (compatible across SB3 versions)
        policy_cls = AuxMlpPolicy if use_aux else "MlpPolicy"
        from stable_baselines3 import PPO
        ppo_cls = CustomPPO if use_aux else PPO
        # P-2 FIX: Skip fresh instantiation when incremental and model was loaded (prevents overwriting learned weights)
        if incremental and 'existing_model' in locals() and existing_model is not None:
            model = existing_model
            if hasattr(model, 'set_env'):
                model.set_env(vec_env)
            logger.info(f"Reusing loaded PPO model for incremental training on {symbol}")
        else:
            model = ppo_cls(
                policy=policy_cls,
                env=vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
                tensorboard_log="./ppo_tensorboard/" if CONFIG.get('USE_TENSORBOARD', False) else None,
                learning_rate=lr_schedule,
                n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                n_epochs=CONFIG.get('PPO_N_EPOCHS', 4),
                gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.93),
                ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.03),
                gamma=CONFIG.get('PPO_GAMMA', 0.97),
                clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.18),
                vf_coef=CONFIG.get('VF_COEF', 1.0),
                max_grad_norm=CONFIG.get('PPO_MAX_GRAD_NORM', 0.4)
            )
        nan_callback = NaNStopCallback()
        # CRIT-12 FIX: Add AuxVolatilityCallback to capture vol targets from env infos
        aux_callback = AuxVolatilityCallback() if use_aux else None
        callbacks = [nan_callback] + ([aux_callback] if aux_callback else [])
        logger.info(f"Training PPO for {symbol} ({timesteps} timesteps)")
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=not incremental)
    logger.info(f"PPO training completed for {symbol}")
    # FIX: Set VecNormalize to inference mode (freeze running stats) before saving/using for inference
    vec_env.training = False
    vec_env.norm_reward = False
    # HIGH-27 FIX: Close old VecNormalize env if it exists (prevents memory leak on online updates)
    old_vec = trainer.vec_norms.get(symbol)
    if old_vec is not None and old_vec is not vec_env:
        try:
            old_vec.close()
            logger.debug(f"[CLEANUP] Closed old VecNormalize for {symbol}")
        except Exception:
            pass
    trainer.ppo_models[symbol] = model
    trainer.vec_norms[symbol] = vec_env
    save_ppo_model(trainer, symbol)

def update_model_weights(trainer, symbol: str, recent_data: pd.DataFrame = None):
    """Fully online/adaptive PPO update — call periodically from bot.py"""
    if symbol not in trainer.ppo_models or trainer.ppo_models[symbol] is None:
        logger.info(f"No existing PPO model for {symbol} — skipping online update")
        return
    full_data = trainer.data_ingestion.get_latest_data(symbol)
    if recent_data is not None:
        full_data = pd.concat([full_data, recent_data]).drop_duplicates().sort_index()
    if len(full_data) < 500:
        logger.warning(f"Insufficient data for online PPO update on {symbol}")
        return
    logger.info(f"Starting online PPO adaptation for {symbol} on {len(full_data)} bars")
    train_ppo(trainer, symbol, full_data, incremental=True)
