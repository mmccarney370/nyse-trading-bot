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

def cosine_annealing_schedule(progress_remaining: float) -> float:
    initial_lr = CONFIG.get('PPO_LEARNING_RATE', 3e-4)
    min_lr = CONFIG.get('PPO_LEARNING_RATE_MIN', 1e-6)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * (1 - progress_remaining)))

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
        ppo_path = os.path.join(MODEL_DIR, f"ppo_{key}.zip")
        vec_path = os.path.join(MODEL_DIR, f"vecnorm_{key}.pkl")

    try:
        model.save(ppo_path)
        joblib.dump(vec_norm, vec_path)
        logger.info(f"Saved PPO model and VecNormalize for {key}: {ppo_path}, {vec_path}")
    except Exception as e:
        logger.error(f"Failed to save model for {key}: {e}")

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
        # BUG-09 FIX: Detect whether saved model is recurrent or not
        # Try to peek at policy class name without full load
        import zipfile
        with zipfile.ZipFile(ppo_path, 'r') as zf:
            if 'policy.pth' in zf.namelist():
                # Rough heuristic: check if 'lstm', 'recurrent', or 'gtrxl' appears in policy file names or metadata
                is_recurrent_hint = any('lstm' in name.lower() or 'recurrent' in name.lower() or 'gtrxl' in name.lower() for name in zf.namelist())
            else:
                is_recurrent_hint = False

        # Combine hint with config fallback
        use_recurrent = (
            is_recurrent_hint or
            (key == "portfolio" and CONFIG.get('PORTFOLIO_USE_RECURRENT', CONFIG.get('PPO_RECURRENT', True)))
        )

        if use_recurrent and recurrent_available:
            model_class = CustomRecurrentPPO
            logger.info(f"Loading {key} as recurrent (CustomRecurrentPPO) — detected hint or config flag")
        else:
            model_class = CustomPPO
            logger.info(f"Loading {key} as non-recurrent (CustomPPO)")

        model = model_class.load(ppo_path)
        vec_norm = joblib.load(vec_path) if os.path.exists(vec_path) else None

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
    env = DummyVecEnv([lambda: Monitor(TradingEnv(trainer.data_ingestion, symbol))])
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Load existing if incremental
    if incremental:
        existing_model, existing_norm = load_ppo_model(trainer, symbol)
        if existing_model:
            model = existing_model
            vec_env = existing_norm or vec_env
            vec_env.training = True
            logger.info(f"Loaded existing model for incremental training on {symbol}")
        else:
            incremental = False # ← FIXED: Set incremental=False BEFORE setting LR schedule
    # Now set LR schedule based on final incremental decision
    online_lr = CONFIG.get('PPO_ONLINE_LEARNING_RATE', 5e-5)
    lr_schedule = cosine_annealing_schedule if not incremental else lambda _: online_lr
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
            if hasattr(model, 'set_env'):
                model.set_env(vec_env)
            logger.info(f"Reusing loaded RecurrentPPO model for incremental training on {symbol}")
        else:
            model = ppo_cls(
                policy=policy_cls,
                env=vec_env,
                verbose=1,
                device=device,
                tensorboard_log="./ppo_tensorboard/" if CONFIG.get('USE_TENSORBOARD', False) else None,
                learning_rate=lr_schedule,
                n_steps=CONFIG.get('PPO_N_STEPS', 2048),
                batch_size=CONFIG.get('PPO_BATCH_SIZE', 256),
                n_epochs=CONFIG.get('PPO_N_EPOCHS', 4),
                gamma=CONFIG.get('PPO_GAMMA', 0.97),
                gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.93),
                clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.18),
                ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.03),
                vf_coef=CONFIG.get('vf_coef', 1.0),
                max_grad_norm=CONFIG.get('PPO_MAX_GRAD_NORM', 0.4),
            )
        logger.info(f"Training RecurrentPPO ({'GTrXL' if use_custom_gtrxl else 'LSTM'}) for {symbol} ({timesteps} timesteps)")
        model.learn(total_timesteps=timesteps, callback=nan_callback, reset_num_timesteps=not incremental)
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
                ent_coef=CONFIG.get('PPO_ENTROPY_COEFF', 0.03),
                gamma=CONFIG.get('PPO_GAMMA', 0.97),
                gae_lambda=CONFIG.get('PPO_GAE_LAMBDA', 0.93),
                clip_range=CONFIG.get('PPO_CLIP_RANGE', 0.18),
                vf_coef=CONFIG.get('vf_coef', 1.0),
                max_grad_norm=CONFIG.get('PPO_MAX_GRAD_NORM', 0.4)
            )
        nan_callback = NaNStopCallback()
        logger.info(f"Training PPO for {symbol} ({timesteps} timesteps)")
        model.learn(total_timesteps=timesteps, callback=[nan_callback], reset_num_timesteps=not incremental)
    logger.info(f"PPO training completed for {symbol}")
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
