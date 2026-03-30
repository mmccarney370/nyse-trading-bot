# __main__.py
# → Placed before ANY other import so pgmpy/GES never sees tqdm
import sys
# === ULTRA-EARLY GLOBAL TQDM SUPPRESSION — MUST BE BEFORE ANY OTHER IMPORT ===
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["LIGHTNING_TIPS"] = "0"
import asyncio
import warnings
import logging
import glob
import shutil
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(override=True) # Loads .env into os.environ
# ─── SUPPRESS NOISY WARNINGS & LOGGING ────────────────────────────────────────────────
# 1. NumPy harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy._core._methods", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy._core._methods", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy._core._methods", message="invalid value encountered in scalar divide")
# 2. PyTorch Lightning flooding
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.trainer")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.loops")
warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
warnings.filterwarnings("ignore", message="ReduceLROnPlateau conditioned on metric val_loss which is not available")
warnings.filterwarnings("ignore", message="GPU available but not used")
warnings.filterwarnings("ignore", message="The 'train_dataloader' does not have many workers")
warnings.filterwarnings("ignore", message="The 'predict_dataloader' does not have many workers")
warnings.filterwarnings("ignore", message="Checkpoint directory.*exists and is not empty")
# 3. LightGBM redundant parameter warning
warnings.filterwarnings("ignore",
                        message="Found `num_iterations` in params. Will use it instead of argument",
                        module="lightgbm")
# 4. Set Lightning root logger to ERROR (suppresses GPU/TPU status, model summary noise)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
# 5. Quiet other noisy libs
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# ─── TENSOR CORE OPTIMIZATION ─────────────────────────────────────────────
import torch
torch.set_float32_matmul_precision('high')
# ─── YOUR ORIGINAL IMPORTS & CODE ─────────────────────────────────────────────────────
from utils.log_setup import setup_logging
from bot import TradingBot
from config import CONFIG
async def main():
    setup_logging()
    # ─── RESTORE FILE LOGGING TO nyse_bot.log ─────────────────────────────────────
    # (REMOVED — this block was duplicating the file handler already added in setup_logging())
    # NEW: Automatic cleanup of lightning_logs/ folder
    lightning_logs_dir = "lightning_logs"
    if os.path.exists(lightning_logs_dir):
        try:
            shutil.rmtree(lightning_logs_dir)
            print(f"[CLEANUP] Deleted lightning_logs/ folder (prevents file flooding)")
        except Exception as e:
            print(f"[CLEANUP] Failed to delete lightning_logs/: {e}")
    # NEW: Startup cleanup for old TFT cache files
    tft_cache_dir = "ppo_checkpoints/tft_cache"
    if os.path.exists(tft_cache_dir):
        cutoff = datetime.now() - timedelta(days=7)
        for f in glob.glob(os.path.join(tft_cache_dir, "*.pkl")):
            try:
                if datetime.fromtimestamp(os.path.getmtime(f)) < cutoff:
                    os.remove(f)
                    print(f"[TFT CACHE CLEANUP] Deleted old cache file: {f}")
            except Exception as e:
                print(f"[TFT CACHE CLEANUP] Failed to delete {f}: {e}")
    # Cleanup old tensorboard logs (keep only the latest run per directory)
    for tb_dir in glob.glob("ppo_tensorboard/*/"):
        runs = sorted(glob.glob(os.path.join(tb_dir, "RecurrentPPO_*")), key=os.path.getmtime)
        if len(runs) > 1:
            for old_run in runs[:-1]:
                try:
                    shutil.rmtree(old_run)
                    print(f"[TB CLEANUP] Deleted old tensorboard run: {old_run}")
                except Exception as e:
                    print(f"[TB CLEANUP] Failed to delete {old_run}: {e}")
    # Cleanup old log files (keep last 7 days)
    log_cutoff = datetime.now() - timedelta(days=7)
    for log_file in glob.glob("logs/*.log*"):
        try:
            if datetime.fromtimestamp(os.path.getmtime(log_file)) < log_cutoff:
                os.remove(log_file)
                print(f"[LOG CLEANUP] Deleted old log: {log_file}")
        except Exception:
            pass
    # Extra memory cleanup at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("[STARTUP] Cleanup completed + GPU/RAM cleared")
    bot = TradingBot(CONFIG)
    await bot.run()
if __name__ == "__main__":
    asyncio.run(main())
