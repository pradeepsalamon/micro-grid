"""
train.py — PPO Training Script
================================
Trains a Proximal Policy Optimization (PPO) agent on the MicrogridEnv
using Stable Baselines3.

Usage:
    python train.py

Output:
    microgrid_model.zip  — saved policy checkpoint
    logs/                — TensorBoard training logs
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from env import MicrogridEnv

# ─────────────────────────────── reproducibility ───────────────────────────
SEED            = 42
TOTAL_TIMESTEPS = 600_000
N_ENVS          = 4           # parallel environments
LOG_DIR         = "logs/"
CHECKPOINT_DIR  = "checkpoints/"

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ──────────────────────────── progress callback ────────────────────────────
class ProgressCallback(BaseCallback):
    """Prints episode reward stats every N steps."""

    def __init__(self, print_every: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.print_every = print_every
        self._ep_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.print_every == 0 and self._ep_rewards:
            mean_r = np.mean(self._ep_rewards[-50:])
            print(
                f"  [Step {self.num_timesteps:>7}]  "
                f"Mean episode reward (last 50): {mean_r:.2f}"
            )
        return True


# ──────────────────────────── environment factory ──────────────────────────

def make_env(seed_offset: int = 0):
    def _init():
        env = MicrogridEnv(max_steps=24, seed=SEED + seed_offset)
        env = Monitor(env)
        return env
    return _init


# ──────────────────────────── training ────────────────────────────────────

def train():
    print("=" * 60)
    print("  Microgrid PPO Training")
    print("=" * 60)

    # Vectorised envs for parallel rollout collection
    train_env = make_vec_env(
        lambda: Monitor(MicrogridEnv(max_steps=24, seed=SEED)),
        n_envs=N_ENVS,
        seed=SEED,
    )

    # Separate evaluation environment
    eval_env = Monitor(MicrogridEnv(max_steps=24, seed=SEED + 999))

    # ── PPO hyperparameters ────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 2e-4,
        n_steps         = 4096,       # steps per env per update
        batch_size      = 128,
        n_epochs        = 10,
        gamma           = 0.995,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.03,      # entropy bonus to encourage exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 0,
        seed            = SEED,
        tensorboard_log = LOG_DIR,
        policy_kwargs   = dict(net_arch=[128, 128]),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = CHECKPOINT_DIR,
        log_path             = LOG_DIR,
        eval_freq            = 5_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "microgrid_ckpt",
        verbose     = 0,
    )

    progress_cb = ProgressCallback(print_every=10_000)

    # ── Learn ─────────────────────────────────────────────────────────────
    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps with {N_ENVS} parallel envs …\n")
    model.learn(
        total_timesteps   = TOTAL_TIMESTEPS,
        callback          = [eval_callback, checkpoint_cb, progress_cb],
        tb_log_name       = "PPO_microgrid",
        reset_num_timesteps = True,
        progress_bar      = True,
    )

    # ── Save final model ──────────────────────────────────────────────────
    model.save("microgrid_model")
    print("\n✅  Training complete.  Model saved →  microgrid_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
