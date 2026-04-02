"""
test.py — Model Inference on Random Episodes
=============================================
Loads the saved PPO model and runs it on randomly generated microgrid states.
Prints: action taken, reward received, and the final constrained decision.

Usage:
    python test.py
    python test.py --episodes 5 --steps 24
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from env import MicrogridEnv, get_final_decision
import json

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test the trained Microgrid PPO agent.")
parser.add_argument("--model",    default="microgrid_model", help="Path to saved model (no .zip)")
parser.add_argument("--episodes", type=int, default=3,       help="Number of test episodes")
parser.add_argument("--steps",    type=int, default=24,      help="Steps per episode")
parser.add_argument("--seed",     type=int, default=0,       help="Random seed for test env")
args = parser.parse_args()


def run_test():
    print("=" * 65)
    print("  Microgrid PPO — Inference Test")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────────
    try:
        model = PPO.load(args.model)
        print(f"✅  Model loaded from '{args.model}.zip'\n")
    except FileNotFoundError:
        print(f"❌  Model file '{args.model}.zip' not found. Run train.py first.")
        return

    env = MicrogridEnv(max_steps=args.steps, seed=args.seed)

    total_rewards   = []
    total_failures  = 0
    inverter_uses   = 0
    total_decisions = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0

        print(f"{'─'*65}")
        print(f"  Episode {ep}")
        print(f"{'─'*65}")
        print(f"{'Step':>4}  {'SOC':>5}  {'Crit→':>10}  {'NCrit→':>10}  "
              f"{'Inv(W)':>7}  {'Reward':>8}")
        print(f"{'─'*65}")

        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            action     = int(action)

            # Grab pre-step state
            state_dict = env.unwrapped._state.copy()

            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            decision = info["decision"]
            ep_reward     += reward
            total_decisions += 1

            # Track failures & inverter usage
            crit_ok = (
                decision["critical_source"] == "grid" and state_dict["grid_available"]
            ) or (
                decision["critical_source"] == "inverter"
                and state_dict["battery_soc"] >= 0.20
                and decision["inverter_total_w"] <= 200.0
            )
            if not crit_ok:
                total_failures += 1
            if decision["critical_source"] == "inverter" or \
               decision["noncritical_source"] == "inverter":
                inverter_uses += 1

            print(
                f"{step+1:>4}  "
                f"{state_dict['battery_soc']:.3f}  "
                f"{decision['critical_source']:>10}  "
                f"{decision['noncritical_source']:>10}  "
                f"{decision['inverter_total_w']:>7.1f}  "
                f"{reward:>+8.2f}"
                + (" ⚠ overload" if decision["overloaded"] else "")
                + (" 🔒 constrained" if decision["constraint_applied"] else "")
            )

            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        print(f"\n  Episode {ep} total reward: {ep_reward:.2f}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Test Summary")
    print("=" * 65)
    print(f"  Episodes        : {args.episodes}")
    print(f"  Avg reward      : {np.mean(total_rewards):.2f}")
    print(f"  Min / Max reward: {np.min(total_rewards):.2f} / {np.max(total_rewards):.2f}")
    print(f"  Critical failures: {total_failures} / {total_decisions} "
          f"({100*total_failures/max(total_decisions,1):.1f}%)")
    print(f"  Inverter usage  : {100*inverter_uses/max(total_decisions,1):.1f}%")
    print("=" * 65)

    env.close()


if __name__ == "__main__":
    run_test()
