"""
metrics.py — Comprehensive Evaluation Metrics
===============================================
Evaluates the trained PPO agent across N episodes and computes:

  • % safe decisions
  • % inverter usage
  • % critical failures  (target: ~0 %)
  • Average reward
  • Reward distribution
  • Battery health score
  • Constraint trigger rate

Usage:
    python metrics.py
    python metrics.py --episodes 50 --steps 24
"""

import argparse
import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO
from env import MicrogridEnv, get_final_decision, SOC_MIN, INVERTER_MAX_W

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate Microgrid PPO metrics.")
parser.add_argument("--model",    default="microgrid_model")
parser.add_argument("--episodes", type=int, default=50,  help="Evaluation episodes")
parser.add_argument("--steps",    type=int, default=24,  help="Steps per episode")
parser.add_argument("--seed",     type=int, default=100, help="Base seed")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    def __init__(self):
        self.episode_rewards    = []
        self.step_rewards       = []
        self.critical_failures  = 0
        self.total_critical_ops = 0
        self.inverter_steps     = 0
        self.grid_only_steps    = 0
        self.both_inverter      = 0
        self.total_steps        = 0
        self.unsafe_soc_steps   = 0
        self.overload_events    = 0
        self.constraint_fixes   = 0
        self.action_counts      = defaultdict(int)
        self.soc_values         = []
        self.outage_correct     = 0
        self.outage_total       = 0

    def record_step(self, state: dict, decision: dict, reward: float):
        self.total_steps      += 1
        self.step_rewards.append(reward)

        # Action counts
        self.action_counts[decision["action_id"]] += 1

        # Critical failure check
        grid_ok     = state["grid_available"] == 1
        soc_ok      = state["battery_soc"] >= SOC_MIN
        crit_src    = decision["critical_source"]
        inv_total   = decision["inverter_total_w"]

        if state["critical_load_w"] > 0:
            self.total_critical_ops += 1
            crit_satisfied = (
                (crit_src == "grid"     and grid_ok) or
                (crit_src == "inverter" and soc_ok and inv_total <= INVERTER_MAX_W)
            )
            if not crit_satisfied:
                self.critical_failures += 1

        # Inverter usage
        uses_inv = (crit_src == "inverter" or decision["noncritical_source"] == "inverter")
        uses_grid = (crit_src == "grid"    or decision["noncritical_source"] == "grid")
        if uses_inv:
            self.inverter_steps += 1
        if uses_inv and not uses_grid:
            self.both_inverter += 1
        if not uses_inv:
            self.grid_only_steps += 1

        # Battery health
        self.soc_values.append(state["battery_soc"])
        if state["battery_soc"] < SOC_MIN:
            self.unsafe_soc_steps += 1

        # Overload / constraint
        if decision["overloaded"]:
            self.overload_events += 1
        if decision["constraint_applied"]:
            self.constraint_fixes += 1

        # Outage handling
        if state["grid_available"] == 0:
            self.outage_total += 1
            if crit_src == "inverter":
                self.outage_correct += 1

    def record_episode(self, ep_reward: float):
        self.episode_rewards.append(ep_reward)

    def report(self) -> dict:
        n  = max(self.total_steps, 1)
        nc = max(self.total_critical_ops, 1)
        no = max(self.outage_total, 1)

        safe_pct    = 100.0 * (1 - self.critical_failures / nc)
        inv_pct     = 100.0 * self.inverter_steps / n
        crit_fail   = 100.0 * self.critical_failures / nc
        avg_reward  = float(np.mean(self.episode_rewards))
        soc_arr     = np.array(self.soc_values)
        outage_acc  = 100.0 * self.outage_correct / no

        return {
            "episodes_evaluated":   len(self.episode_rewards),
            "total_steps":          self.total_steps,
            "avg_episode_reward":   round(avg_reward, 3),
            "std_episode_reward":   round(float(np.std(self.episode_rewards)), 3),
            "min_episode_reward":   round(float(np.min(self.episode_rewards)), 3),
            "max_episode_reward":   round(float(np.max(self.episode_rewards)), 3),
            "pct_safe_decisions":   round(safe_pct, 2),
            "pct_inverter_usage":   round(inv_pct, 2),
            "pct_critical_failures":round(crit_fail, 2),
            "pct_outage_handled":   round(outage_acc, 2),
            "avg_battery_soc":      round(float(np.mean(soc_arr)), 3),
            "min_battery_soc":      round(float(np.min(soc_arr)), 3),
            "pct_unsafe_soc_steps": round(100.0 * self.unsafe_soc_steps / n, 2),
            "overload_events":      self.overload_events,
            "constraint_trigger_%": round(100.0 * self.constraint_fixes / n, 2),
            "action_distribution":  {
                f"action_{k}": v for k, v in sorted(self.action_counts.items())
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# removed obs_to_state


def evaluate(model: PPO) -> dict:
    collector = MetricsCollector()
    env       = MicrogridEnv(max_steps=args.steps, seed=args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0

        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            action     = int(action.item() if hasattr(action, 'item') else action)

            state_dict = env.unwrapped._state.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            decision = info["decision"]
            ep_reward += reward

            collector.record_step(state_dict, decision, reward)

            if terminated or truncated:
                break

        collector.record_episode(ep_reward)

    env.close()
    return collector.report()


def print_report(metrics: dict):
    print("\n" + "═" * 60)
    print("  MICROGRID RL — EVALUATION METRICS REPORT")
    print("═" * 60)

    sections = [
        ("📊 Episode Statistics", [
            ("Episodes evaluated",    metrics["episodes_evaluated"]),
            ("Total steps",           metrics["total_steps"]),
            ("Avg episode reward",    metrics["avg_episode_reward"]),
            ("Std episode reward",    metrics["std_episode_reward"]),
            ("Min / Max reward",
             f"{metrics['min_episode_reward']} / {metrics['max_episode_reward']}"),
        ]),
        ("✅ Reliability Metrics", [
            ("% Safe decisions",      f"{metrics['pct_safe_decisions']} %"),
            ("% Critical failures",   f"{metrics['pct_critical_failures']} %  (target: ~0%)"),
            ("% Outage handled",      f"{metrics['pct_outage_handled']} %"),
        ]),
        ("🔋 Energy & Battery", [
            ("% Inverter usage",      f"{metrics['pct_inverter_usage']} %"),
            ("Avg battery SOC",       f"{metrics['avg_battery_soc']:.3f}"),
            ("Min battery SOC",       f"{metrics['min_battery_soc']:.3f}"),
            ("% Unsafe SOC steps",    f"{metrics['pct_unsafe_soc_steps']} %"),
        ]),
        ("⚙ System Behaviour", [
            ("Overload events",       metrics["overload_events"]),
            ("Constraint trigger %",  f"{metrics['constraint_trigger_%']} %"),
        ]),
    ]

    for section_title, rows in sections:
        print(f"\n  {section_title}")
        print("  " + "─" * 50)
        for label, value in rows:
            print(f"    {label:<30} {value}")

    print(f"\n  🎮 Action Distribution")
    print("  " + "─" * 50)
    for action_label, count in metrics["action_distribution"].items():
        pct = 100.0 * count / max(metrics["total_steps"], 1)
        bar = "█" * int(pct / 2)
        print(f"    {action_label:<12} {count:>6}  ({pct:5.1f}%)  {bar}")

    print("\n" + "═" * 60)

    # ── Rating ────────────────────────────────────────────────────────────
    fail_rate = metrics["pct_critical_failures"]
    safe_rate = metrics["pct_safe_decisions"]
    if fail_rate < 1 and safe_rate > 95:
        grade = "🏆 EXCELLENT"
    elif fail_rate < 5 and safe_rate > 85:
        grade = "✅ GOOD"
    elif fail_rate < 10:
        grade = "⚠️  ACCEPTABLE — needs more training"
    else:
        grade = "❌ POOR — increase training timesteps"

    print(f"  Overall Grade: {grade}")
    print("═" * 60 + "\n")

    return metrics


def main():
    print("=" * 60)
    print("  Microgrid PPO — Metrics Evaluation")
    print("=" * 60)

    try:
        model = PPO.load(args.model)
        print(f"✅  Model loaded: '{args.model}.zip'")
    except FileNotFoundError:
        print(f"❌  '{args.model}.zip' not found. Run train.py first.")
        return

    print(f"    Evaluating {args.episodes} episodes × {args.steps} steps …\n")
    metrics = evaluate(model)
    print_report(metrics)


if __name__ == "__main__":
    main()
