"""Smoke test for env.py — run to verify the environment works."""
from env import MicrogridEnv, get_final_decision
import numpy as np

env = MicrogridEnv(seed=42)
obs, _ = env.reset()
print("Obs shape:", obs.shape)

actions_map = {0: "G/G", 1: "G/I", 2: "I/G", 3: "I/I"}

for action in [0, 1, 2, 3]:
    env.reset()
    obs2, r, _, _, info = env.step(action)
    d = info["decision"]
    print(
        f"  Action {action} ({actions_map[action]}) "
        f"-> crit={d['critical_source']}, ncrit={d['noncritical_source']}, "
        f"inv_w={d['inverter_total_w']:.1f} W, reward={r:.2f}"
    )

print("\nEnvironment OK ✅")
env.close()
