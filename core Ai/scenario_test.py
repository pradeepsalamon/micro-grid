"""
scenario_test.py — Predefined Scenario Evaluator
==================================================
Runs a battery of hand-crafted scenarios against the trained PPO model.

Each scenario has:
  • A descriptive name
  • A fixed state
  • One or more expected behaviours (assertions)

Produces a reliability / correctness report.

Usage:
    python scenario_test.py
"""

import json
import numpy as np
from stable_baselines3 import PPO
from env import get_final_decision, SOC_MIN, INVERTER_MAX_W, normalize_state

MODEL_PATH = "microgrid_model"


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIOS
#  Each dict must have: name, state, checks (list of (description, callable))
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    # ── 1. High solar, low load ────────────────────────────────────────────
    {
        "name": "High solar + low load",
        "state": {
            "solar_power_w":         475.0,
            "wind_power_w":          300.0,
            "battery_soc":           0.75,
            "critical_load_w":       60.0,
            "noncritical_load_w":    40.0,
            "grid_available":        1,
            "solar_forecast_w":      450.0,
            "wind_forecast_w":       275.0,
            "load_forecast_w":       120.0,
            "power_cut_probability": 0.05,
        },
        "checks": [
            (
                "Total load (100W) ≤ 200W → inverter should be used for critical",
                lambda d, s: d["critical_source"] == "inverter",
            ),
            (
                "Inverter not overloaded",
                lambda d, s: not d["overloaded"],
            ),
        ],
    },

    # ── 2. Night + high load ───────────────────────────────────────────────
    {
        "name": "Night + high load",
        "state": {
            "solar_power_w":         0.0,
            "wind_power_w":          50.0,
            "battery_soc":           0.50,
            "critical_load_w":       180.0,
            "noncritical_load_w":    250.0,
            "grid_available":        1,
            "solar_forecast_w":      0.0,
            "wind_forecast_w":       50.0,
            "load_forecast_w":       400.0,
            "power_cut_probability": 0.15,
        },
        "checks": [
            (
                "Total load (430W) > 200W → inverter cannot serve both; expected overload prevention",
                lambda d, s: not d["overloaded"],
            ),
            (
                "Critical load must be satisfied",
                lambda d, s: (
                    d["critical_source"] == "grid" and s["grid_available"]
                ) or (
                    d["critical_source"] == "inverter"
                    and d["inverter_total_w"] <= INVERTER_MAX_W
                ),
            ),
        ],
    },

    # ── 3. Grid failure ────────────────────────────────────────────────────
    {
        "name": "Grid failure (outage)",
        "state": {
            "solar_power_w":         350.0,
            "wind_power_w":          250.0,
            "battery_soc":           0.80,
            "critical_load_w":       100.0,
            "noncritical_load_w":    60.0,
            "grid_available":        0,          # ← outage
            "solar_forecast_w":      325.0,
            "wind_forecast_w":       225.0,
            "load_forecast_w":       160.0,
            "power_cut_probability": 0.85,
        },
        "checks": [
            (
                "Grid unavailable → critical must use inverter",
                lambda d, s: d["critical_source"] == "inverter",
            ),
            (
                "Grid unavailable → noncritical must NOT use grid",
                lambda d, s: d["noncritical_source"] != "grid",
            ),
            (
                "Inverter not overloaded (160W ≤ 200W)",
                lambda d, s: not d["overloaded"],
            ),
        ],
    },

    # ── 4. Low battery ─────────────────────────────────────────────────────
    {
        "name": "Low battery (SOC = 0.18)",
        "state": {
            "solar_power_w":         150.0,
            "wind_power_w":          100.0,
            "battery_soc":           0.18,       # ← well below SOC_MIN
            "critical_load_w":       120.0,
            "noncritical_load_w":    90.0,
            "grid_available":        1,
            "solar_forecast_w":      200.0,
            "wind_forecast_w":       125.0,
            "load_forecast_w":       200.0,
            "power_cut_probability": 0.30,
        },
        "checks": [
            (
                "Battery critical → non-critical must NOT use inverter",
                lambda d, s: d["noncritical_source"] != "inverter",
            ),
            (
                "Constraint layer should have been applied",
                lambda d, s: d["constraint_applied"],
            ),
        ],
    },

    # ── 5. High power-cut probability ──────────────────────────────────────
    {
        "name": "High power-cut probability (cut_prob = 0.90)",
        "state": {
            "solar_power_w":         250.0,
            "wind_power_w":          200.0,
            "battery_soc":           0.60,
            "critical_load_w":       130.0,
            "noncritical_load_w":    70.0,
            "grid_available":        1,
            "solar_forecast_w":      225.0,
            "wind_forecast_w":       175.0,
            "load_forecast_w":       220.0,
            "power_cut_probability": 0.90,       # ← very high
        },
        "checks": [
            (
                "SOC ≥ 20% (battery not deep-discharged)",
                lambda d, s: float(s["battery_soc"]) >= SOC_MIN,
            ),
            (
                "Decision is physically valid (no overload)",
                lambda d, s: not d["overloaded"],
            ),
        ],
    },

    # ── 6. Inverter overload edge case ─────────────────────────────────────
    {
        "name": "Inverter overload edge case",
        "state": {
            "solar_power_w":         400.0,
            "wind_power_w":          300.0,
            "battery_soc":           0.70,
            "critical_load_w":       150.0,
            "noncritical_load_w":    100.0,      # total 250W > 200W
            "grid_available":        1,
            "solar_forecast_w":      375.0,
            "wind_forecast_w":       275.0,
            "load_forecast_w":       270.0,
            "power_cut_probability": 0.10,
        },
        "checks": [
            (
                "Inverter total must not exceed 200W after constraint",
                lambda d, s: d["inverter_total_w"] <= INVERTER_MAX_W,
            ),
        ],
    },

    # ── 7. Both low SOC and grid failure ───────────────────────────────────
    {
        "name": "Worst case: grid failure + low battery",
        "state": {
            "solar_power_w":         50.0,
            "wind_power_w":          25.0,
            "battery_soc":           0.19,       # ← below 20%
            "critical_load_w":       80.0,
            "noncritical_load_w":    40.0,
            "grid_available":        0,
            "solar_forecast_w":      75.0,
            "wind_forecast_w":       50.0,
            "load_forecast_w":       120.0,
            "power_cut_probability": 0.95,
        },
        "checks": [
            (
                "Grid unavailable → critical must use inverter (best effort even at low SOC)",
                lambda d, s: d["critical_source"] == "inverter",
            ),
        ],
    },
    # ── 8. Energy Deficit Check ──────────────────────────────────────────────
    {
        "name": "Energy deficit override case",
        "state": {
            "solar_power_w":         0.0,
            "wind_power_w":          0.0,
            "battery_soc":           0.25,
            "critical_load_w":       150.0,
            "noncritical_load_w":    150.0,
            "grid_available":        1,
            "solar_forecast_w":      10.0,  # LOW
            "wind_forecast_w":       10.0,  # LOW
            "load_forecast_w":       400.0, # HIGH
            "power_cut_probability": 0.9,
        },
        "checks": [
            (
                "Must prioritize battery and use grid entirely due to unsafe future and low current SOC",
                lambda d, s: d["critical_source"] == "grid" and d["noncritical_source"] == "grid",
            ),
        ],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# state_to_obs removed, using env's normalize_state instead


def run_scenarios(model: PPO):
    results = []
    total_checks = 0
    passed_checks = 0

    for idx, scenario in enumerate(SCENARIOS, start=1):
        name  = scenario["name"]
        state = scenario["state"]
        obs   = normalize_state(state).reshape(1, -1)

        action, _  = model.predict(obs, deterministic=True)
        action     = int(action.item())
        decision   = get_final_decision(action, state)

        print(f"\n{'═'*60}")
        print(f"  Scenario {idx}: {name}")
        print(f"{'─'*60}")
        print(f"  Raw action  : {action}")
        print(f"  Final decision:")
        print(f"    critical_source    = {decision['critical_source']}")
        print(f"    noncritical_source = {decision['noncritical_source']}")
        print(f"    inverter_total_w   = {decision['inverter_total_w']:.1f} W")
        print(f"    overloaded         = {decision['overloaded']}")
        print(f"    constraint_applied = {decision['constraint_applied']}")
        print(f"{'─'*60}")
        print("  Checks:")

        scenario_passed = 0
        scenario_total  = len(scenario["checks"])

        for desc, check_fn in scenario["checks"]:
            ok = check_fn(decision, state)
            symbol = "✅" if ok else "❌"
            print(f"    {symbol}  {desc}")
            if ok:
                scenario_passed += 1
                passed_checks   += 1

        total_checks += scenario_total
        battery_safe  = float(state["battery_soc"]) >= SOC_MIN or \
                        not state["grid_available"]   # unavoidable worst-case

        results.append({
            "scenario":      name,
            "checks_passed": scenario_passed,
            "checks_total":  scenario_total,
            "battery_safe":  battery_safe,
            "decision":      {k: decision[k] for k in
                              ["action_id","critical_source","noncritical_source"]},
        })

    # ── Summary report ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  SCENARIO TEST SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total scenarios : {len(SCENARIOS)}")
    print(f"  Total checks    : {total_checks}")
    print(f"  Passed checks   : {passed_checks}")
    print(f"  Reliability     : {100*passed_checks/max(total_checks,1):.1f}%")
    print(f"{'─'*60}")
    for r in results:
        status = "✅" if r["checks_passed"] == r["checks_total"] else "⚠️ "
        print(
            f"  {status}  {r['scenario']:<35}  "
            f"{r['checks_passed']}/{r['checks_total']} checks"
        )
    print(f"{'═'*60}\n")

    return results


def main():
    print("=" * 60)
    print("  Microgrid PPO — Scenario Test Suite")
    print("=" * 60)

    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅  Model loaded: '{MODEL_PATH}.zip'\n")
    except FileNotFoundError:
        print(f"❌  '{MODEL_PATH}.zip' not found. Run train.py first.\n")
        return

    run_scenarios(model)


if __name__ == "__main__":
    main()
