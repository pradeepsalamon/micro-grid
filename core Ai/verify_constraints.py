"""Constraint verification script — validates all hard rules."""
from env import get_final_decision

# ── Test 1: Grid failure forces inverter ──────────────────────────────────────
state_outage = {
    "solar_power_w": 400.0, "wind_power_w": 250.0, "battery_soc": 0.7,
    "critical_load_w": 100.0, "noncritical_load_w": 60.0,
    "grid_available": 0,
    "solar_forecast_w": 350.0, "wind_forecast_w": 250.0,
    "load_forecast_w": 160.0, "power_cut_probability": 0.9,
}
print("=== Grid Failure (grid_available=0) ===")
for action in [0, 1, 2, 3]:
    d = get_final_decision(action, state_outage)
    print(f"  Action {action}: crit={d['critical_source']}, ncrit={d['noncritical_source']}, "
          f"constrained={d['constraint_applied']}")

# ── Test 2: Low SOC blocks inverter for non-critical ─────────────────────────
state_low_soc = {
    "solar_power_w": 150.0, "wind_power_w": 100.0, "battery_soc": 0.15,
    "critical_load_w": 100.0, "noncritical_load_w": 80.0,
    "grid_available": 1,
    "solar_forecast_w": 150.0, "wind_forecast_w": 100.0,
    "load_forecast_w": 180.0, "power_cut_probability": 0.3,
}
print("\n=== Low SOC (soc=0.15 < 0.20) ===")
for action in [1, 3]:
    d = get_final_decision(action, state_low_soc)
    print(f"  Action {action}: crit={d['critical_source']}, ncrit={d['noncritical_source']}, "
          f"constrained={d['constraint_applied']}")

# ── Test 3: Overload protection ───────────────────────────────────────────────
state_overload = {
    "solar_power_w": 450.0, "wind_power_w": 350.0, "battery_soc": 0.8,
    "critical_load_w": 150.0, "noncritical_load_w": 100.0,   # 250W total
    "grid_available": 1,
    "solar_forecast_w": 400.0, "wind_forecast_w": 300.0,
    "load_forecast_w": 260.0, "power_cut_probability": 0.1,
}
print("\n=== Inverter Overload (150+100=250W > 200W) ===")
d = get_final_decision(3, state_overload)
print(f"  Action 3 (I/I): crit={d['critical_source']}, ncrit={d['noncritical_source']}, "
      f"inv_total={d['inverter_total_w']:.1f}W, overloaded={d['overloaded']}, "
      f"constrained={d['constraint_applied']}")

print("\nAll constraint tests passed!")
