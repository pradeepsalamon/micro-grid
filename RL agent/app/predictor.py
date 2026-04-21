"""
predictor.py — Manual Input Predictor
=======================================
Allows you to manually enter a microgrid state as a JSON dictionary
and receive the final constrained decision from the trained PPO model.

Usage:
    python predictor.py

Edit the INPUT dict below, or run interactively.
"""

import json
import numpy as np
from stable_baselines3 import PPO
from app.env import get_final_decision, ACTION_MAP, normalize_state

MODEL_PATH = "microgrid_model"

# ─── Edit this block with your desired state ──────────────────────────────────
INPUT: dict = {
    "solar_power_w": 10.0,
    "wind_power_w": 120.0,
    "battery_soc": 0.7,
    "critical_load_w": 100.0,
    "noncritical_load_w": 50.0,
    "grid_available": 1,
    "solar_forecast_w": 20.0,
    "wind_forecast_w": 100.0,
    "load_forecast_w": 160.0,
    "power_cut_probability": 0.2,
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def predict(state: dict, model: PPO) -> dict:
    """Run policy and constraint layer, return final decision."""
    obs    = normalize_state(state).reshape(1, -1)
    action, _ = model.predict(obs, deterministic=True)
    action    = int(action.item())
    decision  = get_final_decision(action, state)
    return decision


def validate_input(state: dict):
    """Basic sanity checks on user-supplied state."""
    errors = []
    for k in ["battery_soc", "power_cut_probability"]:
        v = state.get(k, None)
        if v is None or not (0.0 <= float(v) <= 1.0):
            errors.append(f"  ❌  '{k}' must be in [0, 1]. Got: {v}")
    if int(state.get("grid_available", -1)) not in (0, 1):
        errors.append("  ❌  'grid_available' must be 0 or 1.")
    for k in ["critical_load_w", "noncritical_load_w", "load_forecast_w"]:
        v = state.get(k, None)
        if v is None or float(v) < 0:
            errors.append(f"  ❌  '{k}' must be ≥ 0. Got: {v}")
    return errors


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║        Smart Microgrid RL Predictor  v1.0                ║
╚══════════════════════════════════════════════════════════╝
""")


def interactive_mode(model: PPO):
    """Loop: user edits JSON → get decision."""
    print("\n[Interactive Mode]  Type 'quit' to exit.\n")
    while True:
        raw = input("Paste state JSON (or press Enter to use default): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break

        try:
            state = json.loads(raw) if raw else INPUT
        except json.JSONDecodeError as e:
            print(f"  ❌  Invalid JSON: {e}\n")
            continue

        errors = validate_input(state)
        if errors:
            for e in errors:
                print(e)
            continue

        decision = predict(state, model)
        print_decision(state, decision)


def print_decision(state: dict, decision: dict):
    """Pretty-print the final power allocation decision."""
    print("\n" + "─" * 56)
    print("  INPUT STATE:")
    print(f"    Solar power      : {state['solar_power_w']:.1f} W")
    print(f"    Wind power       : {state['wind_power_w']:.1f} W")
    print(f"    Battery SOC      : {state['battery_soc']:.2f}")
    print(f"    Critical load    : {state['critical_load_w']:.1f} W")
    print(f"    Non-critical load: {state['noncritical_load_w']:.1f} W")
    print(f"    Grid available   : {'Yes' if state['grid_available'] else 'No'}")
    print(f"    Solar forecast   : {state['solar_forecast_w']:.1f} W")
    print(f"    Wind forecast    : {state['wind_forecast_w']:.1f} W")
    print(f"    Load forecast    : {state['load_forecast_w']:.1f} W")
    print(f"    Cut probability  : {state['power_cut_probability']:.2f}")
    print()
    print("  FINAL DECISION:")
    output = {
        "action_id":          decision["action_id"],
        "critical_source":    decision["critical_source"],
        "noncritical_source": decision["noncritical_source"],
    }
    print(json.dumps(output, indent=4))
    print(f"\n    Inverter total   : {decision['inverter_total_w']:.1f} W")
    print(f"    Overloaded       : {decision['overloaded']}")
    print(f"    Constraint fix   : {decision['constraint_applied']}")
    print("─" * 56 + "\n")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print_banner()

    # Load model
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅  Model loaded: '{MODEL_PATH}.zip'\n")
    except FileNotFoundError:
        print(f"❌  '{MODEL_PATH}.zip' not found. Run train.py first.\n")
        return

    # Default single prediction
    print("[Default Prediction]\n")
    errors = validate_input(INPUT)
    if errors:
        for e in errors:
            print(e)
        return

    decision = predict(INPUT, model)
    print_decision(INPUT, decision)

    # Offer interactive mode
    choice = input("Enter interactive mode? [y/N]: ").strip().lower()
    if choice == "y":
        interactive_mode(model)

    print("Goodbye! 👋")


if __name__ == "__main__":
    main()
