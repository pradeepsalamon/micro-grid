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
from pathlib import Path
from stable_baselines3 import PPO
from pydantic import BaseModel, Field, validator
from env import get_final_decision, ACTION_MAP, normalize_state

MODEL_FILENAME = "microgrid_model.zip"
MODEL_PATH = Path(__file__).resolve().parent.parent / "agent" / MODEL_FILENAME
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME


class PredictionRequest(BaseModel):
    solar_power_w: float = Field(..., ge=0)
    wind_power_w: float = Field(..., ge=0)
    battery_soc: float = Field(..., ge=0.0, le=1.0)
    critical_load_w: float = Field(..., ge=0)
    noncritical_load_w: float = Field(..., ge=0)
    grid_available: int = Field(...)
    solar_forecast_w: float = Field(..., ge=0)
    wind_forecast_w: float = Field(..., ge=0)
    load_forecast_w: float = Field(..., ge=0)
    power_cut_probability: float = Field(..., ge=0.0, le=1.0)

    @validator("grid_available")
    def validate_grid_available(cls, value):
        if value not in (0, 1):
            raise ValueError("grid_available must be 0 or 1")
        return value


class PredictionResponse(BaseModel):
    action_id: int
    critical_source: str
    noncritical_source: str
    inverter_total_w: float
    overloaded: bool
    constraint_applied: bool


# ─── Model helpers ───────────────────────────────────────────────────────────

def load_model(path: Path = MODEL_PATH) -> PPO:
    """Load the trained PPO model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return PPO.load(str(path))


def predict_payload(payload: PredictionRequest, model: PPO) -> PredictionResponse:
    """Predict from a validated request payload and return a response model."""
    state = payload.dict()

    # ── Print input state before decision ────────────────────────────────────
    print("\n" + "─" * 56)
    print("  📥  INCOMING REQUEST STATE (before decision):")
    print(f"    Solar power       : {state['solar_power_w']:.1f} W")
    print(f"    Wind power        : {state['wind_power_w']:.1f} W")
    print(f"    Battery SOC       : {state['battery_soc']:.2f}")
    print(f"    Critical load     : {state['critical_load_w']:.1f} W")
    print(f"    Non-critical load : {state['noncritical_load_w']:.1f} W")
    print(f"    Grid available    : {'Yes' if state['grid_available'] else 'No'}")
    print(f"    Solar forecast    : {state['solar_forecast_w']:.1f} W")
    print(f"    Wind forecast     : {state['wind_forecast_w']:.1f} W")
    print(f"    Load forecast     : {state['load_forecast_w']:.1f} W")
    print(f"    Cut probability   : {state['power_cut_probability']:.2f}")
    print("─" * 56)
    # ─────────────────────────────────────────────────────────────────────────

    obs = normalize_state(state).reshape(1, -1)
    action, _ = model.predict(obs, deterministic=True)
    decision = get_final_decision(int(action.item()), state)
    response = PredictionResponse(
        action_id=int(decision["action_id"]),
        critical_source=str(decision["critical_source"]),
        noncritical_source=str(decision["noncritical_source"]),
        inverter_total_w=float(decision["inverter_total_w"]),
        overloaded=bool(decision["overloaded"]),
        constraint_applied=bool(decision["constraint_applied"]),
    )

    # ── Print response before sending ────────────────────────────────────────
    print("  📤  OUTGOING RESPONSE (before send):")
    print(f"    Action ID         : {response.action_id}")
    print(f"    Critical source   : {response.critical_source}")
    print(f"    Non-critical src  : {response.noncritical_source}")
    print(f"    Inverter total    : {response.inverter_total_w:.1f} W")
    print(f"    Overloaded        : {response.overloaded}")
    print(f"    Constraint applied: {response.constraint_applied}")
    print("─" * 56 + "\n")
    # ─────────────────────────────────────────────────────────────────────────

    return response


def make_decision(state: dict, model: PPO) -> dict:
    """Validate state, run the policy, and return the final constrained decision."""
    errors = validate_input(state)
    if errors:
        raise ValueError(errors)
    return predict(state, model)


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
