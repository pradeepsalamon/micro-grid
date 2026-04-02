"""
env.py — Smart Microgrid Custom Gym Environment
================================================
Implements a Gymnasium-compatible environment for PPO-based microgrid control.

State  : 10 continuous features (solar, wind, SOC, loads, grid, forecasts, etc.)
Actions: 4 discrete choices for (critical_source × noncritical_source)
Reward : Multi-objective shaped signal covering reliability, battery health,
         renewable preference, and future-awareness.
Constraint Layer: Every raw action is passed through `get_final_decision()`
                  before being applied, ensuring physical/operational validity.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ─────────────────────────── constants ────────────────────────────────────────
INVERTER_MAX_W        = 200.0   # hard inverter output cap
SOC_MIN               = 0.20    # battery floor
SOC_MAX               = 1.00
SOC_SAFE_LOW          = 0.30    # "healthy" lower bound for reward bonus
SOC_SAFE_HIGH         = 0.90    # "healthy" upper bound for reward bonus
BATTERY_CAPACITY      = 1.0     # 1 kWh usable battery capacity

# Discharge rate per step at full (200 W) inverter output
# Assumes 1-kWh usable battery; 200 W / 1000 Wh → 0.20 per step (1-hour episode)
DISCHARGE_RATE_PER_W  = 0.20 / 200.0   # SOC lost per Watt per step

# Charge rate from renewables (simplified)
CHARGE_RATE_PER_W     = 0.15 / 200.0   # SOC gained per unit of W RES

# ──────────────────────── action map ──────────────────────────────────────────
ACTION_MAP = {
    0: {"critical_source": "grid",     "noncritical_source": "grid"},
    1: {"critical_source": "grid",     "noncritical_source": "inverter"},
    2: {"critical_source": "inverter", "noncritical_source": "grid"},
    3: {"critical_source": "inverter", "noncritical_source": "inverter"},
}

def normalize_state(s):
    return np.array([
        s["solar_power_w"] / 500,
        s["wind_power_w"] / 500,
        s["battery_soc"],
        s["critical_load_w"] / 300,
        s["noncritical_load_w"] / 300,
        s["grid_available"],
        s["solar_forecast_w"] / 500,
        s["wind_forecast_w"] / 500,
        s["load_forecast_w"] / 300,
        s["power_cut_probability"]
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRAINT LAYER
# ══════════════════════════════════════════════════════════════════════════════

def get_final_decision(
    action_id: int,
    state: dict,
) -> dict:
    """
    Maps a raw policy action to a physically valid final decision.

    Rules enforced (in priority order):
      1. Grid unavailable  → force inverter for critical load
      2. Battery SOC < 20% → block inverter for non-critical load
      3. Inverter overload (total load > 200W) → fallback to grid
      4. If a source contributes 0 W it is re-labelled "grid"

    Returns
    -------
    dict with keys: action_id, critical_source, noncritical_source,
                    critical_load_w, noncritical_load_w,
                    inverter_total_w, overloaded, constraint_applied
    """
    raw = ACTION_MAP[action_id].copy()
    crit_src    = raw["critical_source"]
    noncrit_src = raw["noncritical_source"]

    grid_available    = bool(state["grid_available"])
    battery_soc       = float(state["battery_soc"])
    critical_load_w   = float(state["critical_load_w"])
    noncritical_load_w = float(state["noncritical_load_w"])
    solar_forecast_w  = float(state["solar_forecast_w"])
    wind_forecast_w   = float(state["wind_forecast_w"])
    load_forecast_w   = float(state["load_forecast_w"])
    power_cut_probability = float(state["power_cut_probability"])

    constraint_applied = False
    overloaded         = False

    # ── Energy-aware check ──────────────────────────────────────────────────
    available_energy = battery_soc * BATTERY_CAPACITY
    required_energy = (critical_load_w + noncritical_load_w) / 1000.0

    if available_energy < required_energy:
        if noncrit_src == "inverter":
            noncrit_src = "grid"
            constraint_applied = True

    # ── Hard Battery Safety Rule ─────────────────────────────────────────
    if battery_soc <= 0.25:
        if noncrit_src == "inverter":
            noncrit_src = "grid"
            constraint_applied = True

    # ── Extreme risk & Future-aware overriding ──────────────────────────
    future_generation = solar_forecast_w + wind_forecast_w
    future_load = load_forecast_w

    if (
        battery_soc <= 0.5 and
        power_cut_probability > 0.8 and
        future_generation < future_load
    ):
        if crit_src != "grid" or noncrit_src != "grid":
            crit_src = "grid"
            noncrit_src = "grid"
            constraint_applied = True
    elif (
        battery_soc <= 0.35 and
        power_cut_probability >= 0.7 and
        future_generation < (0.5 * future_load)
    ):
        if crit_src != "grid" or noncrit_src != "grid":
            crit_src = "grid"
            noncrit_src = "grid"
            constraint_applied = True

    # ADD HARD RULE: NO INVERTER FOR NON-CRITICAL IN RISK MODE
    if (
        battery_soc <= 0.35 and
        power_cut_probability >= 0.7
    ):
        if noncrit_src == "inverter":
            noncrit_src = "grid"
            constraint_applied = True

    # ── Safe Override for Critical-only ─────────────────────────────────
    if (
        noncritical_load_w == 0 and
        battery_soc > 0.6 and
        critical_load_w <= INVERTER_MAX_W and
        power_cut_probability < 0.3
    ):
        if crit_src != "inverter":
            crit_src = "inverter"
            constraint_applied = True

    # ── Prioritize Critical Load on Inverter ────────────────────────────
    total_load = critical_load_w + noncritical_load_w
    if total_load > INVERTER_MAX_W:
        if crit_src != "inverter" or noncrit_src != "grid":
            crit_src = "inverter"
            noncrit_src = "grid"
            constraint_applied = True

    # ── Rule 1 : grid unavailable ──────────────────────────────────────────
    if not grid_available:
        if crit_src == "grid":
            crit_src           = "inverter"
            constraint_applied = True
        if noncrit_src == "grid":
            noncrit_src        = "inverter"
            constraint_applied = True

    # ── Rule 2 : low battery → protect non-critical ────────────────────────
    if battery_soc < SOC_MIN and noncrit_src == "inverter":
        noncrit_src        = "grid" if grid_available else "inverter"  # best effort
        constraint_applied = True

    # ── Rule 3 : inverter overload check ──────────────────────────────────
    inv_crit    = critical_load_w   if crit_src    == "inverter" else 0.0
    inv_noncrit = noncritical_load_w if noncrit_src == "inverter" else 0.0
    inverter_total_w = inv_crit + inv_noncrit

    if inverter_total_w > INVERTER_MAX_W:
        overloaded = True
        constraint_applied = True
        # Priority: keep critical on inverter if possible
        if inv_crit <= INVERTER_MAX_W:
            noncrit_src      = "grid" if grid_available else noncrit_src
            inv_noncrit       = 0.0 if noncrit_src == "grid" else inv_noncrit
            inverter_total_w  = inv_crit
        else:
            # Even critical alone exceeds cap → fallback both to grid
            crit_src         = "grid" if grid_available else "inverter"
            noncrit_src      = "grid" if grid_available else "inverter"
            inverter_total_w = 0.0

    # ── Rule 4 : zero-watt inverter re-labelling ───────────────────────────
    if inverter_total_w == 0:
        if crit_src == "inverter":
            crit_src = "grid"
        if noncrit_src == "inverter":
            noncrit_src = "grid"

    # NEVER allow critical failure
    if crit_src not in ["grid", "inverter"]:
        crit_src = "grid"

    # if grid unavailable → must use inverter
    if not grid_available:
        crit_src = "inverter"

    return {
        "action_id":          action_id,
        "critical_source":    crit_src,
        "noncritical_source": noncrit_src,
        "critical_load_w":    critical_load_w,
        "noncritical_load_w": noncritical_load_w,
        "inverter_total_w":   inverter_total_w,
        "overloaded":         overloaded,
        "constraint_applied": constraint_applied,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(decision: dict, state: dict, prev_action: int) -> float:
    """
    Multi-objective shaped reward signal.

    Component breakdown
    -------------------
    +10 / -20  : critical load coverage
    +3  / -3   : inverter vs grid preference
    -10        : deep discharge penalty (SOC < 0.20)
    +2         : healthy SOC band (0.30–0.90)
    +5 / -5    : outage handling
    -2         : action switching penalty
    +3         : decision aligned with forecast
    -5         : inverter overload penalty
    -3         : unrealistic decision (force-action mismatch)
    """
    reward = 0.0

    soc                = float(state["battery_soc"])
    grid_available     = bool(state["grid_available"])
    critical_load_w    = float(state["critical_load_w"])
    solar_forecast_w   = float(state["solar_forecast_w"])
    wind_forecast_w    = float(state["wind_forecast_w"])
    load_forecast_w    = float(state["load_forecast_w"])
    cut_prob           = float(state["power_cut_probability"])
    noncritical_load_w = float(state["noncritical_load_w"])

    crit_src    = decision["critical_source"]
    noncrit_src = decision["noncritical_source"]
    overloaded  = decision["overloaded"]
    inv_total   = decision["inverter_total_w"]
    action_id   = decision["action_id"]

    # 1️⃣ RAW ACTION PENALTY
    raw_inverter_load = 0.0
    if action_id in [2, 3]:
        raw_inverter_load += critical_load_w
    if action_id in [1, 3]:
        raw_inverter_load += noncritical_load_w

    if raw_inverter_load > INVERTER_MAX_W:
        reward -= 8.0

    # ── Critical load satisfaction ─────────────────────────────────────────
    critical_satisfied = (
        (crit_src == "grid"     and grid_available) or
        (crit_src == "inverter" and soc >= SOC_MIN  and inv_total <= INVERTER_MAX_W)
    )
    if not critical_satisfied:
        reward -= 50.0   # very strong penalty
    else:
        reward += 10.0

    # ── Inverter / grid preference ─────────────────────────────────────────
    uses_inverter = (crit_src == "inverter" or noncrit_src == "inverter")
    uses_grid     = (crit_src == "grid"     or noncrit_src == "grid")

    if uses_inverter and not overloaded:
        reward += 3.0
    if uses_grid:
        reward -= 3.0

    # ── Battery health ─────────────────────────────────────────────────────
    if soc < SOC_MIN:
        reward -= 10.0
    elif SOC_SAFE_LOW < soc < SOC_SAFE_HIGH:
        reward += 2.0

    # ── Outage handling ────────────────────────────────────────────────────
    if not grid_available:
        if crit_src == "inverter" and critical_satisfied:
            reward += 5.0
        elif crit_src == "grid":
            reward -= 5.0   # tried to use unavailable grid

    # ── Action repetition penalty (Action balance) ──────────────────────────
    if prev_action is not None and action_id == prev_action:
        reward -= 1.0

    # ── Improve outage reward ──────────────────────────────────────────────
    if cut_prob > 0.7:
        if crit_src == "inverter":
            reward += 8.0
        else:
            reward -= 12.0

    # ── Fix over-conservative battery behavior ─────────────────────────────
    total_load = critical_load_w + noncritical_load_w
    if (
        soc > 0.6 and
        total_load <= INVERTER_MAX_W and
        cut_prob < 0.3
    ):
        if action_id == 3:
            reward += 5.0
        elif action_id == 2:
            reward -= 3.0

    # ── Critical-only Load Logic ───────────────────────────────────────────
    if noncritical_load_w == 0:
        if (
            soc > 0.6 and
            critical_load_w <= INVERTER_MAX_W and
            cut_prob < 0.3
        ):
            if crit_src == "inverter":
                reward += 6.0
            else:
                reward -= 6.0
        elif (
            soc > 0.5 and
            critical_load_w <= INVERTER_MAX_W
        ):
            if crit_src == "inverter":
                reward += 3.0
            else:
                reward -= 3.0

    # ── Penalize unnecessary grid usage ────────────────────────────────────
    if (
        soc > 0.6 and
        total_load <= INVERTER_MAX_W
    ):
        if crit_src == "grid":
            reward -= 4.0
        if noncrit_src == "grid":
            reward -= 3.0

    # ── Encourage action 3 (full inverter) ─────────────────────────────────
    if action_id == 3:
        if total_load <= INVERTER_MAX_W and soc > 0.5:
            reward += 3.0

    # ── Reduce battery hoarding ────────────────────────────────────────────
    if soc > 0.9:
        if action_id in [0, 1]:
            reward -= 3.0

    # ── Encourage balanced usage ───────────────────────────────────────────
    if 0.5 < soc < 0.9:
        if action_id == 3:
            reward += 2.0

    # ── Small penalty for action 2 overuse ─────────────────────────────────
    if action_id == 2:
        reward -= 1.0

    # heavy penalty for unsafe battery usage
    if (
        soc <= 0.35 and
        cut_prob >= 0.7 and
        noncrit_src == "inverter"
    ):
        reward -= 10.0

    # ── Penalty for wrong low-SOC usage ───────────────────────────────────
    if soc <= 0.25 and noncrit_src == "inverter":
        reward -= 8.0

    # ── Fix High-Risk Future Case ──────────────────────────────────────────
    future_generation = solar_forecast_w + wind_forecast_w
    future_load = load_forecast_w

    if (
        soc <= 0.5 and
        cut_prob > 0.7 and
        future_generation < future_load
    ):
        if crit_src == "inverter" or noncrit_src == "inverter":
            reward -= 10.0

    # ── Overload penalty ───────────────────────────────────────────────────
    if overloaded:
        reward -= 5.0

    # ── Constraint mismatch penalty ────────────────────────────────────────
    if decision["constraint_applied"]:
        reward -= 3.0

    return float(reward)


# ══════════════════════════════════════════════════════════════════════════════
#  GYMNASIUM ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class MicrogridEnv(gym.Env):
    """
    Smart Microgrid Discrete Control Environment.

    Observation space : Box(10,) — all features in [0, 1] except load values
    Action space      : Discrete(4)
    Episode length    : 24 steps (hourly dispatch over one day)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 24, seed: int = 42):
        super().__init__()

        self.max_steps  = max_steps
        self._seed      = seed
        self.rng        = np.random.default_rng(seed)

        # ── Spaces ────────────────────────────────────────────────────────
        # [solar, wind, soc, crit_load, noncrit_load, grid_avail, solar_fc, wind_fc, load_fc, cut_prob]
        # Normalized appropriately
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0,    0,    0, 0, 0, 0,    0],    dtype=np.float32),
            high=np.array([4, 4, 1, 10,  10,   1, 4, 4, 10,   1],   dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # ── Episode state ─────────────────────────────────────────────────
        self._state      = None
        self._step_count = 0
        self._prev_action = None
        self.episode_log  = []

    # ── helpers ───────────────────────────────────────────────────────────

    def _sample_state(self) -> dict:
        """Generate a plausible random microgrid state dict."""
        scenario = self.rng.choice([
            "normal", "normal", "high_solar", "no_solar",
            "grid_failure", "high_load", "high_outage", "critical_only"
        ])
        
        solar_w = float(self.rng.uniform(0, 500))
        wind_w  = float(self.rng.uniform(0, 500))
        soc     = float(self.rng.uniform(0.15, 1.0))
        crit_w  = float(self.rng.uniform(50, 300))
        noncrit_w = float(self.rng.uniform(0, 400))
        grid    = int(self.rng.choice([0, 1], p=[0.2, 0.8]))
        cut_p   = float(self.rng.uniform(0, 1))

        if scenario == "high_solar":
            solar_w = float(self.rng.uniform(400, 500))
        elif scenario == "no_solar":
            solar_w = 0.0
        elif scenario == "grid_failure":
            grid = 0
            cut_p = float(self.rng.uniform(0.8, 1.0))
        elif scenario == "high_load":
            crit_w = float(self.rng.uniform(250, 400))
            noncrit_w = float(self.rng.uniform(300, 400))
        elif scenario == "high_outage":
            cut_p = float(self.rng.uniform(0.7, 1.0))
        elif scenario == "critical_only":
            noncrit_w = 0.0

        sol_fc_w = float(np.clip(solar_w + self.rng.normal(0, 50), 0, 500))
        win_fc_w = float(np.clip(wind_w  + self.rng.normal(0, 50), 0, 500))
        load_fc_w = float(np.clip(crit_w + noncrit_w + self.rng.normal(0, 50), 0, 1000))
        return {
            "solar_power_w":          solar_w,
            "wind_power_w":           wind_w,
            "battery_soc":            soc,
            "critical_load_w":        crit_w,
            "noncritical_load_w":     noncrit_w,
            "grid_available":         grid,
            "solar_forecast_w":       sol_fc_w,
            "wind_forecast_w":        win_fc_w,
            "load_forecast_w":        load_fc_w,
            "power_cut_probability":  cut_p,
        }

    def _update_battery_soc(self, decision: dict, state_dict: dict) -> float:
        """Simulate SOC change based on inverter load and renewable generation."""
        soc = state_dict["battery_soc"]

        # Discharge: inverter load draws from battery
        inv_w = decision["inverter_total_w"]
        soc  -= DISCHARGE_RATE_PER_W * inv_w

        # Charge: renewable surplus replenishes battery
        renewable_w = state_dict["solar_power_w"] + state_dict["wind_power_w"]
        excess_w    = max(0.0, renewable_w - inv_w)
        soc         += CHARGE_RATE_PER_W * excess_w

        return float(np.clip(soc, 0.0, SOC_MAX))

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._state       = self._sample_state()
        self._step_count  = 0
        self._prev_action = None
        self.episode_log  = []
        return normalize_state(self._state), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        state_dict = self._state

        # ── Constraint layer ──────────────────────────────────────────────
        decision = get_final_decision(action, state_dict)

        # ── Reward ────────────────────────────────────────────────────────
        reward = compute_reward(decision, state_dict, self._prev_action)

        # ── Update SOC ────────────────────────────────────────────────────
        new_soc = self._update_battery_soc(decision, state_dict)

        # ── Advance state (next hour) ─────────────────────────────────────
        next_state = self._sample_state()
        next_state["battery_soc"] = new_soc    # carry updated SOC
        self._state   = next_state

        self._step_count  += 1
        self._prev_action  = action

        terminated = False
        truncated  = self._step_count >= self.max_steps

        # ── Log ───────────────────────────────────────────────────────────
        self.episode_log.append({
            "step":     self._step_count,
            "state":    state_dict,
            "decision": decision,
            "reward":   reward,
        })

        info = {"decision": decision, "soc_after": new_soc}
        return normalize_state(self._state), reward, terminated, truncated, info

    def render(self, mode="human"):
        if not self.episode_log:
            print("No steps recorded yet.")
            return
        last = self.episode_log[-1]
        d    = last["decision"]
        print(
            f"Step {last['step']:>3} | "
            f"SOC={last['state']['battery_soc']:.2f} | "
            f"Crit→{d['critical_source']:<8} "
            f"NonCrit→{d['noncritical_source']:<8} | "
            f"Inv={d['inverter_total_w']:.1f}W | "
            f"Reward={last['reward']:+.2f}"
        )
