import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.dynamics import update_temperature, update_battery, compute_energy
from env.reward import compute_comfort_penalty, compute_reward, is_comfortable

AC_ACTIONS = [22, 24, 26, None]
BATTERY_ACTIONS = ["CHARGE", "DISCHARGE", "IDLE"]

def encode_action(ac_action, battery_action):
    """
    Encode an (ac_action, battery_action) tuple into an integer in [0, 11].
    AC setpoints: [22, 24, 26, None] (indices 0..3)
    Battery actions: ["CHARGE", "DISCHARGE", "IDLE"] (indices 0..2)
    """
    if ac_action not in AC_ACTIONS:
        raise ValueError(f"Invalid ac_action {ac_action}, must be one of {AC_ACTIONS}")
    if battery_action not in BATTERY_ACTIONS:
        raise ValueError(f"Invalid battery_action {battery_action}, must be one of {BATTERY_ACTIONS}")
    ac_idx = AC_ACTIONS.index(ac_action)
    bat_idx = BATTERY_ACTIONS.index(battery_action)
    return ac_idx * len(BATTERY_ACTIONS) + bat_idx

def decode_action(action_idx):
    """
    Decode an integer in [0, 11] into an (ac_action, battery_action) tuple.
    """
    if not (0 <= action_idx < 12):
        raise ValueError(f"action_idx {action_idx} must be in [0, 11]")
    ac_idx = int(action_idx) // len(BATTERY_ACTIONS)
    bat_idx = int(action_idx) % len(BATTERY_ACTIONS)
    return AC_ACTIONS[ac_idx], BATTERY_ACTIONS[bat_idx]


class SmartHomeEnv(gym.Env):
    """
    Gymnasium-compliant Smart Home Energy Optimization Environment.

    State:
        [indoor_temp, outdoor_temp, electricity_price, battery_soc, occupancy]
        - indoor_temp: Indoor temperature in °C
        - outdoor_temp: Outdoor temperature in °C
        - electricity_price: Electricity price in $/kWh
        - battery_soc: Battery state of charge in % [0, 100]
        - occupancy: Binary indicator (1 if occupied, 0 if vacant)

    Action:
        Discrete(12) representing the Cartesian product of:
        - ac_action: [22, 24, 26, None]
        - battery_action: ["CHARGE", "DISCHARGE", "IDLE"]
        Also polymorphically accepts raw tuple actions (ac_action, battery_action).
    """
    metadata = {"render_modes": []}

    def __init__(self, mode="easy", max_steps=50):
        super().__init__()
        self.mode = mode
        self.max_steps = max_steps
        self.comfort_min = 22
        self.comfort_max = 26

        # Gymnasium action and observation spaces
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([45.0, 50.0, 5.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Environment-owned seeded random generator
        self.np_random = np.random.default_rng(seed=None)

        # Internal state variables
        self.indoor_temp = 30.0
        self.outdoor_temp = 35.0
        self.electricity_price = 1.0
        self.battery_soc = 50.0
        self.occupancy = 1
        self.prev_ac_action = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.prev_ac_action = None
        self.indoor_temp = 30.0
        self.outdoor_temp = 35.0
        self.battery_soc = 50.0

        if self.mode == "hard":
            self.electricity_price = float(self.np_random.choice([0.5, 1.0, 1.5]))
            self.occupancy = int(self.np_random.choice([0, 1], p=[0.3, 0.7]))
        else:
            self.electricity_price = 1.0
            self.occupancy = 1

        obs = self._get_state()
        info = {
            "current_step": self.current_step,
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": self.outdoor_temp,
            "electricity_price": self.electricity_price,
            "battery_soc": self.battery_soc,
            "occupancy": self.occupancy,
        }
        return obs, info

    def step(self, action):
        # Allow polymorphic action input: integer index or (ac, battery) tuple
        if isinstance(action, (int, np.integer)):
            ac_action, battery_action = decode_action(int(action))
        elif isinstance(action, (tuple, list)):
            ac_action, battery_action = action
        else:
            raise TypeError(f"Invalid action type: {type(action)}. Must be int or tuple.")

        # Capture step starting conditions
        step_price = self.electricity_price
        step_occupancy = self.occupancy

        # ---- Dynamics ----
        self.indoor_temp = update_temperature(self.indoor_temp, self.outdoor_temp, ac_action)
        energy = compute_energy(ac_action, battery_action, self.battery_soc)
        self.battery_soc = update_battery(self.battery_soc, battery_action)
        cost = step_price * energy

        # ---- Reward ----
        comfort_penalty = compute_comfort_penalty(
            self.indoor_temp, self.comfort_min, self.comfort_max, occupancy=step_occupancy
        )
        switching_penalty = 1.0 if (self.prev_ac_action is not None and self.prev_ac_action != ac_action) else 0.0
        reward = compute_reward(cost, comfort_penalty, switching_penalty)

        self.prev_ac_action = ac_action
        self.current_step += 1

        # Episode termination / truncation
        terminated = False
        truncated = bool(self.current_step >= self.max_steps)

        # Advance exogenous state variables for the next step so next observation is current
        if not (terminated or truncated) and self.mode == "hard":
            self.electricity_price = float(self.np_random.choice([0.5, 1.0, 1.5]))
            self.occupancy = int(self.np_random.choice([0, 1], p=[0.3, 0.7]))

        obs = self._get_state()
        info = {
            "energy": float(energy),
            "cost": float(cost),
            "comfort_penalty": float(comfort_penalty),
            "switching_penalty": float(switching_penalty),
            "indoor_temp": float(self.indoor_temp),
            "outdoor_temp": float(self.outdoor_temp),
            "electricity_price": float(step_price),
            "battery_soc": float(self.battery_soc),
            "occupancy": int(step_occupancy),
            "is_comfortable": is_comfortable(self.indoor_temp, self.comfort_min, self.comfort_max, step_occupancy),
            "terminated": terminated,
            "truncated": truncated,
            "current_step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def _get_state(self):
        return np.array([
            self.indoor_temp,
            self.outdoor_temp,
            self.electricity_price,
            self.battery_soc,
            self.occupancy
        ], dtype=np.float32)


class LegacySmartHomeEnv(gym.Wrapper):
    """
    Backward-compatible wrapper returning standard 4-tuple (state, reward, done, info)
    and list states for legacy scripts.
    """
    def __init__(self, env=None):
        if env is None:
            env = SmartHomeEnv()
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return list(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return list(obs), reward, done, info
