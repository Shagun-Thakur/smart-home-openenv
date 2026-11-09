import math
import yaml
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.environment import AC_ACTIONS, BATTERY_ACTIONS, encode_action, decode_action
from env.reward import compute_comfort_penalty, is_comfortable

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "realistic.yaml")

class RealisticSmartHomeEnv(gym.Env):
    """
    Realistic Smart Home Environment with continuous diurnal thermal cycle,
    Markovian occupancy transitions, time-of-use tariffs, and first-order thermal inertia.

    Observation Space (Box(7)):
        [0] indoor_temperature (°C)
        [1] outdoor_temperature (°C)
        [2] electricity_price ($/kWh)
        [3] battery_soc (%) [0, 100]
        [4] occupancy (0 = Away, 1 = Home)
        [5] sin_hour (sin(2*pi*hour / 24))
        [6] cos_hour (cos(2*pi*hour / 24))

    Action Space (Discrete(12)):
        Same 12 discrete actions mapping (ac_setpoint, battery_action).
    """
    metadata = {"render_modes": []}

    def __init__(self, config=None, config_path=None):
        super().__init__()

        if config is not None:
            self.cfg = config
        else:
            path = config_path or DEFAULT_CONFIG_PATH
            with open(path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        sim_cfg = self.cfg["simulation"]
        self.step_minutes = sim_cfg["step_minutes"]
        self.steps_per_day = sim_cfg["steps_per_day"]
        self.step_hours = self.step_minutes / 60.0
        self.comfort_min = sim_cfg["comfort_min"]
        self.comfort_max = sim_cfg["comfort_max"]

        self.therm_cfg = self.cfg["thermal"]
        self.outdoor_cfg = self.cfg["outdoor"]
        self.tariff_cfg = self.cfg["tariff"]
        self.occ_cfg = self.cfg["occupancy"]
        self.bat_cfg = self.cfg["battery"]
        self.rew_cfg = self.cfg["reward"]

        # Action space: 12 discrete actions
        self.action_space = spaces.Discrete(12)

        # Observation space: 7 continuous features
        self.observation_space = spaces.Box(
            low=np.array([10.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([45.0, 50.0, 5.0, 100.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.np_random = np.random.default_rng(seed=None)

        # State variables
        self.step_idx = 0
        self.hour = 0.0
        self.indoor_temp = float(self.therm_cfg["initial_indoor_temp"])
        self.outdoor_temp = 30.0
        self.electricity_price = 1.0
        self.battery_soc = float(self.bat_cfg["initial_soc"])
        self.occupancy = int(self.occ_cfg["initial_occupancy"])
        self.prev_ac_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_idx = 0
        self.hour = 0.0
        self.prev_ac_action = None
        self.indoor_temp = float(self.therm_cfg["initial_indoor_temp"])
        self.battery_soc = float(self.bat_cfg["initial_soc"])
        self.occupancy = int(self.occ_cfg["initial_occupancy"])

        self.outdoor_temp = self._compute_outdoor_temp(self.hour)
        self.electricity_price = self._compute_price(self.hour)

        obs = self._get_obs()
        info = self._get_info(energy=0.0, cost=0.0, comfort_penalty=0.0, switching_penalty=0.0)
        return obs, info

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            ac_action, battery_action = decode_action(int(action))
        elif isinstance(action, (tuple, list)):
            ac_action, battery_action = action
        else:
            raise TypeError(f"Invalid action type: {type(action)}")

        current_price = self.electricity_price
        current_occ = self.occupancy

        # ---- 1. Thermal Inertia Dynamics ----
        if ac_action is not None:
            cooling_pull = self.therm_cfg["cooling_rate"] * (ac_action - self.indoor_temp)
            ambient_pull = self.therm_cfg["ambient_heat_gain"] * (self.outdoor_temp - self.indoor_temp)
            self.indoor_temp += cooling_pull + ambient_pull
            ac_power_kw = float(self.therm_cfg["ac_cooling_powers"][ac_action])
        else:
            drift = self.therm_cfg["drift_rate"] * (self.outdoor_temp - self.indoor_temp)
            self.indoor_temp += drift
            ac_power_kw = 0.0

        ac_energy_kwh = ac_power_kw * self.step_hours

        # ---- 2. Battery Flow Dynamics & Consistency ----
        cap_kwh = self.bat_cfg["capacity_kwh"]
        step_flow_kwh = self.bat_cfg["step_flow_kwh"]
        flow_pct = (step_flow_kwh / cap_kwh) * 100.0

        grid_battery_draw_kwh = 0.0
        battery_discharge_relief_kwh = 0.0

        if battery_action == "CHARGE":
            if self.battery_soc < 100.0:
                actual_pct = min(flow_pct, 100.0 - self.battery_soc)
                energy_stored = (actual_pct / 100.0) * cap_kwh
                grid_battery_draw_kwh = energy_stored / self.bat_cfg["charge_efficiency"]
                self.battery_soc += actual_pct
            # If soc == 100, ineffective action (clipped to 0 flow)
        elif battery_action == "DISCHARGE":
            if self.battery_soc > 0.0:
                actual_pct = min(flow_pct, self.battery_soc)
                energy_released = (actual_pct / 100.0) * cap_kwh * self.bat_cfg["discharge_efficiency"]
                self.battery_soc -= actual_pct
                battery_discharge_relief_kwh = energy_released
            # If soc == 0, ineffective action (clipped to 0 flow)

        net_grid_kwh = max(0.0, ac_energy_kwh + grid_battery_draw_kwh - battery_discharge_relief_kwh)
        step_cost = current_price * net_grid_kwh

        # ---- 3. Reward Calculation ----
        comfort_penalty = compute_comfort_penalty(
            self.indoor_temp, self.comfort_min, self.comfort_max, occupancy=current_occ
        )
        switching_penalty = 1.0 if (self.prev_ac_action is not None and self.prev_ac_action != ac_action) else 0.0

        reward = - (
            self.rew_cfg["w_cost"] * step_cost +
            self.rew_cfg["w_comfort"] * comfort_penalty +
            self.rew_cfg["w_switch"] * switching_penalty
        )
        if comfort_penalty == 0.0:
            reward += self.rew_cfg["comfort_bonus"]

        # ---- 4. Step Progression & Exogenous Updates ----
        self.step_idx += 1
        self.hour = (self.step_idx * self.step_hours) % 24.0
        self.prev_ac_action = ac_action

        terminated = False
        truncated = bool(self.step_idx >= self.steps_per_day)

        if not (terminated or truncated):
            self.outdoor_temp = self._compute_outdoor_temp(self.hour)
            self.electricity_price = self._compute_price(self.hour)
            self.occupancy = self._next_occupancy(current_occ)

        obs = self._get_obs()
        info = self._get_info(
            energy=net_grid_kwh,
            cost=step_cost,
            comfort_penalty=comfort_penalty,
            switching_penalty=switching_penalty,
            terminated=terminated,
            truncated=truncated
        )

        return obs, float(reward), terminated, truncated, info

    def _compute_outdoor_temp(self, hour):
        mean = self.outdoor_cfg["mean_temp"]
        amp = self.outdoor_cfg["amplitude"]
        peak_h = self.outdoor_cfg["peak_hour"]
        std = self.outdoor_cfg["noise_std"]

        # Sinusoidal peak at peak_h (3 PM)
        rad = 2.0 * math.pi * (hour - (peak_h - 6.0)) / 24.0
        cycle = mean + amp * math.sin(rad)
        noise = float(self.np_random.normal(0.0, std))
        return float(cycle + noise)

    def _compute_price(self, hour):
        # 3-tier time-of-use
        if 14.0 <= hour < 20.0:
            return float(self.tariff_cfg["peak_price"])      # Peak $2.0
        elif (6.0 <= hour < 14.0) or (20.0 <= hour < 22.0):
            return float(self.tariff_cfg["mid_peak_price"])  # Mid $1.0
        else:
            return float(self.tariff_cfg["off_peak_price"])  # Off-peak $0.5

    def _next_occupancy(self, current_occ):
        # Markov chain transition
        if current_occ == 0:
            # P(Away -> Away) = p_stay_away
            stay = self.np_random.random() < self.occ_cfg["p_stay_away"]
            return 0 if stay else 1
        else:
            # P(Home -> Home) = p_stay_home
            stay = self.np_random.random() < self.occ_cfg["p_stay_home"]
            return 1 if stay else 0

    def _get_obs(self):
        rad = 2.0 * math.pi * self.hour / 24.0
        sin_time = float(math.sin(rad))
        cos_time = float(math.cos(rad))
        return np.array([
            self.indoor_temp,
            self.outdoor_temp,
            self.electricity_price,
            self.battery_soc,
            float(self.occupancy),
            sin_time,
            cos_time
        ], dtype=np.float32)

    def _get_info(self, energy=0.0, cost=0.0, comfort_penalty=0.0, switching_penalty=0.0, terminated=False, truncated=False):
        return {
            "step": self.step_idx,
            "hour": self.hour,
            "indoor_temp": float(self.indoor_temp),
            "outdoor_temp": float(self.outdoor_temp),
            "electricity_price": float(self.electricity_price),
            "battery_soc": float(self.battery_soc),
            "occupancy": int(self.occupancy),
            "energy": float(energy),
            "cost": float(cost),
            "comfort_penalty": float(comfort_penalty),
            "switching_penalty": float(switching_penalty),
            "is_comfortable": is_comfortable(self.indoor_temp, self.comfort_min, self.comfort_max, self.occupancy),
            "terminated": terminated,
            "truncated": truncated,
        }
