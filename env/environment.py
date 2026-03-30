from env.dynamics import update_temperature, update_battery, compute_energy
from env.reward import compute_comfort_penalty, compute_reward
import numpy as np
class SmartHomeEnv:
    def __init__(self):
        #initial state
        self.indoor_temp = 30
        self.outdoor_temp = 35
        self.electricity_price = 1.0
        self.battery_soc = 50
        self.occupancy = 1
        self.comfort_min = 22
        self.comfort_max = 26
        self.prev_ac_action = None

    def reset(self):
        self.indoor_temp = 30
        self.outdoor_temp = 35
        self.electricity_price = 1
        self.battery_soc = 50
        self.occupancy = 1
        return self._get_state()
    
    def step(self, action):
        ac_action, battery_action = action

        # ----dynamics----
        self.indoor_temp = update_temperature(self.indoor_temp, self.outdoor_temp, ac_action)
        self.battery_soc = update_battery(self.battery_soc, battery_action)
        energy = compute_energy(ac_action, battery_action)
        cost = self.electricity_price * energy

        # ----reward----
        comfort_penalty = compute_comfort_penalty(self.indoor_temp, self.comfort_min, self.comfort_max)
        switching_penalty = 0
        if self.prev_ac_action is not None and self.prev_ac_action != ac_action:
            switching_penalty = 1
        reward = compute_reward(cost, comfort_penalty, switching_penalty)
        self.prev_ac_action = ac_action
        done = False  # can add episode length later

        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        return [
            self.indoor_temp,
            self.outdoor_temp,
            self.electricity_price,
            self.battery_soc,
            self.occupancy
        ]
    
# env = SmartHomeEnv()
# state = env.reset()

# for _ in range(10):
#     action = (22, "IDLE")  # test action
#     state, reward, done, _ = env.step(action)
#     print(state, reward)