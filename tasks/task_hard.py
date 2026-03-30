# Task 3: Dynamic & Uncertain Environment
from env.environment import SmartHomeEnv
import numpy as np

class HardTask:
    def __init__(self):
        self.env = SmartHomeEnv()
        self.max_steps = 50

    def reset(self):
        state = self.env.reset()
        self.env.outdoor_temp = 35
        self.env.occupancy = 1
        return state
    def run_episode(self, agent):
        state = self.env.reset()
        comfort_steps = 0
        total_cost = 0
        total_steps = 0

        for step in range(self.max_steps):
            # ----Dynamic Changes----

            # Electricity price varies over time
            self.env.electricity_price = np.random.choice([0.5, 1.0, 1.5])
            # Occupancy changes (simulate real life)
            self.env.occupancy = np.random.choice([0, 1], p = [0.3, 0.7])
            
            # ----Agent action----
            action = agent.act(state)
            state, reward, done, _ = self.env.step(action)
            # ----Comfort Tracking----
            if self.env.occupancy == 1:
                if self.env.comfort_min <= self.env.indoor_temp <= self.env.comfort_max:
                    comfort_steps += 1
            else:
                # relaxed when empty
                comfort_steps += 1  
           # print(self.env.indoor_temp, self.env.occupancy)
            # ----Cost Tracking----
            ac_action, battery_action = action
            if ac_action == 22:
                energy = 3
            elif ac_action == 24:
                energy = 2
            elif ac_action == 26:
                energy = 1
            else:
                energy = 0

            if battery_action == "DISCHARGE":
                energy *= 0.5
            elif battery_action == "CHARGE":
                energy *= 1.2
            
            step_cost = self.env.electricity_price * energy
            total_cost += step_cost
            total_steps += 1
        comfort_ratio = comfort_steps/ total_steps
        max_possible_cost = 3 * self.max_steps * 1.5
        normalized_cost = float(total_cost / max_possible_cost)

        return{"comfort_ratio": comfort_ratio, "normalized_cost": normalized_cost}
        

# class DummyAgent:
#     def act(self, state):
#         return (26, "IDLE")
    
# task = HardTask()
# result = task.run_episode(DummyAgent())
# print(result)

            