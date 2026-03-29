# Task 2: Balance Comfort and Cost
from env.environment import SmartHomeEnv
class MediumTask:
    def __init__(self):
        self.env = SmartHomeEnv()
        self.max_steps = 50
    def reset(self):
        state = self.env.reset()
        # same environment, but cost now matters
        self.env.outdoor_temp = 35
        self.env.electricity_price = 1.0
        self.env.occupancy = 1
        return state
    def run_episode(self, agent):
        state = self.reset()
        comfort_steps = 0
        total_cost = 0
        total_steps = 0

        for _ in range(self.max_steps):
            action = agent.act(state)
            state, reward, done, _ = self.env.step(action)
            if self.env.comfort_min <= self.env.indoor_temp <= self.env.comfort_max:
                comfort_steps += 1

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
        comfort_ratio = comfort_steps / total_steps
        # Normalize cost
        max_possible_cost = 3 * self.max_steps
        normalized_cost = total_cost / max_possible_cost

        return{"comfort_ratio": comfort_ratio, "normalized_cost": normalized_cost}
    

# class DummyAgent:
#     def act(self, state):
#         return (None, "IDLE")
    
# task = MediumTask()
# result = task.run_episode(DummyAgent())
# print(result)
