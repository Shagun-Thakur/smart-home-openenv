# Task 1: maintain Comfort
from env.environment import SmartHomeEnv
class EasyTask:
    def __init__(self):
        self.env = SmartHomeEnv()
        self.max_steps = 50

    def reset(self):
        state = self.env.reset()
        #print("After reset: ", state)
        self.env.outdoor_temp = 35
        self.env.electricity_price = 1.0
        self.env.occupancy = 1

    def run_episode(self, agent):
        state = self.env.reset()
        done = False
        comfort_steps = 0
        total_steps = 0
        for _ in range(self.max_steps):
            #print("Before act: ", state)
            action = agent.act(state)
            state, reward, done, _ = self.env.step(action)
            # check comfort
            if self.env.comfort_min <= self.env.indoor_temp <= self.env.comfort_max:
                comfort_steps += 1
            total_steps += 1

        return {"comfort_ratio": comfort_steps/ total_steps}
    

# class DummyAgent:
#     def act(self, state):
#         return (26, "IDLE")
    
# task = EasyTask()
# result = task.run_episode(DummyAgent())
# print(result)
