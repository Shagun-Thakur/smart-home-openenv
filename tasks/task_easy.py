# Task 1: Maintain Comfort
from env.environment import SmartHomeEnv

class EasyTask:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.env = SmartHomeEnv(mode="easy", max_steps=max_steps)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def run_episode(self, agent, seed=None):
        state, _ = self.env.reset(seed=seed)
        comfort_steps = 0
        total_steps = 0
        done = False

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if info["is_comfortable"]:
                comfort_steps += 1
            total_steps += 1

        comfort_ratio = comfort_steps / total_steps if total_steps > 0 else 1.0
        return {"comfort_ratio": float(comfort_ratio)}
