# Task 3: Dynamic & Uncertain Environment
from env.environment import SmartHomeEnv

class HardTask:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.env = SmartHomeEnv(mode="hard", max_steps=max_steps)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def run_episode(self, agent, seed=None):
        state, _ = self.env.reset(seed=seed)
        comfort_steps = 0
        total_cost = 0.0
        total_steps = 0
        done = False

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if info["is_comfortable"]:
                comfort_steps += 1
            total_cost += info["cost"]
            total_steps += 1

        comfort_ratio = comfort_steps / total_steps if total_steps > 0 else 1.0
        # Maximum possible cost: AC set to 22 (3 kW) at peak price $1.5/kWh
        max_possible_cost = 3.0 * total_steps * 1.5
        normalized_cost = total_cost / max_possible_cost if max_possible_cost > 0 else 0.0

        return {
            "comfort_ratio": float(comfort_ratio),
            "normalized_cost": float(normalized_cost),
            "total_cost": float(total_cost),
        }
