import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.environment import SmartHomeEnv

env = SmartHomeEnv()
state, _ = env.reset()

for _ in range(10):
    action = (22, "IDLE")  # test action
    state, reward, terminated, truncated, _ = env.step(action)
    print(state, reward)
