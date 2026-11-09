import pytest
import numpy as np
from env.environment import SmartHomeEnv

def test_observation_contains_current_step_variables():
    env = SmartHomeEnv(mode="hard", max_steps=10)
    obs, info = env.reset(seed=999)

    # obs: [indoor_temp, outdoor_temp, electricity_price, battery_soc, occupancy]
    assert obs[2] == info["electricity_price"]
    assert obs[4] == info["occupancy"]

    for _ in range(5):
        action = (24, "IDLE")
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # The cost of the step must have been evaluated against the price that was in effect
        assert step_info["cost"] == pytest.approx(step_info["electricity_price"] * step_info["energy"])
        
        # If not done, next_obs holds the active price & occupancy for the upcoming decision
        if not (terminated or truncated):
            assert next_obs[2] == env.electricity_price
            assert next_obs[4] == env.occupancy
