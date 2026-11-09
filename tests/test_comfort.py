import pytest
from env.reward import compute_comfort_penalty, compute_reward, is_comfortable
from env.environment import SmartHomeEnv

def test_comfort_penalty_occupied():
    # Occupied and inside bounds [22, 26] -> penalty 0
    assert compute_comfort_penalty(24.0, 22, 26, occupancy=1) == 0.0
    assert is_comfortable(24.0, 22, 26, occupancy=1) is True
    
    # Occupied and too hot (30°C) -> penalty 4.0
    assert compute_comfort_penalty(30.0, 22, 26, occupancy=1) == 4.0
    assert is_comfortable(30.0, 22, 26, occupancy=1) is False

    # Occupied and too cold (20°C) -> penalty 2.0
    assert compute_comfort_penalty(20.0, 22, 26, occupancy=1) == 2.0
    assert is_comfortable(20.0, 22, 26, occupancy=1) is False

def test_comfort_penalty_vacant_no_discomfort():
    # Vacant (occupancy=0) -> penalty is always 0.0, condition is comfortable
    assert compute_comfort_penalty(35.0, 22, 26, occupancy=0) == 0.0
    assert is_comfortable(35.0, 22, 26, occupancy=0) is True

    assert compute_comfort_penalty(18.0, 22, 26, occupancy=0) == 0.0
    assert is_comfortable(18.0, 22, 26, occupancy=0) is True

def test_reward_bonus_when_comfortable():
    # Penalty 0 gives +1 bonus
    r_comf = compute_reward(cost=0.0, comfort_penalty=0.0, switching_penalty=0.0)
    assert r_comf == 1.0

    r_uncomf = compute_reward(cost=0.0, comfort_penalty=2.0, switching_penalty=0.0)
    assert r_uncomf == -1.0  # - (0.5 * 2.0) = -1.0

def test_env_comfort_consistency_when_vacant():
    env = SmartHomeEnv()
    env.reset()
    env.indoor_temp = 32.0  # Hot
    env.occupancy = 0       # Nobody home
    
    _, reward, terminated, truncated, info = env.step((None, "IDLE"))
    assert info["comfort_penalty"] == 0.0
    assert is_comfortable(env.indoor_temp, env.comfort_min, env.comfort_max, env.occupancy) is True
