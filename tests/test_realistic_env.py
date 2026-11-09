import pytest
import numpy as np
from env.realistic_env import RealisticSmartHomeEnv

def test_realistic_env_specs():
    env = RealisticSmartHomeEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (7,)
    assert env.observation_space.shape == (7,)
    assert env.action_space.n == 12
    assert info["step"] == 0
    assert info["hour"] == 0.0

def test_realistic_env_diurnal_tariff_and_temp():
    env = RealisticSmartHomeEnv()
    obs, _ = env.reset(seed=100)

    # At hour 0 (midnight) -> off-peak price 0.5
    assert obs[2] == 0.5

    # Run until hour 15 (step 30 = 15:00) -> peak price 2.0
    for _ in range(30):
        obs, reward, term, trunc, info = env.step(11) # (None, "IDLE")

    assert info["hour"] == 15.0
    assert obs[2] == 2.0 # Peak price

def test_realistic_env_thermal_inertia():
    env = RealisticSmartHomeEnv()
    env.reset(seed=42)
    env.indoor_temp = 30.0
    env.outdoor_temp = 35.0

    # Step with AC=24
    obs, _, _, _, _ = env.step((24, "IDLE"))
    # In first-order model, temp drops smoothly toward 24, but doesn't jump to 24 in 1 step
    assert 24.0 < obs[0] < 30.0

def test_realistic_env_termination():
    env = RealisticSmartHomeEnv()
    env.reset(seed=42)

    for step in range(47):
        obs, r, term, trunc, info = env.step((24, "IDLE"))
        assert not trunc

    obs, r, term, trunc, info = env.step((24, "IDLE"))
    assert trunc is True
    assert info["step"] == 48

def test_realistic_env_determinism():
    env1 = RealisticSmartHomeEnv()
    env2 = RealisticSmartHomeEnv()

    obs1, _ = env1.reset(seed=999)
    obs2, _ = env2.reset(seed=999)
    np.testing.assert_array_equal(obs1, obs2)

    for _ in range(48):
        obs1, r1, t1, tr1, inf1 = env1.step(5)
        obs2, r2, t2, tr2, inf2 = env2.step(5)
        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2
        assert inf1 == inf2
