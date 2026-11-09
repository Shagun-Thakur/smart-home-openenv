import pytest
from env.environment import SmartHomeEnv, LegacySmartHomeEnv

def test_episode_truncation_at_horizon():
    max_steps = 20
    env = SmartHomeEnv(max_steps=max_steps)
    obs, info = env.reset(seed=123)
    assert info["current_step"] == 0

    for step in range(1, max_steps):
        obs, reward, terminated, truncated, info = env.step((24, "IDLE"))
        assert not terminated
        assert not truncated
        assert info["current_step"] == step

    # Final step reaching max_steps
    obs, reward, terminated, truncated, info = env.step((24, "IDLE"))
    assert not terminated
    assert truncated is True
    assert info["current_step"] == max_steps

def test_legacy_wrapper_done_flag():
    max_steps = 10
    raw_env = SmartHomeEnv(max_steps=max_steps)
    env = LegacySmartHomeEnv(raw_env)
    state = env.reset(seed=123)
    assert isinstance(state, list)

    for _ in range(max_steps - 1):
        state, reward, done, info = env.step((24, "IDLE"))
        assert done is False

    state, reward, done, info = env.step((24, "IDLE"))
    assert done is True
