import pytest
import numpy as np
from env.environment import SmartHomeEnv, encode_action, decode_action, AC_ACTIONS, BATTERY_ACTIONS

def test_encode_decode_round_trip():
    # Verify that all 12 combinations round-trip perfectly
    for ac in AC_ACTIONS:
        for bat in BATTERY_ACTIONS:
            idx = encode_action(ac, bat)
            assert 0 <= idx < 12
            decoded_ac, decoded_bat = decode_action(idx)
            assert decoded_ac == ac
            assert decoded_bat == bat

def test_action_space_properties():
    env = SmartHomeEnv()
    assert env.action_space.n == 12
    assert env.observation_space.shape == (5,)
    assert env.observation_space.dtype == np.float32

def test_invalid_action_exceptions():
    with pytest.raises(ValueError):
        encode_action(30, "IDLE")
    with pytest.raises(ValueError):
        encode_action(24, "FLY")
    with pytest.raises(ValueError):
        decode_action(-1)
    with pytest.raises(ValueError):
        decode_action(12)

def test_polymorphic_action_acceptance():
    env = SmartHomeEnv()
    env.reset(seed=42)

    # Integer action
    obs1, r1, term1, trunc1, info1 = env.step(0)  # (22, "CHARGE")
    assert obs1.shape == (5,)

    # Tuple action
    obs2, r2, term2, trunc2, info2 = env.step((24, "IDLE"))
    assert obs2.shape == (5,)

    # Invalid action type
    with pytest.raises(TypeError):
        env.step("invalid_action")
