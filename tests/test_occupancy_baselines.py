"""Base-install safe tests for occupancy-aware baseline agents.

Verifies:
1. Determinism under identical inputs
2. Valid action ranges and action indices in [0, 11]
3. AC turns off (None) during vacant periods when T < T_hot
4. AC sets 24°C on occupied periods
5. AC sets 26°C for OccupancyAware_Precool when vacant and T > T_hot
6. OccupancyAware_TOU never charges the battery during peak pricing
"""

import pytest
import numpy as np
from env.environment import encode_action, decode_action, AC_ACTIONS, BATTERY_ACTIONS
from env.realistic_env import RealisticSmartHomeEnv
from agents.baseline_agent import (
    OccupancyAware_24,
    OccupancyAware_Precool,
    OccupancyAware_TOU,
)


def test_occupancy_baselines_determinism():
    obs = np.array([25.0, 30.0, 1.2, 50.0, 1.0, 0.5, 0.5], dtype=np.float32)

    agents = [
        OccupancyAware_24(),
        OccupancyAware_Precool(t_hot=26.0),
        OccupancyAware_TOU(),
    ]

    for agent in agents:
        act1 = agent.act(obs)
        act2 = agent.act(obs)
        assert act1 == act2, f"{agent.__class__.__name__} is not deterministic"
        idx1 = agent.act_index(obs)
        idx2 = agent.act_index(obs)
        assert idx1 == idx2


def test_occupancy_baselines_valid_action_indices():
    agents = [
        OccupancyAware_24(),
        OccupancyAware_Precool(t_hot=26.0),
        OccupancyAware_TOU(),
    ]

    # Test across a grid of temperatures, prices, and occupancy values
    for temp in [20.0, 25.0, 28.0, 32.0]:
        for price in [0.2, 0.5, 1.0, 1.8, 2.0]:
            for soc in [0.0, 50.0, 100.0]:
                for occ in [0.0, 1.0]:
                    obs = np.array([temp, 30.0, price, soc, occ, 0.0, 1.0], dtype=np.float32)
                    for agent in agents:
                        ac, bat = agent.act(obs)
                        assert ac in AC_ACTIONS, f"Invalid AC action {ac} from {agent}"
                        assert bat in BATTERY_ACTIONS, f"Invalid battery action {bat} from {agent}"

                        idx = agent.act_index(obs)
                        assert 0 <= idx < 12, f"Invalid action index {idx} from {agent}"
                        assert encode_action(ac, bat) == idx

                        dec_ac, dec_bat = decode_action(idx)
                        assert dec_ac == ac
                        assert dec_bat == bat


def test_ac_off_when_vacant():
    agent_24 = OccupancyAware_24()
    agent_precool = OccupancyAware_Precool(t_hot=26.0)
    agent_tou = OccupancyAware_TOU()

    # Vacant and below T_hot (25.0 < 26.0)
    vacant_cool_obs = np.array([25.0, 30.0, 0.8, 40.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for agent in [agent_24, agent_precool, agent_tou]:
        ac, _ = agent.act(vacant_cool_obs)
        assert ac is None, f"{agent.__class__.__name__} failed to turn AC off when vacant and cool"


def test_ac_24_when_occupied():
    agents = [
        OccupancyAware_24(),
        OccupancyAware_Precool(t_hot=26.0),
        OccupancyAware_TOU(),
    ]

    for temp in [21.0, 25.0, 30.0]:
        occupied_obs = np.array([temp, 32.0, 1.0, 60.0, 1.0, 0.0, 1.0], dtype=np.float32)
        for agent in agents:
            ac, _ = agent.act(occupied_obs)
            assert ac == 24, f"{agent.__class__.__name__} failed to set 24°C when occupied"


def test_occupancy_precool_behavior():
    agent = OccupancyAware_Precool(t_hot=26.0)

    # Vacant and hot (27.5 > 26.0) -> should precool at 26°C
    vacant_hot_obs = np.array([27.5, 33.0, 0.8, 50.0, 0.0, 0.0, 1.0], dtype=np.float32)
    ac, bat = agent.act(vacant_hot_obs)
    assert ac == 26
    assert bat == "IDLE"

    # Vacant and cool (24.0 <= 26.0) -> should turn off
    vacant_cool_obs = np.array([24.0, 28.0, 0.8, 50.0, 0.0, 0.0, 1.0], dtype=np.float32)
    ac, bat = agent.act(vacant_cool_obs)
    assert ac is None
    assert bat == "IDLE"


def test_occupancy_tou_never_charges_at_peak():
    agent = OccupancyAware_TOU()

    # Peak tariffs (e.g. 1.5, 2.0 $/kWh) across various SOC levels
    for price in [1.5, 1.8, 2.0, 2.5]:
        for soc in [0.0, 20.0, 50.0, 90.0, 100.0]:
            for occ in [0.0, 1.0]:
                obs = np.array([25.0, 32.0, price, soc, occ, 0.0, 1.0], dtype=np.float32)
                _, bat = agent.act(obs)
                assert bat != "CHARGE", f"OccupancyAware_TOU charged at peak price {price} with SOC {soc}"
                if soc > 10.0:
                    assert bat == "DISCHARGE"
                else:
                    assert bat == "IDLE"


def test_step_execution_in_realistic_env():
    env = RealisticSmartHomeEnv()
    obs, _ = env.reset(seed=123)

    agents = [
        OccupancyAware_24(),
        OccupancyAware_Precool(t_hot=26.0),
        OccupancyAware_TOU(),
    ]

    for agent in agents:
        action_tuple = agent.act(obs)
        next_obs, r, term, trunc, info = env.step(action_tuple)
        assert next_obs.shape == (7,)
        assert not np.isnan(r)

        action_idx = agent.act_index(obs)
        next_obs2, r2, term2, trunc2, info2 = env.step(action_idx)
        assert next_obs2.shape == (7,)
        assert not np.isnan(r2)
