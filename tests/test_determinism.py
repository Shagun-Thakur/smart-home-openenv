import numpy as np
from env.environment import SmartHomeEnv
from tasks.task_hard import HardTask
from agents.baseline_agent import BaselineAgent

def test_identical_seed_produces_identical_trajectory():
    env1 = SmartHomeEnv(mode="hard", max_steps=30)
    env2 = SmartHomeEnv(mode="hard", max_steps=30)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_array_equal(obs1, obs2)

    agent = BaselineAgent()
    for _ in range(30):
        action1 = agent.act(obs1)
        action2 = agent.act(obs2)
        assert action1 == action2

        obs1, r1, term1, trunc1, info1 = env1.step(action1)
        obs2, r2, term2, trunc2, info2 = env2.step(action2)

        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        assert info1 == info2

def test_different_seeds_produce_different_trajectories():
    task1 = HardTask(max_steps=50)
    task2 = HardTask(max_steps=50)
    agent = BaselineAgent()

    res1 = task1.run_episode(agent, seed=1)
    res2 = task2.run_episode(agent, seed=2)

    # With different seeds, dynamic pricing/occupancy draws should differ
    assert (res1["comfort_ratio"] != res2["comfort_ratio"]) or (res1["normalized_cost"] != res2["normalized_cost"])
