import pytest
from env.environment import SmartHomeEnv
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask
from agents.baseline_agent import BaselineAgent

def test_cost_accounting_single_source_of_truth():
    env = SmartHomeEnv(mode="medium", max_steps=10)
    obs, _ = env.reset(seed=10)

    # Step with AC=22 (base 3.0), IDLE, price 1.0 -> cost 3.0
    _, reward, _, _, info = env.step((22, "IDLE"))
    assert info["energy"] == 3.0
    assert info["cost"] == 3.0

    # Step with AC=24 (base 2.0), DISCHARGE (halved to 1.0) -> cost 1.0
    _, reward, _, _, info = env.step((24, "DISCHARGE"))
    assert info["energy"] == 1.0
    assert info["cost"] == 1.0

def test_tasks_use_env_cost_accounting():
    task = MediumTask(max_steps=20)
    agent = BaselineAgent()
    res = task.run_episode(agent, seed=123)

    assert "total_cost" in res
    assert "normalized_cost" in res
    assert 0.0 <= res["normalized_cost"] <= 1.0
