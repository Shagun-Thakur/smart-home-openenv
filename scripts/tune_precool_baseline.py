"""Tune T_hot for OccupancyAware_Precool using DESIGN SEEDS 2000-2049.

Strict separation:
- Training seeds: 0-9
- Evaluation seeds: 1000-1049
- Design/tuning seeds: 2000-2049

Saves summary to results/baseline_design.csv.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.realistic_env import RealisticSmartHomeEnv
from env.environment import decode_action
from agents.baseline_agent import OccupancyAware_Precool, AlwaysOnAgent

MAX_POSSIBLE_COST = 144.0


def evaluate_agent_on_design_seeds(agent, design_seeds, config_path="configs/realistic.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = RealisticSmartHomeEnv(config=cfg)

    scores = []
    comforts = []
    costs = []

    for seed in design_seeds:
        obs, info = env.reset(seed=seed)
        done = False
        total_cost = 0.0
        occupied_steps = 0
        comfort_occupied_steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, term, trunc, step_info = env.step(action)
            done = term or trunc

            total_cost += step_info["cost"]
            if step_info["occupancy"] == 1:
                occupied_steps += 1
                if step_info["is_comfortable"]:
                    comfort_occupied_steps += 1

        comfort_ratio = (comfort_occupied_steps / occupied_steps) if occupied_steps > 0 else 1.0
        norm_cost = min(1.0, total_cost / MAX_POSSIBLE_COST)
        score = 0.6 * comfort_ratio + 0.4 * (1.0 - norm_cost)

        scores.append(score)
        comforts.append(comfort_ratio)
        costs.append(total_cost)

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_comfort": float(np.mean(comforts)),
        "mean_cost": float(np.mean(costs)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


def main():
    design_seeds = list(range(2000, 2050))
    grid = [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]

    print(f"Running grid search over T_hot in {grid} on {len(design_seeds)} design seeds (2000-2049)...")
    results = []
    for t_hot in grid:
        agent = OccupancyAware_Precool(t_hot=t_hot)
        metrics = evaluate_agent_on_design_seeds(agent, design_seeds)
        row = {"t_hot": str(t_hot), **metrics}
        results.append(row)
        print(f"T_hot={t_hot:.1f}°C -> Score={metrics['mean_score']:.4f} ± {metrics['std_score']:.4f} | Comfort={metrics['mean_comfort']:.4f} | Cost=${metrics['mean_cost']:.2f}")

    # Evaluate AlwaysOn_24 reference on design seeds
    print("\nEvaluating AlwaysOn_24 reference on design seeds (2000-2049)...")
    ao24_agent = AlwaysOnAgent(setpoint=24)
    ao24_metrics = evaluate_agent_on_design_seeds(ao24_agent, design_seeds)
    ao24_row = {"t_hot": "AlwaysOn_24_ref", **ao24_metrics}
    results.append(ao24_row)
    print(f"AlwaysOn_24 (reference) -> Score={ao24_metrics['mean_score']:.4f} ± {ao24_metrics['std_score']:.4f} | Comfort={ao24_metrics['mean_comfort']:.4f} | Cost=${ao24_metrics['mean_cost']:.2f}")

    df = pd.DataFrame(results)

    # Filter to numeric grid rows to pick the best T_hot
    grid_df = df[df["t_hot"] != "AlwaysOn_24_ref"]
    best_row = grid_df.loc[grid_df["mean_score"].idxmax()]
    print("\nBest T_hot parameter based strictly on design seeds:")
    print(best_row)

    out_path = os.path.join("results", "baseline_design.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved design tuning results to {out_path}")


if __name__ == "__main__":
    main()
