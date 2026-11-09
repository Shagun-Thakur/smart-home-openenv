import os
import sys
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask

from graders.grader_easy import EasyGrader
from graders.grader_medium import MediumGrader
from graders.grader_hard import HardGrader

from agents.baseline_agent import BaselineAgent

def evaluate_baseline(num_seeds=10, output_csv="results/tier1_baseline.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    agent = BaselineAgent()

    tasks = [
        ("easy", EasyTask(), EasyGrader()),
        ("medium", MediumTask(), MediumGrader()),
        ("hard", HardTask(), HardGrader()),
    ]

    records = []
    task_stats = {}

    for task_name, task, grader in tasks:
        comfort_list = []
        cost_list = []
        score_list = []

        for seed in range(num_seeds):
            res = task.run_episode(agent, seed=seed)
            score = grader.grade(res)
            comfort = float(res.get("comfort_ratio", 0.0))
            cost = float(res.get("normalized_cost", 0.0))

            comfort_list.append(comfort)
            cost_list.append(cost)
            score_list.append(score)

            records.append({
                "task": task_name,
                "seed": seed,
                "comfort_ratio": comfort,
                "normalized_cost": cost,
                "score": score
            })

        task_stats[task_name] = {
            "comfort_mean": float(np.mean(comfort_list)),
            "comfort_sd": float(np.std(comfort_list, ddof=1)),
            "cost_mean": float(np.mean(cost_list)),
            "cost_sd": float(np.std(cost_list, ddof=1)),
            "score_mean": float(np.mean(score_list)),
            "score_sd": float(np.std(score_list, ddof=1)),
        }

    # Write per-seed CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "seed", "comfort_ratio", "normalized_cost", "score"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Per-seed evaluations saved to {output_csv}\n")

    # Write summary CSV
    summary_file = os.path.splitext(output_csv)[0] + "_summary.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "comfort_mean", "comfort_sd", "cost_mean", "cost_sd", "score_mean", "score_sd"])
        for task_name, s in task_stats.items():
            writer.writerow([
                task_name,
                s["comfort_mean"], s["comfort_sd"],
                s["cost_mean"], s["cost_sd"],
                s["score_mean"], s["score_sd"]
            ])

    print("=== TIER 1 BASELINE AGENT BENCHMARK (10 SEEDS: 0-9) ===")
    for task_name, s in task_stats.items():
        print(f"Task: {task_name.upper()}")
        print(f"  Comfort Ratio:   {s['comfort_mean']:.4f} ± {s['comfort_sd']:.4f}")
        print(f"  Normalized Cost: {s['cost_mean']:.4f} ± {s['cost_sd']:.4f}")
        print(f"  Score:           {s['score_mean']:.4f} ± {s['score_sd']:.4f}\n")

    return records, task_stats

if __name__ == "__main__":
    evaluate_baseline()
