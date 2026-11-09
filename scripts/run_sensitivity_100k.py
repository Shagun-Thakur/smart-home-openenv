"""Extended 100,000-Step Sensitivity Analysis for PPO and DQN across Seeds 0-9.

Evaluates sensitivity to training budget by comparing:
- 100k-step trained PPO (seeds 0-9)
- 100k-step trained DQN (seeds 0-9)
against reference baselines (AlwaysOn_24, OccupancyAware_Precool, AdaptedRuleAgent)
and against the 20k-step models.

Reports statistics at BOTH levels:
1. Episode-level: paired Wilcoxon signed-rank test across the 50 evaluation episodes.
2. Training-seed-level: per-seed mean scores compared against reference policies
   (seed wins out of 10, bootstrap 95% CI, one-sample Wilcoxon signed-rank test).
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO, DQN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.realistic_env import RealisticSmartHomeEnv
from env.environment import decode_action
from scripts.evaluate_all import run_eval_episode
from scripts.stats_utils import (
    holm_bonferroni,
    bootstrap_mean_ci,
    load_checkpoint_manifest,
    verify_checkpoint,
)


def train_and_eval_sensitivity(
    config_path="configs/realistic.yaml",
    save_dir="results/models/sensitivity",
    output_dir="results",
    train_seeds=tuple(range(10)),
    eval_seeds_count=50,
    start_eval_seed=1000,
    total_timesteps=100000,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = RealisticSmartHomeEnv(config=cfg)
    eval_seeds = list(range(start_eval_seed, start_eval_seed + eval_seeds_count))

    # 1. Train PPO and DQN models for 100k steps (seeds 0-9)
    algos = ["ppo", "dqn"]
    for algo in algos:
        for s in train_seeds:
            model_path = os.path.join(save_dir, f"{algo}_100k_seed{s}.zip")
            if os.path.exists(model_path):
                print(f"Found existing model: {model_path}, skipping training.")
                continue

            print(f"Training {algo.upper()} seed {s} for {total_timesteps} steps...")
            train_env = RealisticSmartHomeEnv(config=cfg)
            train_env.reset(seed=s)

            if algo == "ppo":
                model = PPO("MlpPolicy", train_env, verbose=0, seed=s)
            else:
                model = DQN("MlpPolicy", train_env, verbose=0, seed=s)

            model.learn(total_timesteps=total_timesteps)
            model.save(model_path)
            print(f"Saved: {model_path}")

    # 2. Evaluate all 100k models on the 50 shared evaluation episodes
    base_models_dir = os.path.dirname(os.path.abspath(save_dir))
    manifest_path = os.path.join(base_models_dir, "MANIFEST.sha256")
    manifest = load_checkpoint_manifest(manifest_path)

    eval_records = []
    for algo in algos:
        for s in train_seeds:
            model_path = os.path.join(save_dir, f"{algo}_100k_seed{s}.zip")
            verify_checkpoint(model_path, models_dir=base_models_dir, manifest=manifest)
            if algo == "ppo":
                model = PPO.load(model_path)
            else:
                model = DQN.load(model_path)

            agent_fn = lambda obs, m=model: int(m.predict(obs, deterministic=True)[0])

            print(f"Evaluating {algo.upper()} 100k (train seed {s}) on {len(eval_seeds)} episodes...")
            for es in eval_seeds:
                rec = run_eval_episode(env, agent_fn, es)
                rec["agent"] = f"{algo.upper()}_100k"
                rec["algo"] = algo.upper()
                rec["train_seed"] = s
                eval_records.append(rec)

    df_100k = pd.DataFrame(eval_records)

    # 3. Load baseline evaluation records from results/evaluation_metrics.csv
    base_metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
    if os.path.exists(base_metrics_path):
        base_df = pd.read_csv(base_metrics_path)
    else:
        raise FileNotFoundError(f"Missing {base_metrics_path}")

    ao24_mean = float(base_df[base_df["agent"] == "AlwaysOn_24"]["score"].mean())
    rule_mean = float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["score"].mean())
    precool_mean = float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["score"].mean())

    # 4. Generate per-training-seed breakdown (results/sensitivity_100k_per_seed.csv)
    per_seed_rows = []
    seed_scores_dict = {"PPO": {}, "DQN": {}}

    for algo in ["PPO", "DQN"]:
        sub_df = df_100k[df_100k["algo"] == algo]
        for s in train_seeds:
            s_df = sub_df[sub_df["train_seed"] == s]
            m_score = float(s_df["score"].mean())
            m_comfort = float(s_df["comfort_ratio"].mean())
            m_cost = float(s_df["total_cost"].mean())
            seed_scores_dict[algo][s] = m_score

            per_seed_rows.append({
                "Algorithm": algo,
                "Train_Seed": s,
                "Mean_Score": m_score,
                "Mean_Comfort": m_comfort,
                "Mean_Cost": m_cost,
                "Beats_AlwaysOn_24": bool(m_score > ao24_mean),
                "Beats_AdaptedRuleAgent": bool(m_score > rule_mean),
                "Beats_OccupancyAware_Precool": bool(m_score > precool_mean),
            })

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_csv = os.path.join(output_dir, "sensitivity_100k_per_seed.csv")
    per_seed_df.to_csv(per_seed_csv, index=False)
    print(f"\nPer-training-seed breakdown saved to {per_seed_csv}")

    # 5. Episode-Level Statistical Testing (Wilcoxon Signed-Rank with Holm Correction)
    per_ep_dict = {
        "PPO_100k": df_100k[df_100k["agent"] == "PPO_100k"].groupby("seed")["score"].mean().sort_index(),
        "DQN_100k": df_100k[df_100k["agent"] == "DQN_100k"].groupby("seed")["score"].mean().sort_index(),
        "AlwaysOn_24": base_df[base_df["agent"] == "AlwaysOn_24"].set_index("seed")["score"].sort_index(),
        "OccupancyAware_Precool": base_df[base_df["agent"] == "OccupancyAware_Precool"].set_index("seed")["score"].sort_index(),
        "AdaptedRuleAgent": base_df[base_df["agent"] == "AdaptedRuleAgent"].set_index("seed")["score"].sort_index(),
        "PPO_20k": base_df[base_df["agent"] == "PPO"].groupby("seed")["score"].mean().sort_index(),
        "DQN_20k": base_df[base_df["agent"] == "DQN"].groupby("seed")["score"].mean().sort_index(),
    }
    ep_df = pd.DataFrame(per_ep_dict)
    assert not ep_df.isna().any().any(), "Evaluation table has missing seeds!"

    ep_comparisons = [
        ("PPO_100k", "AlwaysOn_24"),
        ("PPO_100k", "OccupancyAware_Precool"),
        ("PPO_100k", "AdaptedRuleAgent"),
        ("PPO_100k", "PPO_20k"),
        ("DQN_100k", "AlwaysOn_24"),
        ("DQN_100k", "OccupancyAware_Precool"),
        ("DQN_100k", "AdaptedRuleAgent"),
        ("DQN_100k", "DQN_20k"),
        ("DQN_100k", "PPO_100k"),
    ]

    ep_stats_rows = []
    p_vals = []
    for a, b in ep_comparisons:
        sa = ep_df[a]
        sb = ep_df[b]
        diff = sa - sb
        w_stat, p_val = stats.wilcoxon(sa, sb)
        p_vals.append(p_val)
        ep_stats_rows.append({
            "Level": "Episode",
            "Comparison": f"{a} vs {b}",
            "Median_Paired_Diff": float(np.median(diff)),
            "Mean_Paired_Diff": float(np.mean(diff)),
            "Wins": int((diff > 0).sum()),
            "Total_Units": len(diff),
            "Wilcoxon_W": float(w_stat),
            "P_Value_Raw": float(p_val),
            "Note": "Averages over training seeds; ignores seed variance",
        })

    adj_p = holm_bonferroni(p_vals)
    for r, p_adj in zip(ep_stats_rows, adj_p):
        r["P_Value_Holm"] = float(p_adj)
        r["Significant_p05"] = bool(p_adj < 0.05)

    # 6. Training-Seed-Level Statistical Testing (Per-seed mean scores vs reference policies)
    seed_comparisons = [
        ("PPO_100k", "AlwaysOn_24", ao24_mean),
        ("PPO_100k", "OccupancyAware_Precool", precool_mean),
        ("PPO_100k", "AdaptedRuleAgent", rule_mean),
        ("DQN_100k", "AlwaysOn_24", ao24_mean),
        ("DQN_100k", "OccupancyAware_Precool", precool_mean),
        ("DQN_100k", "AdaptedRuleAgent", rule_mean),
    ]

    seed_stats_rows = []
    seed_p_vals = []
    for algo_label, ref_label, ref_val in seed_comparisons:
        algo_name = "PPO" if "PPO" in algo_label else "DQN"
        scores = np.array([seed_scores_dict[algo_name][s] for s in train_seeds])
        diffs = scores - ref_val
        w_stat, p_val = stats.wilcoxon(diffs)
        seed_p_vals.append(p_val)
        ci_low, ci_high = bootstrap_mean_ci(scores, num_bootstraps=10000, ci=0.95)

        seed_stats_rows.append({
            "Level": "Training_Seed",
            "Comparison": f"{algo_label} vs {ref_label}",
            "Median_Paired_Diff": float(np.median(diffs)),
            "Mean_Paired_Diff": float(np.mean(diffs)),
            "Wins": int((diffs > 0).sum()),
            "Total_Units": len(diffs),
            "Wilcoxon_W": float(w_stat),
            "P_Value_Raw": float(p_val),
            "Note": f"Seed mean score bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]",
        })

    adj_seed_p = holm_bonferroni(seed_p_vals)
    for r, p_adj in zip(seed_stats_rows, adj_seed_p):
        r["P_Value_Holm"] = float(p_adj)
        r["Significant_p05"] = bool(p_adj < 0.05)

    # Combined stats dataframe
    stats_df = pd.DataFrame(ep_stats_rows + seed_stats_rows)
    sens_csv = os.path.join(output_dir, "sensitivity_100k.csv")
    stats_df.to_csv(sens_csv, index=False)
    print(f"\nSensitivity statistical test results saved to {sens_csv}")

    # 7. Aggregate Summary Table
    summary_rows = [
        {
            "Agent": "PPO_100k (all 10 seeds)",
            "Score_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["score"].mean()),
            "Score_SD": float(df_100k[df_100k["agent"] == "PPO_100k"]["score"].std()),
            "Comfort_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["comfort_ratio"].mean()),
            "Comfort_SD": float(df_100k[df_100k["agent"] == "PPO_100k"]["comfort_ratio"].std()),
            "Cost_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["total_cost"].mean()),
            "Cost_SD": float(df_100k[df_100k["agent"] == "PPO_100k"]["total_cost"].std()),
            "Switches_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "DQN_100k (all 10 seeds)",
            "Score_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["score"].mean()),
            "Score_SD": float(df_100k[df_100k["agent"] == "DQN_100k"]["score"].std()),
            "Comfort_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["comfort_ratio"].mean()),
            "Comfort_SD": float(df_100k[df_100k["agent"] == "DQN_100k"]["comfort_ratio"].std()),
            "Cost_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["total_cost"].mean()),
            "Cost_SD": float(df_100k[df_100k["agent"] == "DQN_100k"]["total_cost"].std()),
            "Switches_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "AlwaysOn_24",
            "Score_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "AlwaysOn_24"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["comfort_ratio"].mean()),
            "Comfort_SD": float(base_df[base_df["agent"] == "AlwaysOn_24"]["comfort_ratio"].std()),
            "Cost_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["total_cost"].mean()),
            "Cost_SD": float(base_df[base_df["agent"] == "AlwaysOn_24"]["total_cost"].std()),
            "Switches_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "PPO_20k (all 10 seeds)",
            "Score_Mean": float(base_df[base_df["agent"] == "PPO"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "PPO"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "PPO"]["comfort_ratio"].mean()),
            "Comfort_SD": float(base_df[base_df["agent"] == "PPO"]["comfort_ratio"].std()),
            "Cost_Mean": float(base_df[base_df["agent"] == "PPO"]["total_cost"].mean()),
            "Cost_SD": float(base_df[base_df["agent"] == "PPO"]["total_cost"].std()),
            "Switches_Mean": float(base_df[base_df["agent"] == "PPO"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "OccupancyAware_Precool",
            "Score_Mean": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["comfort_ratio"].mean()),
            "Comfort_SD": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["comfort_ratio"].std()),
            "Cost_Mean": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["total_cost"].mean()),
            "Cost_SD": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["total_cost"].std()),
            "Switches_Mean": float(base_df[base_df["agent"] == "OccupancyAware_Precool"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "OccupancyAware_TOU",
            "Score_Mean": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["comfort_ratio"].mean()),
            "Comfort_SD": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["comfort_ratio"].std()),
            "Cost_Mean": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["total_cost"].mean()),
            "Cost_SD": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["total_cost"].std()),
            "Switches_Mean": float(base_df[base_df["agent"] == "OccupancyAware_TOU"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "AdaptedRuleAgent",
            "Score_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["comfort_ratio"].mean()),
            "Comfort_SD": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["comfort_ratio"].std()),
            "Cost_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["total_cost"].mean()),
            "Cost_SD": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["total_cost"].std()),
            "Switches_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "DQN_20k (all 10 seeds)",
            "Score_Mean": float(base_df[base_df["agent"] == "DQN"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "DQN"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "DQN"]["comfort_ratio"].mean()),
            "Comfort_SD": float(base_df[base_df["agent"] == "DQN"]["comfort_ratio"].std()),
            "Cost_Mean": float(base_df[base_df["agent"] == "DQN"]["total_cost"].mean()),
            "Cost_SD": float(base_df[base_df["agent"] == "DQN"]["total_cost"].std()),
            "Switches_Mean": float(base_df[base_df["agent"] == "DQN"]["ac_switch_count"].mean()),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    sens_summary_csv = os.path.join(output_dir, "sensitivity_100k_summary.csv")
    summary_df.to_csv(sens_summary_csv, index=False)
    print(f"Sensitivity summary saved to {sens_summary_csv}")

    print("\n=== 100K SENSITIVITY BENCHMARK SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== STATISTICAL TESTS (100K vs Reference Policies) ===")
    print(stats_df.to_string(index=False))

    print("\n=== PER-TRAINING-SEED BREAKDOWN (100k) ===")
    print(per_seed_df.to_string(index=False))

    return summary_df, stats_df, per_seed_df


if __name__ == "__main__":
    train_and_eval_sensitivity()
