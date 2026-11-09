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
from scripts.evaluate_all import run_eval_episode, holm_bonferroni


def train_and_eval_sensitivity(
    config_path="configs/realistic.yaml",
    save_dir="results/models/sensitivity",
    output_dir="results",
    train_seeds=(0, 1, 2, 3, 4),
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

    # 1. Train PPO and DQN models for 100k steps
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

    # 2. Evaluate on the 50 shared evaluation episodes
    eval_records = []
    for algo in algos:
        for s in train_seeds:
            model_path = os.path.join(save_dir, f"{algo}_100k_seed{s}.zip")
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

    # Build evaluation episode score table aligned by seed
    per_ep_dict = {
        "PPO_100k": df_100k[df_100k["agent"] == "PPO_100k"].groupby("seed")["score"].mean().sort_index(),
        "DQN_100k": df_100k[df_100k["agent"] == "DQN_100k"].groupby("seed")["score"].mean().sort_index(),
        "AlwaysOn_24": base_df[base_df["agent"] == "AlwaysOn_24"].set_index("seed")["score"].sort_index(),
        "AdaptedRuleAgent": base_df[base_df["agent"] == "AdaptedRuleAgent"].set_index("seed")["score"].sort_index(),
        "PPO_20k": base_df[base_df["agent"] == "PPO"].groupby("seed")["score"].mean().sort_index(),
        "DQN_20k": base_df[base_df["agent"] == "DQN"].groupby("seed")["score"].mean().sort_index(),
    }
    ep_df = pd.DataFrame(per_ep_dict)
    assert not ep_df.isna().any().any(), "Evaluation table has missing seeds!"

    # 4. Statistical Testing across sensitivity family
    comparisons = [
        ("PPO_100k", "AlwaysOn_24"),
        ("PPO_100k", "AdaptedRuleAgent"),
        ("PPO_100k", "PPO_20k"),
        ("DQN_100k", "AlwaysOn_24"),
        ("DQN_100k", "AdaptedRuleAgent"),
        ("DQN_100k", "PPO_100k"),
        ("DQN_100k", "DQN_20k"),
    ]

    stats_rows = []
    p_vals = []
    for a, b in comparisons:
        sa = ep_df[a]
        sb = ep_df[b]
        diff = sa - sb
        w_stat, p_val = stats.wilcoxon(sa, sb)
        p_vals.append(p_val)
        stats_rows.append({
            "Comparison": f"{a} vs {b}",
            "Median_Paired_Diff": float(np.median(diff)),
            "Mean_Paired_Diff": float(np.mean(diff)),
            "Wins": int((diff > 0).sum()),
            "Total_Episodes": len(diff),
            "Wilcoxon_W": float(w_stat),
            "P_Value_Raw": float(p_val),
        })

    adj_p = holm_bonferroni(p_vals)
    for r, p_adj in zip(stats_rows, adj_p):
        r["P_Value_Holm"] = float(p_adj)
        r["Significant_p05"] = bool(p_adj < 0.05)

    stats_df = pd.DataFrame(stats_rows)

    # 5. Aggregate Summary Table for 100k vs Baselines
    summary_rows = [
        {
            "Agent": "PPO_100k (seeds 0-4)",
            "Score_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["score"].mean()),
            "Score_SD": float(df_100k[df_100k["agent"] == "PPO_100k"]["score"].std()),
            "Comfort_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["comfort_ratio"].mean()),
            "Cost_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["total_cost"].mean()),
            "Switches_Mean": float(df_100k[df_100k["agent"] == "PPO_100k"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "DQN_100k (seeds 0-4)",
            "Score_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["score"].mean()),
            "Score_SD": float(df_100k[df_100k["agent"] == "DQN_100k"]["score"].std()),
            "Comfort_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["comfort_ratio"].mean()),
            "Cost_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["total_cost"].mean()),
            "Switches_Mean": float(df_100k[df_100k["agent"] == "DQN_100k"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "AlwaysOn_24",
            "Score_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "AlwaysOn_24"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["comfort_ratio"].mean()),
            "Cost_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["total_cost"].mean()),
            "Switches_Mean": float(base_df[base_df["agent"] == "AlwaysOn_24"]["ac_switch_count"].mean()),
        },
        {
            "Agent": "AdaptedRuleAgent",
            "Score_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["score"].mean()),
            "Score_SD": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["score"].std()),
            "Comfort_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["comfort_ratio"].mean()),
            "Cost_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["total_cost"].mean()),
            "Switches_Mean": float(base_df[base_df["agent"] == "AdaptedRuleAgent"]["ac_switch_count"].mean()),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    # 6. Save sensitivity_100k.csv
    sens_csv = os.path.join(output_dir, "sensitivity_100k.csv")
    stats_df.to_csv(sens_csv, index=False)
    print(f"\nSensitivity statistical test results saved to {sens_csv}")

    sens_summary_csv = os.path.join(output_dir, "sensitivity_100k_summary.csv")
    summary_df.to_csv(sens_summary_csv, index=False)
    print(f"Sensitivity summary saved to {sens_summary_csv}")

    print("\n=== 100K SENSITIVITY BENCHMARK SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== PAIRED STATISTICAL TESTS (100K vs Baselines) ===")
    print(stats_df.to_string(index=False))

    return summary_df, stats_df


if __name__ == "__main__":
    train_and_eval_sensitivity()
