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
from agents.baseline_agent import (
    BaselineAgent,
    AdaptedRuleAgent,
    RandomAgent,
    AlwaysOnAgent
)

MAX_POSSIBLE_COST = 144.0  # 48 steps * 0.5 hr * 3.0 kW max AC * 2.0 peak tariff

def run_eval_episode(env, agent_fn, seed):
    obs, info = env.reset(seed=seed)
    done = False

    total_cost = 0.0
    occupied_steps = 0
    comfort_occupied_steps = 0
    ac_switch_count = 0
    battery_violations = 0
    prev_ac = None

    while not done:
        action = agent_fn(obs)

        # Decode for switching and violation diagnostics
        if isinstance(action, (int, np.integer)):
            ac_act, bat_act = decode_action(int(action))
        else:
            ac_act, bat_act = action

        if prev_ac is not None and prev_ac != ac_act:
            ac_switch_count += 1
        prev_ac = ac_act

        obs, reward, term, trunc, step_info = env.step(action)
        done = term or trunc

        total_cost += step_info["cost"]
        if step_info["occupancy"] == 1:
            occupied_steps += 1
            if step_info["is_comfortable"]:
                comfort_occupied_steps += 1

        # Check battery physical bounds (0% to 100%)
        if step_info["battery_soc"] < 0.0 or step_info["battery_soc"] > 100.0:
            battery_violations += 1

    comfort_ratio = (comfort_occupied_steps / occupied_steps) if occupied_steps > 0 else 1.0
    norm_cost = min(1.0, total_cost / MAX_POSSIBLE_COST)
    score = 0.6 * comfort_ratio + 0.4 * (1.0 - norm_cost)

    return {
        "seed": seed,
        "total_cost": float(total_cost),
        "comfort_ratio": float(comfort_ratio),
        "normalized_cost": float(norm_cost),
        "score": float(score),
        "ac_switch_count": int(ac_switch_count),
        "battery_violations": int(battery_violations),
        "occupied_steps": int(occupied_steps),
    }


def compute_per_episode_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame indexed by evaluation episode seed with agents as columns.
    For learned agents (PPO, DQN), score is averaged across all training seeds
    for each evaluation episode seed.
    For rule/heuristic baselines, the single episode score is used.
    """
    agents = df["agent"].unique()
    per_ep_dict = {}
    for agent in agents:
        agent_df = df[df["agent"] == agent]
        if agent in ["PPO", "DQN"]:
            s = agent_df.groupby("seed")["score"].mean().sort_index()
        else:
            s = agent_df.set_index("seed")["score"].sort_index()
        per_ep_dict[agent] = s

    ep_df = pd.DataFrame(per_ep_dict)
    assert not ep_df.isna().any().any(), "Evaluation table has missing seeds!"
    return ep_df


def evaluate_all(
    config_path="configs/realistic.yaml",
    models_dir="results/models",
    output_dir="results",
    eval_seeds_count=50,
    start_eval_seed=1000,
    reuse_metrics=False,
):
    os.makedirs(output_dir, exist_ok=True)
    metrics_csv = os.path.join(output_dir, "evaluation_metrics.csv")

    if reuse_metrics and os.path.exists(metrics_csv):
        print(f"Loading existing metrics from {metrics_csv}...")
        df = pd.read_csv(metrics_csv)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        env = RealisticSmartHomeEnv(config=cfg)
        eval_seeds = list(range(start_eval_seed, start_eval_seed + eval_seeds_count))

        all_records = []

        # 1. Evaluate Rule & Heuristic Baselines
        baselines = {
            "Random": lambda obs: RandomAgent(seed=int(obs[0] * 100)).act(obs),
            "AlwaysOn_22": lambda obs: AlwaysOnAgent(22).act(obs),
            "AlwaysOn_24": lambda obs: AlwaysOnAgent(24).act(obs),
            "AlwaysOn_26": lambda obs: AlwaysOnAgent(26).act(obs),
            "BaselineAgent (Rule)": lambda obs: BaselineAgent().act(obs),
            "AdaptedRuleAgent": lambda obs: AdaptedRuleAgent().act(obs),
        }

        for name, agent_fn in baselines.items():
            print(f"Evaluating {name} on {len(eval_seeds)} episodes...")
            for s in eval_seeds:
                rec = run_eval_episode(env, agent_fn, s)
                rec["agent"] = name
                rec["group"] = "baseline"
                rec["train_seed"] = None
                all_records.append(rec)

        # 2. Evaluate Trained PPO & DQN Models (Seeds 0-9)
        for algo in ["ppo", "dqn"]:
            for t_seed in range(10):
                model_file = os.path.join(models_dir, f"{algo}_seed{t_seed}.zip")
                if not os.path.exists(model_file):
                    print(f"Warning: Model file not found: {model_file}, skipping.")
                    continue

                print(f"Evaluating {algo.upper()} (train seed {t_seed}) on {len(eval_seeds)} episodes...")
                if algo == "ppo":
                    model = PPO.load(model_file)
                else:
                    model = DQN.load(model_file)

                def make_act(m):
                    return lambda obs: int(m.predict(obs, deterministic=True)[0])

                agent_fn = make_act(model)

                for s in eval_seeds:
                    rec = run_eval_episode(env, agent_fn, s)
                    rec["agent"] = f"{algo.upper()}"
                    rec["group"] = algo
                    rec["train_seed"] = t_seed
                    all_records.append(rec)

        df = pd.DataFrame(all_records)
        df.to_csv(metrics_csv, index=False)
        print(f"\nDetailed episode evaluation metrics saved to {metrics_csv}")

    # 3. Aggregate Summary Table (mean ± SD across seeds/episodes)
    summary_rows = []
    for agent_name, grp in df.groupby("agent"):
        summary_rows.append({
            "Agent": agent_name,
            "Group": grp["group"].iloc[0],
            "Score_Mean": grp["score"].mean(),
            "Score_SD": grp["score"].std(),
            "Comfort_Mean": grp["comfort_ratio"].mean(),
            "Comfort_SD": grp["comfort_ratio"].std(),
            "Cost_Mean": grp["total_cost"].mean(),
            "Cost_SD": grp["total_cost"].std(),
            "NormCost_Mean": grp["normalized_cost"].mean(),
            "NormCost_SD": grp["normalized_cost"].std(),
            "SwitchCount_Mean": grp["ac_switch_count"].mean(),
            "BatteryViolations": grp["battery_violations"].sum(),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Score_Mean", ascending=False)
    summary_csv = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table saved to {summary_csv}")

    # 4. Statistical Testing (Paired Wilcoxon Signed-Rank Test)
    # Aligned strictly by evaluation seed
    ep_df = compute_per_episode_scores(df)

    stats_records = []
    for algo in ["PPO", "DQN"]:
        if algo not in ep_df.columns or "AdaptedRuleAgent" not in ep_df.columns:
            continue
        series_algo = ep_df[algo]
        series_rule = ep_df["AdaptedRuleAgent"]
        assert (series_algo.index == series_rule.index).all(), "Evaluation seed indices must be identical"

        paired_diffs = series_algo - series_rule
        median_diff = np.median(paired_diffs)
        mean_diff = np.mean(paired_diffs)

        stat_val, p_val = stats.wilcoxon(series_algo, series_rule)

        stats_records.append({
            "Comparison": f"{algo} vs AdaptedRuleAgent",
            "Median_Paired_Diff": float(median_diff),
            "Mean_Paired_Diff": float(mean_diff),
            "Wilcoxon_Statistic": float(stat_val),
            "P_Value": float(p_val),
            "Significant_p05": bool(p_val < 0.05),
        })

    stats_df = pd.DataFrame(stats_records)
    stats_csv = os.path.join(output_dir, "statistical_tests.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Statistical test results saved to {stats_csv}\n")

    print("=== EVALUATION BENCHMARK SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== PAIRED STATISTICAL TESTS (Wilcoxon Signed-Rank) ===")
    print(stats_df.to_string(index=False))

    return df, summary_df, stats_df

    print("=== EVALUATION BENCHMARK SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== PAIRED STATISTICAL TESTS (Wilcoxon Signed-Rank) ===")
    print(stats_df.to_string(index=False))

    return df, summary_df, stats_df


if __name__ == "__main__":
    evaluate_all()
