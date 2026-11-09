"""Analyze arrival dynamics and pre-cooling behavior across agents on 50 evaluation episodes.

Computes metrics for every vacant-to-occupied transition:
- Indoor temperature at arrival (°C)
- Number of steps until indoor temperature is within [22.0, 26.0]°C after arrival
- AC setpoint distribution in the 4 steps before arrival
- Fraction of pre-arrival steps with AC on

Saves:
- results/arrival_behavior.csv
- results/arrival_behavior.png
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.realistic_env import RealisticSmartHomeEnv
from env.environment import decode_action
from agents.baseline_agent import (
    AlwaysOnAgent,
    OccupancyAware_24,
    OccupancyAware_Precool,
    OccupancyAware_TOU,
    AdaptedRuleAgent,
)
from scripts.stats_utils import load_checkpoint_manifest, verify_checkpoint


def run_arrival_evaluation(
    config_path="configs/realistic.yaml",
    models_dir="results/models",
    output_dir="results",
    eval_seeds_count=50,
    start_eval_seed=1000,
):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = RealisticSmartHomeEnv(config=cfg)
    eval_seeds = list(range(start_eval_seed, start_eval_seed + eval_seeds_count))

    # Manifest verification
    manifest_path = os.path.join(models_dir, "MANIFEST.sha256")
    manifest = load_checkpoint_manifest(manifest_path)

    # 1. Define Rule Baselines
    agents = {
        "AlwaysOn_24": lambda obs: AlwaysOnAgent(24).act(obs),
        "OccupancyAware_24": lambda obs: OccupancyAware_24().act(obs),
        "OccupancyAware_Precool": lambda obs: OccupancyAware_Precool().act(obs),
        "OccupancyAware_TOU": lambda obs: OccupancyAware_TOU().act(obs),
        "AdaptedRuleAgent": lambda obs: AdaptedRuleAgent().act(obs),
    }

    all_agent_transitions = {}

    def collect_transitions_for_agent_fn(agent_fn):
        transitions = []
        for s in eval_seeds:
            obs, _ = env.reset(seed=s)
            history = []
            done = False
            while not done:
                temp = float(obs[0])
                occ = int(round(obs[4]))
                act = agent_fn(obs)
                ac_act, bat_act = decode_action(act) if isinstance(act, (int, np.integer)) else act
                history.append({"temp": temp, "occ": occ, "ac": ac_act})

                obs, rew, term, trunc, info = env.step(act)
                done = term or trunc

            # Extract arrival transitions
            for i in range(1, len(history)):
                if history[i]["occ"] == 1 and history[i - 1]["occ"] == 0:
                    arr_temp = history[i]["temp"]
                    pre = history[max(0, i - 4) : i]
                    ac_on = sum(1 for h in pre if h["ac"] is not None)
                    pre_frac = ac_on / len(pre) if pre else 0.0
                    ac_22 = sum(1 for h in pre if h["ac"] == 22) / len(pre) if pre else 0.0
                    ac_24 = sum(1 for h in pre if h["ac"] == 24) / len(pre) if pre else 0.0
                    ac_26 = sum(1 for h in pre if h["ac"] == 26) / len(pre) if pre else 0.0
                    ac_off = sum(1 for h in pre if h["ac"] is None) / len(pre) if pre else 0.0

                    # Steps to comfort
                    steps_rec = 0
                    for j in range(i, len(history)):
                        if history[j]["occ"] == 0:
                            break
                        if 22.0 <= history[j]["temp"] <= 26.0:
                            break
                        steps_rec += 1

                    transitions.append({
                        "arr_temp": arr_temp,
                        "is_comf_arr": (22.0 <= arr_temp <= 26.0),
                        "steps_to_comf": steps_rec,
                        "pre_on_frac": pre_frac,
                        "pre_22": ac_22,
                        "pre_24": ac_24,
                        "pre_26": ac_26,
                        "pre_off": ac_off,
                    })
        return transitions

    for name, agent_fn in agents.items():
        print(f"Collecting arrival dynamics for {name}...")
        all_agent_transitions[name] = collect_transitions_for_agent_fn(agent_fn)

    # 2. Evaluate Learned Models (PPO_20k, PPO_100k, DQN_20k, DQN_100k across all 10 seeds)
    from stable_baselines3 import PPO, DQN

    learned_configs = [
        ("PPO_20k", PPO, models_dir, "ppo_seed", 10),
        ("PPO_100k", PPO, os.path.join(models_dir, "sensitivity"), "ppo_100k_seed", 10),
        ("DQN_20k", DQN, models_dir, "dqn_seed", 10),
        ("DQN_100k", DQN, os.path.join(models_dir, "sensitivity"), "dqn_100k_seed", 10),
    ]

    for label, model_cls, folder, prefix, num_seeds in learned_configs:
        print(f"Collecting arrival dynamics for {label} (all {num_seeds} seeds)...")
        combined_trans = []
        for s in range(num_seeds):
            m_path = os.path.join(folder, f"{prefix}{s}.zip")
            verify_checkpoint(m_path, models_dir=models_dir, manifest=manifest)
            model = model_cls.load(m_path)
            agent_fn = lambda obs, m=model: int(m.predict(obs, deterministic=True)[0])
            combined_trans.extend(collect_transitions_for_agent_fn(agent_fn))
        all_agent_transitions[label] = combined_trans

    # 3. Aggregate Summary Table
    summary_rows = []
    display_order = [
        "AlwaysOn_24",
        "OccupancyAware_Precool",
        "OccupancyAware_TOU",
        "AdaptedRuleAgent",
        "OccupancyAware_24",
        "PPO_20k",
        "PPO_100k",
        "DQN_20k",
        "DQN_100k",
    ]

    for name in display_order:
        trans = all_agent_transitions[name]
        arr_temps = [t["arr_temp"] for t in trans]
        steps = [t["steps_to_comf"] for t in trans]
        pre_on = [t["pre_on_frac"] for t in trans]
        pre_22 = [t["pre_22"] for t in trans]
        pre_24 = [t["pre_24"] for t in trans]
        pre_26 = [t["pre_26"] for t in trans]
        pre_off = [t["pre_off"] for t in trans]
        is_comf = [1.0 if t["is_comf_arr"] else 0.0 for t in trans]

        summary_rows.append({
            "Agent_Group": name,
            "Arrival_Temp_Mean": float(np.mean(arr_temps)),
            "Arrival_Temp_SD": float(np.std(arr_temps)),
            "Steps_To_Comfort_Mean": float(np.mean(steps)),
            "Comfortable_At_Arrival_Pct": float(np.mean(is_comf) * 100.0),
            "Pre_Arrival_AC_On_Pct": float(np.mean(pre_on) * 100.0),
            "Pre_Arrival_22_Pct": float(np.mean(pre_22) * 100.0),
            "Pre_Arrival_24_Pct": float(np.mean(pre_24) * 100.0),
            "Pre_Arrival_26_Pct": float(np.mean(pre_26) * 100.0),
            "Pre_Arrival_Off_Pct": float(np.mean(pre_off) * 100.0),
            "Total_Transitions": len(trans),
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "arrival_behavior.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nArrival behavior summary saved to {csv_path}")
    print(summary_df.to_string(index=False))

    # 4. Generate Diagnostic Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    names = summary_df["Agent_Group"]
    x = np.arange(len(names))

    # Colors
    colors = [
        "#1f77b4" if "AlwaysOn" in n else
        "#2ca02c" if "Occupancy" in n else
        "#d62728" if "Adapted" in n else
        "#ff7f0e" if "PPO" in n else "#9467bd"
        for n in names
    ]

    # Subplot 1: Arrival Temperature
    axes[0].bar(x, summary_df["Arrival_Temp_Mean"], yerr=summary_df["Arrival_Temp_SD"], capsize=4, color=colors, alpha=0.85)
    axes[0].axhline(26.0, color="crimson", linestyle="--", linewidth=1.5, label="Comfort Upper Limit (26°C)")
    axes[0].axhline(22.0, color="navy", linestyle=":", linewidth=1.5, label="Comfort Lower Limit (22°C)")
    axes[0].set_ylabel("Indoor Temperature at Arrival (°C)", fontsize=11)
    axes[0].set_title("(a) Arrival Indoor Temperature", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylim(20.0, 30.0)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Subplot 2: Pre-Arrival AC On Fraction (%)
    axes[1].bar(x, summary_df["Pre_Arrival_AC_On_Pct"], color=colors, alpha=0.85)
    axes[1].set_ylabel("Pre-Arrival Steps with AC On (%)", fontsize=11)
    axes[1].set_title("(b) Pre-Arrival Cooling Activity (Past 4 Steps)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.3)

    # Subplot 3: Recovery Steps to Comfort
    axes[2].bar(x, summary_df["Steps_To_Comfort_Mean"], color=colors, alpha=0.85)
    axes[2].set_ylabel("Steps Until Inside Comfort Band (30 min / step)", fontsize=11)
    axes[2].set_title("(c) Thermal Recovery Delay Post-Arrival", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes[2].set_ylim(0, 2.5)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "arrival_behavior.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Arrival diagnostic figure saved to {fig_path}")

    return summary_df


if __name__ == "__main__":
    run_arrival_evaluation()
