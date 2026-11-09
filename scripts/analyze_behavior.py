import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.realistic_env import RealisticSmartHomeEnv
from env.environment import decode_action
from agents.baseline_agent import (
    AdaptedRuleAgent,
    RandomAgent,
    AlwaysOnAgent,
    OccupancyAware_24,
    OccupancyAware_Precool,
    OccupancyAware_TOU,
)


def get_tariff_tier(price):
    if price >= 1.99:
        return "peak"
    elif price >= 0.99:
        return "mid"
    else:
        return "off_peak"


def run_episode_steps(env, agent_fn, seed, train_seed=None):
    obs, info = env.reset(seed=seed)
    done = False
    steps = []
    ep_id = f"{seed}_{train_seed}" if train_seed is not None else f"{seed}_0"

    while not done:
        action = agent_fn(obs)
        if isinstance(action, (int, np.integer)):
            ac_act, bat_act = decode_action(int(action))
        else:
            ac_act, bat_act = action

        current_price = float(info["electricity_price"])
        current_soc = float(info["battery_soc"])
        tier = get_tariff_tier(current_price)

        obs, reward, term, trunc, step_info = env.step(action)
        done = term or trunc
        info = step_info

        steps.append({
            "ep_id": ep_id,
            "seed": seed,
            "train_seed": train_seed,
            "step": step_info["step"],
            "hour": step_info["hour"],
            "tier": tier,
            "price": current_price,
            "soc": current_soc,
            "ac_action": ac_act if ac_act is not None else "off",
            "battery_action": bat_act,
            "occupancy": int(step_info["occupancy"]),
            "indoor_temp": float(step_info["indoor_temp"]),
            "grid_energy": float(step_info["energy"]),
            "cost": float(step_info["cost"]),
        })

    return steps


def analyze_agent_steps(step_records):
    df_steps = pd.DataFrame(step_records)

    # 1. Battery action fractions by tariff tier
    tiers = ["off_peak", "mid", "peak"]
    bat_actions = ["CHARGE", "DISCHARGE", "IDLE"]
    metrics = {}

    for t in tiers:
        sub_t = df_steps[df_steps["tier"] == t]
        total_t = len(sub_t) if len(sub_t) > 0 else 1
        for ba in bat_actions:
            frac = (sub_t["battery_action"] == ba).sum() / total_t
            metrics[f"{t}_{ba.lower()}_frac"] = float(frac)
        metrics[f"{t}_mean_soc"] = float(sub_t["soc"].mean()) if len(sub_t) > 0 else 0.0

    # 2. Total peak grid energy (kWh) per episode (averaged across all unique episode runs)
    peak_steps = df_steps[df_steps["tier"] == "peak"]
    peak_energy_per_ep = peak_steps.groupby("ep_id")["grid_energy"].sum()
    metrics["peak_grid_energy_kwh"] = float(peak_energy_per_ep.mean()) if len(peak_energy_per_ep) > 0 else 0.0

    # 3. AC setpoint distribution across all steps
    total_steps = len(df_steps) if len(df_steps) > 0 else 1
    for ac in [22, 24, 26, "off"]:
        frac = (df_steps["ac_action"] == ac).sum() / total_steps
        metrics[f"ac_{ac}_frac"] = float(frac)

    # 4. Mean indoor temperature when occupied
    occupied_steps = df_steps[df_steps["occupancy"] == 1]
    metrics["occupied_mean_temp"] = float(occupied_steps["indoor_temp"].mean()) if len(occupied_steps) > 0 else 0.0

    return metrics


def analyze_all_behavior(
    config_path="configs/realistic.yaml",
    models_dir="results/models",
    output_dir="results",
    eval_seeds_count=50,
    start_eval_seed=1000
):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = RealisticSmartHomeEnv(config=cfg)
    eval_seeds = list(range(start_eval_seed, start_eval_seed + eval_seeds_count))

    # 1. Baselines
    baselines = {
        "AlwaysOn_24": lambda obs: AlwaysOnAgent(24).act(obs),
        "AdaptedRuleAgent": lambda obs: AdaptedRuleAgent().act(obs),
        "OccupancyAware_24": lambda obs: OccupancyAware_24().act(obs),
        "OccupancyAware_Precool": lambda obs: OccupancyAware_Precool(t_hot=26.0).act(obs),
        "OccupancyAware_TOU": lambda obs: OccupancyAware_TOU().act(obs),
        "Random": lambda obs: RandomAgent(seed=int(obs[0] * 100)).act(obs),
    }

    baseline_steps = {name: [] for name in baselines}
    for name, agent_fn in baselines.items():
        print(f"Collecting behavior data for {name} on {len(eval_seeds)} episodes...")
        for s in eval_seeds:
            steps = run_episode_steps(env, agent_fn, s)
            baseline_steps[name].extend(steps)

    # 2. Learned PPO & DQN Models
    ppo_all_steps = []
    dqn_all_steps = []
    dqn_seed1_steps = []
    dqn_seed8_steps = []
    dqn_best_steps = []  # seed 3 is best DQN

    print("Collecting behavior data for PPO models (seeds 0-9)...")
    for t_seed in range(10):
        model_path = os.path.join(models_dir, f"ppo_seed{t_seed}.zip")
        if not os.path.exists(model_path):
            continue
        model = PPO.load(model_path)
        agent_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
        for s in eval_seeds:
            steps = run_episode_steps(env, agent_fn, s, train_seed=t_seed)
            ppo_all_steps.extend(steps)

    print("Collecting behavior data for DQN models (seeds 0-9)...")
    for t_seed in range(10):
        model_path = os.path.join(models_dir, f"dqn_seed{t_seed}.zip")
        if not os.path.exists(model_path):
            continue
        model = DQN.load(model_path)
        agent_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
        for s in eval_seeds:
            steps = run_episode_steps(env, agent_fn, s, train_seed=t_seed)
            dqn_all_steps.extend(steps)
            if t_seed == 1:
                dqn_seed1_steps.extend(steps)
            if t_seed == 8:
                dqn_seed8_steps.extend(steps)
            if t_seed == 3:
                dqn_best_steps.extend(steps)

    all_groups = {
        "AlwaysOn_24": baseline_steps["AlwaysOn_24"],
        "AdaptedRuleAgent": baseline_steps["AdaptedRuleAgent"],
        "OccupancyAware_24": baseline_steps["OccupancyAware_24"],
        "OccupancyAware_Precool": baseline_steps["OccupancyAware_Precool"],
        "OccupancyAware_TOU": baseline_steps["OccupancyAware_TOU"],
        "Random": baseline_steps["Random"],
        "PPO (all 10 seeds)": ppo_all_steps,
        "DQN (all 10 seeds)": dqn_all_steps,
        "DQN (seed 1 - worst)": dqn_seed1_steps,
        "DQN (seed 8 - worst)": dqn_seed8_steps,
        "DQN (seed 3 - best)": dqn_best_steps,
    }

    summary_rows = []
    for grp_name, steps in all_groups.items():
        if len(steps) == 0:
            continue
        m = analyze_agent_steps(steps)
        m["Agent_Group"] = grp_name
        summary_rows.append(m)

    summary_df = pd.DataFrame(summary_rows)
    # Reorder columns with Agent_Group first
    cols = ["Agent_Group"] + [c for c in summary_df.columns if c != "Agent_Group"]
    summary_df = summary_df[cols]

    csv_path = os.path.join(output_dir, "behavior_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nBehavior summary saved to {csv_path}")
    print(summary_df.to_string(index=False))

    # Generate Figure: results/behavior_by_tariff.png
    plot_behavior(summary_df, output_dir)
    return summary_df


def plot_behavior(df, output_dir):
    agents = [
        "AlwaysOn_24",
        "AdaptedRuleAgent",
        "OccupancyAware_24",
        "OccupancyAware_Precool",
        "OccupancyAware_TOU",
        "Random",
        "PPO (all 10 seeds)",
        "DQN (all 10 seeds)",
        "DQN (seed 1 - worst)",
        "DQN (seed 8 - worst)",
        "DQN (seed 3 - best)",
    ]
    df_plot = df.set_index("Agent_Group").reindex(agents).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(df_plot))

    # 1. Discharge fraction across tariff tiers
    ax = axes[0, 0]
    w = 0.25
    ax.bar(x - w, df_plot["off_peak_discharge_frac"] * 100, width=w, label="Off-Peak ($0.5)", color="#2ca02c")
    ax.bar(x, df_plot["mid_discharge_frac"] * 100, width=w, label="Mid-Peak ($1.0)", color="#ff7f0e")
    ax.bar(x + w, df_plot["peak_discharge_frac"] * 100, width=w, label="Peak ($2.0)", color="#d62728")
    ax.set_title("Battery Discharge Rate by Tariff Tier (%)", fontweight="bold")
    ax.set_ylabel("Discharge Action %")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Mean Battery SoC by Tariff Tier
    ax = axes[0, 1]
    ax.bar(x - w, df_plot["off_peak_mean_soc"], width=w, label="Off-Peak ($0.5)", color="#2ca02c")
    ax.bar(x, df_plot["mid_mean_soc"], width=w, label="Mid-Peak ($1.0)", color="#ff7f0e")
    ax.bar(x + w, df_plot["peak_mean_soc"], width=w, label="Peak ($2.0)", color="#d62728")
    ax.set_title("Mean Battery State of Charge (SoC %) by Tariff Tier", fontweight="bold")
    ax.set_ylabel("Mean SoC (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Peak Grid Energy (kWh)
    ax = axes[1, 0]
    bars = ax.bar(x, df_plot["peak_grid_energy_kwh"], color="#1f77b4", edgecolor="black")
    ax.set_title("Total Peak Grid Energy Consumption (kWh/day)", fontweight="bold")
    ax.set_ylabel("Peak Energy (kWh)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=35, ha="right", fontsize=8)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. AC Setpoint Distribution
    ax = axes[1, 1]
    bottom = np.zeros(len(df_plot))
    for ac, col in [(22, "#1f77b4"), (24, "#2ca02c"), (26, "#ff7f0e"), ("off", "#7f7f7f")]:
        vals = df_plot[f"ac_{ac}_frac"] * 100
        ax.bar(x, vals, bottom=bottom, label=f"AC {ac}", color=col, edgecolor="white", width=0.6)
        bottom += vals
    ax.set_title("AC Setpoint Action Distribution (%)", fontweight="bold")
    ax.set_ylabel("Fraction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=35, ha="right", fontsize=8)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "behavior_by_tariff.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Behavior diagnostic figure saved to {fig_path}")


if __name__ == "__main__":
    analyze_all_behavior()
