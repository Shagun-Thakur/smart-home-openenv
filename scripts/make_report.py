import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.realistic_env import RealisticSmartHomeEnv
from env.environment import decode_action
from agents.baseline_agent import AdaptedRuleAgent
from stable_baselines3 import PPO

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def generate_learning_curves_plot(curves_csv="results/learning_curves.csv", output_png="results/learning_curves.png"):
    if not os.path.exists(curves_csv):
        print(f"File not found: {curves_csv}, skipping learning curves plot.")
        return

    df = pd.read_csv(curves_csv)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = {"ppo": "#1f77b4", "dqn": "#ff7f0e"}

    for algo in ["ppo", "dqn"]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue

        # Bin timesteps into intervals to compute smooth mean and std across seeds
        bins = np.linspace(0, 25000, 50)
        sub["bin"] = pd.cut(sub["timestep"], bins=bins)
        binned = sub.groupby(["bin", "seed"], observed=False)["episode_reward"].mean().reset_index()
        stats_df = binned.groupby("bin", observed=False)["episode_reward"].agg(["mean", "std"]).reset_index()

        # Midpoint of bins
        bin_centers = [b.mid for b in stats_df["bin"]]
        mean_vals = stats_df["mean"].values
        std_vals = stats_df["std"].fillna(0.0).values

        ax.plot(bin_centers, mean_vals, label=algo.upper(), color=colors[algo], linewidth=2)
        ax.fill_between(bin_centers, mean_vals - std_vals, mean_vals + std_vals, color=colors[algo], alpha=0.2)

    ax.set_title("Training Learning Curves (Mean ± 1 SD over 10 Seeds)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Timesteps", fontsize=11)
    ax.set_ylabel("Episodic Return (Reward)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Saved learning curves plot to {output_png}")


def generate_cost_vs_comfort_plot(metrics_csv="results/evaluation_metrics.csv", output_png="results/cost_vs_comfort.png"):
    if not os.path.exists(metrics_csv):
        print(f"File not found: {metrics_csv}, skipping cost vs comfort plot.")
        return

    df = pd.read_csv(metrics_csv)
    summary = df.groupby("agent").agg(
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        comfort_mean=("comfort_ratio", "mean"),
        comfort_std=("comfort_ratio", "std"),
        group=("group", "first")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    color_map = {
        "Random": "#7f7f7f",
        "AlwaysOn_22": "#9467bd",
        "AlwaysOn_24": "#8c564b",
        "AlwaysOn_26": "#e377c2",
        "BaselineAgent (Rule)": "#bcbd22",
        "AdaptedRuleAgent": "#2ca02c",
        "OccupancyAware_24": "#17becf",
        "OccupancyAware_Precool": "#1b9e77",
        "OccupancyAware_TOU": "#d95f02",
        "PPO": "#1f77b4",
        "DQN": "#ff7f0e",
    }

    marker_map = {
        "baseline": "o",
        "ppo": "s",
        "dqn": "^"
    }

    for _, row in summary.iterrows():
        name = row["agent"]
        color = color_map.get(name, "#333333")
        grp = row["group"]
        marker = marker_map.get(grp, "o")

        ax.errorbar(
            row["cost_mean"],
            row["comfort_mean"],
            xerr=row["cost_std"],
            yerr=row["comfort_std"],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markersize=9,
            label=name,
            alpha=0.9
        )

        # Annotate name offset
        ax.annotate(
            name,
            (row["cost_mean"], row["comfort_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="semibold" if name in ["PPO", "DQN", "AdaptedRuleAgent"] else "normal"
        )

    ax.set_title("Cost vs. Comfort Trade-Off (50 Shared Evaluation Episodes)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Episode Cost ($) [Lower is Better]", fontsize=11)
    ax.set_ylabel("Occupied Comfort Ratio [Higher is Better]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.3, label="Optimal Comfort (1.0)")
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Saved cost vs comfort plot to {output_png}")


def generate_example_day_plot(config_path="configs/realistic.yaml", output_png="results/example_day_trajectory.png"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = RealisticSmartHomeEnv(config=cfg)
    agent = AdaptedRuleAgent()

    obs, info = env.reset(seed=1000)
    done = False

    hours = []
    indoor_temps = []
    outdoor_temps = []
    prices = []
    occupancies = []
    battery_socs = []
    ac_setpoints = []
    battery_actions = []

    while not done:
        action = agent.act(obs)
        ac_act, bat_act = action

        hours.append(info["hour"])
        indoor_temps.append(info["indoor_temp"])
        outdoor_temps.append(info["outdoor_temp"])
        prices.append(info["electricity_price"])
        occupancies.append(info["occupancy"])
        battery_socs.append(info["battery_soc"])
        ac_setpoints.append(ac_act if ac_act is not None else 0)
        battery_actions.append(bat_act)

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

    # Generate 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), dpi=300, sharex=True)

    # Panel 1: Temperature & Comfort bounds
    ax1.plot(hours, outdoor_temps, label="Outdoor Temp (°C)", color="#d62728", linestyle="--", linewidth=1.8)
    ax1.plot(hours, indoor_temps, label="Indoor Temp (°C)", color="#1f77b4", linewidth=2.2)
    ax1.axhspan(22, 26, color="green", alpha=0.15, label="Comfort Range [22°C - 26°C]")
    ax1.set_ylabel("Temperature (°C)", fontsize=10, fontweight="bold")
    ax1.set_title("24-Hour Profile — Example Day Simulation (Seed 1000)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 2: Electricity Price & Occupancy
    ax2_occ = ax2.twinx()
    ax2.step(hours, prices, where="post", color="#ff7f0e", linewidth=2.0, label="TOU Price ($/kWh)")
    ax2_occ.step(hours, occupancies, where="post", color="#2ca02c", linestyle=":", linewidth=1.8, label="Occupancy (1=Home)")
    ax2.set_ylabel("Price ($/kWh)", fontsize=10, fontweight="bold", color="#ff7f0e")
    ax2_occ.set_ylabel("Occupancy", fontsize=10, fontweight="bold", color="#2ca02c")
    ax2_occ.set_ylim(-0.1, 1.2)
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2_occ, labels_2_occ = ax2_occ.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2_occ, labels_2 + labels_2_occ, loc="upper right", frameon=True, fontsize=9)

    # Panel 3: Battery State of Charge & Actions
    ax3_ac = ax3.twinx()
    ax3.plot(hours, battery_socs, color="#9467bd", linewidth=2.0, label="Battery SoC (%)")
    ax3_ac.scatter(hours, ac_setpoints, color="#17becf", marker="s", s=30, label="AC Setpoint (0=OFF)")
    ax3.set_ylabel("Battery SoC (%)", fontsize=10, fontweight="bold", color="#9467bd")
    ax3_ac.set_ylabel("AC Setpoint (°C)", fontsize=10, fontweight="bold", color="#17becf")
    ax3.set_xlabel("Hour of Day", fontsize=11, fontweight="bold")
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 25, 2))
    ax3.set_ylim(-5, 105)
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3_ac, labels_3_ac = ax3_ac.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3_ac, labels_3 + labels_3_ac, loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Saved example day trajectory plot to {output_png}")


def make_report():
    os.makedirs("results", exist_ok=True)
    generate_learning_curves_plot()
    generate_cost_vs_comfort_plot()
    generate_example_day_plot()
    print("Report generation complete!")

if __name__ == "__main__":
    make_report()
