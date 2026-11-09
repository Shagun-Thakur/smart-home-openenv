import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.realistic_env import RealisticSmartHomeEnv

class LearningCurveCallback(BaseCallback):
    """Callback to record episodic rewards during training."""
    def __init__(self, algo_name, seed, verbose=0):
        super().__init__(verbose)
        self.algo_name = algo_name
        self.seed = seed
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self._cur_ep_rew = 0.0
        self._cur_ep_len = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        self._cur_ep_rew += float(rewards[0])
        self._cur_ep_len += 1

        if dones[0]:
            self.episode_rewards.append(self._cur_ep_rew)
            self.episode_lengths.append(self._cur_ep_len)
            self.timesteps.append(self.num_timesteps)
            self._cur_ep_rew = 0.0
            self._cur_ep_len = 0
        return True


def train_single(algo="ppo", seed=0, config_path="configs/realistic.yaml", total_timesteps=25000, save_dir="results"):
    models_dir = os.path.join(save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Initialize environment with seed
    env = RealisticSmartHomeEnv(config=cfg)
    env.reset(seed=seed)

    cb = LearningCurveCallback(algo_name=algo, seed=seed)

    if algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=3e-4,
            n_steps=240,       # 5 full days per rollout buffer
            batch_size=48,     # 1 day batch size
            n_epochs=10,
            gamma=0.99,
        )
    elif algo.lower() == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=480,   # 10 days warm-up
            batch_size=64,
            gamma=0.99,
            target_update_interval=240,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}. Must be 'ppo' or 'dqn'.")

    print(f"[{algo.upper()}] Training with seed={seed} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=cb)

    model_path = os.path.join(models_dir, f"{algo.lower()}_seed{seed}.zip")
    model.save(model_path)
    print(f"[{algo.upper()}] Model saved to {model_path}")

    # Return curve dataframe
    curve_df = pd.DataFrame({
        "algo": algo.lower(),
        "seed": seed,
        "timestep": cb.timesteps,
        "episode_reward": cb.episode_rewards,
        "episode_length": cb.episode_lengths,
    })
    return model, curve_df


def train_all_seeds(config_path="configs/realistic.yaml", total_timesteps=25000, save_dir="results"):
    all_curves = []
    for algo in ["ppo", "dqn"]:
        for seed in range(10):
            _, df = train_single(
                algo=algo,
                seed=seed,
                config_path=config_path,
                total_timesteps=total_timesteps,
                save_dir=save_dir
            )
            all_curves.append(df)

    combined_curves = pd.concat(all_curves, ignore_index=True)
    curves_csv = os.path.join(save_dir, "learning_curves.csv")
    combined_curves.to_csv(curves_csv, index=False)
    print(f"\nAll learning curves saved to {curves_csv}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO or DQN on Realistic Smart Home Env")
    parser.add_argument("--config", type=str, default="configs/realistic.yaml", help="Path to config yaml")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="RL algorithm")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=25000, help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save artifacts")
    parser.add_argument("--train-all", action="store_true", help="Train all 10 seeds for both PPO and DQN")

    args = parser.parse_args()

    if args.train_all:
        train_all_seeds(config_path=args.config, total_timesteps=args.timesteps, save_dir=args.save_dir)
    else:
        train_single(
            algo=args.algo,
            seed=args.seed,
            config_path=args.config,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()
