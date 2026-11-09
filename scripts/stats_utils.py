"""Pure statistics and data processing utilities for evaluation and sensitivity analysis.

Dependencies: numpy, pandas, scipy (no torch or stable_baselines3).
"""

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(p_values):
    """Compute Holm-Bonferroni adjusted p-values.

    p_adj[k] = min(1.0, max((m - j + 1) * p[j] for j <= k)) for sorted p-values.
    """
    m = len(p_values)
    if m == 0:
        return np.array([])
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adj_p = np.zeros(m)
    running_max = 0.0
    for k in range(m):
        val = (m - k) * sorted_p[k]
        running_max = max(running_max, val)
        adj_p[k] = min(1.0, running_max)
    result = np.zeros(m)
    result[sorted_indices] = adj_p
    return result


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


def bootstrap_mean_ci(data, num_bootstraps=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean of a 1D array."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(data, size=len(data), replace=True).mean() for _ in range(num_bootstraps)])
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return low, high


def load_checkpoint_manifest(manifest_path="results/models/MANIFEST.sha256"):
    """Load sha256 manifest file returning a dict mapping relative path to expected hash."""
    import os
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                h, p = parts
                manifest[p.replace("\\", "/")] = h.lower()
    return manifest


def verify_checkpoint(checkpoint_path, models_dir="results/models", manifest=None):
    """Verify that a checkpoint matches its expected sha256 hash in MANIFEST.sha256.

    Raises ValueError on mismatch, FileNotFoundError if missing.
    """
    import hashlib
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    if manifest is None:
        manifest_path = os.path.join(models_dir, "MANIFEST.sha256")
        manifest = load_checkpoint_manifest(manifest_path)

    rel_path = os.path.relpath(checkpoint_path, models_dir).replace("\\", "/")
    if rel_path not in manifest:
        raise ValueError(f"Checkpoint {rel_path} not registered in manifest at {models_dir}/MANIFEST.sha256!")

    with open(checkpoint_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().lower()

    expected_hash = manifest[rel_path].lower()
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 hash mismatch for checkpoint {rel_path}!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}\n"
            f"Checkpoint diverges from results/models/MANIFEST.sha256!"
        )
    return True
