import pytest
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
from scripts.stats_utils import compute_per_episode_scores


def test_stats_pairing_preserves_seed_alignment():
    """Verify that compute_per_episode_scores maintains strict seed alignment

    and that sorting scores breaks pairing on a synthetic non-monotonic dataset.
    """
    # Synthetic dataset across 4 evaluation seeds [100, 101, 102, 103]
    # Agent A beats Agent B by exactly +0.1 on every single seed.
    records = []
    seeds = [100, 101, 102, 103]
    a_scores = [0.2, 0.9, 0.3, 0.8]  # Non-monotonic scores
    b_scores = [0.1, 0.8, 0.2, 0.7]  # A beats B by +0.1 on every seed

    for s, score_a, score_b in zip(seeds, a_scores, b_scores):
        # Learned agent PPO with 2 training seeds
        records.append({"agent": "PPO", "train_seed": 0, "seed": s, "score": score_a})
        records.append({"agent": "PPO", "train_seed": 1, "seed": s, "score": score_a})
        # Baseline agent
        records.append({"agent": "AdaptedRuleAgent", "train_seed": None, "seed": s, "score": score_b})

    df = pd.DataFrame(records)

    # Compute aligned evaluation table
    ep_df = compute_per_episode_scores(df)

    # 1. Check index alignment
    assert list(ep_df.index) == seeds
    assert (ep_df["PPO"].index == ep_df["AdaptedRuleAgent"].index).all()

    # 2. Check true paired differences
    aligned_diffs = ep_df["PPO"] - ep_df["AdaptedRuleAgent"]
    expected_diffs = np.array([0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(aligned_diffs.values, expected_diffs, atol=1e-6)

    # 3. Demonstrate that the bug (.sort_values().values) changes the paired differences
    buggy_ppo = ep_df["PPO"].sort_values().values  # [0.2, 0.3, 0.8, 0.9]
    rule_by_seed = ep_df["AdaptedRuleAgent"].values  # [0.1, 0.8, 0.2, 0.7]
    buggy_diffs = buggy_ppo - rule_by_seed  # [0.1, -0.5, 0.6, 0.2]

    # Verify that buggy sorting produces an incorrect negative difference at seed 101
    assert buggy_diffs[1] < 0  # Erroneously reports A lost seed 101
    assert not np.allclose(buggy_diffs, aligned_diffs.values)
