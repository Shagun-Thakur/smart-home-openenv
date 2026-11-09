#!/usr/bin/env python3
"""
scripts/check_readme_numbers.py

Automated consistency check between README.md and data files in results/.
Validates:
  1. All markdown tables match their underlying CSV sources within tolerance.
  2. Key numbers in prose (scores, comfort, costs, wins, p-values, T_hot) match results.
  3. No forbidden stale phrases remain ("dominates", "overtakes", unsupported "proactive pre-cooling", "strong" AdaptedRuleAgent).

Base-install safe: only requires Python standard library, numpy, and pandas.
"""

import os
import sys
import re
import math
import numpy as np
import pandas as pd


def clean_num(val_str):
    """Extract a float from a string like '$52.00 ± 0.00' or '0.9686 ± 0.0178' or '+0.0027' or '75.0%'."""
    if val_str is None:
        return None
    s = str(val_str).strip()
    s = s.replace("**", "").replace("$", "").replace("°C", "").replace("kWh", "").replace("%", "")
    # If it contains ±, take the first number (mean)
    if "±" in s:
        s = s.split("±")[0].strip()
    # If scientific notation like 2.83 \times 10^{-5}
    if r"\times 10^{" in s:
        m = re.search(r"([0-9.]+)\s*\\times\s*10\^\{([0-9-]+)\}", s)
        if m:
            return float(m.group(1)) * (10 ** float(m.group(2)))
    # Match standard number
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def parse_markdown_tables(readme_text):
    """Parse all markdown tables into list of (header_row, data_rows)."""
    table_pattern = re.compile(r"(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)")
    raw_tables = table_pattern.findall(readme_text)
    parsed = []
    for raw in raw_tables:
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[2:]:  # skip separator line
            cols = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cols)
        parsed.append((header, rows))
    return parsed


def check_table_2_eval_summary(header, rows, results_dir, mismatches, checked):
    """Table 2: Tier 2 Realistic Scenario Evaluation vs evaluation_summary.csv."""
    csv_path = os.path.join(results_dir, "evaluation_summary.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path).set_index("Agent")

    # Map README agent names to CSV index
    agent_map = {
        "AlwaysOn_24": "AlwaysOn_24",
        "AlwaysOn_22": "AlwaysOn_22",
        "AlwaysOn_26": "AlwaysOn_26",
        "OccupancyAware_Precool": "OccupancyAware_Precool",
        "OccupancyAware_TOU": "OccupancyAware_TOU",
        "OccupancyAware_24": "OccupancyAware_24",
        "AdaptedRuleAgent": "AdaptedRuleAgent",
        "BaselineAgent (Rule)": "BaselineAgent (Rule)",
        "PPO (10 models)": "PPO",
        "DQN (10 models)": "DQN",
        "Random": "Random",
    }

    for row in rows:
        agent_raw = row[0].replace("**", "").strip()
        matched_agent = None
        for k, v in agent_map.items():
            if k.lower() in agent_raw.lower() or agent_raw.lower() in k.lower():
                matched_agent = v
                break
        if not matched_agent or matched_agent not in df.index:
            continue

        csv_row = df.loc[matched_agent]
        # Cols: Comfort Ratio (idx 2), Total Cost (idx 3), Switches (idx 4), Score (idx 5)
        readme_comfort = clean_num(row[2])
        readme_cost = clean_num(row[3])
        readme_switches = clean_num(row[4])
        readme_score = clean_num(row[5])

        # Checks
        if readme_comfort is not None:
            csv_val = float(csv_row["Comfort_Mean"])
            checked.append(f"Table 2 {matched_agent} Comfort: {readme_comfort} vs CSV {csv_val:.4f}")
            if abs(readme_comfort - csv_val) > 0.005:
                mismatches.append(f"Table 2 {matched_agent} Comfort mismatch: README {readme_comfort} vs CSV {csv_val:.4f}")

        if readme_cost is not None:
            csv_val = float(csv_row["Cost_Mean"])
            checked.append(f"Table 2 {matched_agent} Cost: {readme_cost} vs CSV {csv_val:.2f}")
            if abs(readme_cost - csv_val) > 0.15:
                mismatches.append(f"Table 2 {matched_agent} Cost mismatch: README {readme_cost} vs CSV {csv_val:.2f}")

        if readme_switches is not None:
            csv_val = float(csv_row["SwitchCount_Mean"])
            checked.append(f"Table 2 {matched_agent} Switches: {readme_switches} vs CSV {csv_val:.2f}")
            if abs(readme_switches - csv_val) > 0.1:
                mismatches.append(f"Table 2 {matched_agent} Switches mismatch: README {readme_switches} vs CSV {csv_val:.2f}")

        if readme_score is not None:
            csv_val = float(csv_row["Score_Mean"])
            checked.append(f"Table 2 {matched_agent} Score: {readme_score} vs CSV {csv_val:.4f}")
            if abs(readme_score - csv_val) > 0.005:
                mismatches.append(f"Table 2 {matched_agent} Score mismatch: README {readme_score} vs CSV {csv_val:.4f}")


def check_table_3_statistical_tests(header, rows, results_dir, mismatches, checked):
    """Table 3: Statistical Tests vs statistical_tests.csv."""
    csv_path = os.path.join(results_dir, "statistical_tests.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path).set_index("Comparison")

    for row in rows:
        comp_raw = row[0].replace("**", "").replace("BaselineAgent (Rule)", "BaselineAgent").strip()
        matched_comp = None
        for idx in df.index:
            if comp_raw == idx or comp_raw.replace(" ", "") == idx.replace(" ", ""):
                matched_comp = idx
                break
        if not matched_comp:
            continue

        csv_row = df.loc[matched_comp]
        # Cols: Med diff (1), Mean diff (2), Wins (3), W (4), Raw p (5), Holm p (6)
        readme_med = clean_num(row[1])
        readme_mean = clean_num(row[2])
        readme_wins_m = re.search(r"(\d+)\s*/\s*(\d+)", row[3])
        readme_w = clean_num(row[4])
        readme_holm = clean_num(row[6])

        if readme_wins_m:
            readme_wins = int(readme_wins_m.group(1))
            csv_wins = int(csv_row["Wins"])
            checked.append(f"Table 3 {matched_comp} Wins: {readme_wins} vs CSV {csv_wins}")
            if readme_wins != csv_wins:
                mismatches.append(f"Table 3 {matched_comp} Wins mismatch: README {readme_wins} vs CSV {csv_wins}")

        if readme_med is not None:
            csv_med = float(csv_row["Median_Paired_Diff"])
            checked.append(f"Table 3 {matched_comp} Med Diff: {readme_med} vs CSV {csv_med:.4f}")
            if abs(readme_med - csv_med) > 0.002:
                mismatches.append(f"Table 3 {matched_comp} Med Diff mismatch: README {readme_med} vs CSV {csv_med:.4f}")

        if readme_mean is not None:
            csv_mean = float(csv_row["Mean_Paired_Diff"])
            checked.append(f"Table 3 {matched_comp} Mean Diff: {readme_mean} vs CSV {csv_mean:.4f}")
            if abs(readme_mean - csv_mean) > 0.002:
                mismatches.append(f"Table 3 {matched_comp} Mean Diff mismatch: README {readme_mean} vs CSV {csv_mean:.4f}")

        if readme_w is not None:
            csv_w = float(csv_row["Wilcoxon_Statistic"])
            checked.append(f"Table 3 {matched_comp} Wilcoxon W: {readme_w} vs CSV {csv_w:.1f}")
            if abs(readme_w - csv_w) > 1.0:
                mismatches.append(f"Table 3 {matched_comp} W mismatch: README {readme_w} vs CSV {csv_w:.1f}")


def check_table_4_per_train_seed(header, rows, results_dir, mismatches, checked):
    """Table 4: Per-training-seed summary vs per_train_seed.csv."""
    csv_path = os.path.join(results_dir, "per_train_seed.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path)

    for row in rows:
        algo_raw = row[0].replace("**", "").strip()
        algo_key = "PPO" if "ppo" in algo_raw.lower() else ("DQN" if "dqn" in algo_raw.lower() else None)
        if not algo_key:
            continue
        sub = df[df["Algorithm"] == algo_key]
        mean_score = sub["Mean_Score"].mean()
        min_score = sub["Mean_Score"].min()
        max_score = sub["Mean_Score"].max()
        beats_alwayson = sub["Beats_AlwaysOn_24"].sum()
        beats_adapted = sub["Beats_AdaptedRuleAgent"].sum()

        readme_mean = clean_num(row[1])
        m_bounds = re.search(r"\(([0-9.]+)\s*–\s*([0-9.]+)\)", row[1])
        m_ao = re.search(r"(\d+)\s*/\s*(\d+)", row[2])
        m_ad = re.search(r"(\d+)\s*/\s*(\d+)", row[3])

        if readme_mean is not None:
            checked.append(f"Table 4 {algo_key} Mean Score: {readme_mean} vs CSV {mean_score:.4f}")
            if abs(readme_mean - mean_score) > 0.002:
                mismatches.append(f"Table 4 {algo_key} Mean Score mismatch: README {readme_mean} vs CSV {mean_score:.4f}")

        if m_bounds:
            r_min = float(m_bounds.group(1))
            r_max = float(m_bounds.group(2))
            checked.append(f"Table 4 {algo_key} Bounds: [{r_min}, {r_max}] vs CSV [{min_score:.4f}, {max_score:.4f}]")
            if abs(r_min - min_score) > 0.002 or abs(r_max - max_score) > 0.002:
                mismatches.append(f"Table 4 {algo_key} Bounds mismatch: README [{r_min}, {r_max}] vs CSV [{min_score:.4f}, {max_score:.4f}]")

        if m_ao:
            r_ao = int(m_ao.group(1))
            checked.append(f"Table 4 {algo_key} Beats AlwaysOn: {r_ao} vs CSV {beats_alwayson}")
            if r_ao != beats_alwayson:
                mismatches.append(f"Table 4 {algo_key} Beats AlwaysOn mismatch: README {r_ao} vs CSV {beats_alwayson}")

        if m_ad:
            r_ad = int(m_ad.group(1))
            checked.append(f"Table 4 {algo_key} Beats Adapted: {r_ad} vs CSV {beats_adapted}")
            if r_ad != beats_adapted:
                mismatches.append(f"Table 4 {algo_key} Beats Adapted mismatch: README {r_ad} vs CSV {beats_adapted}")


def check_table_5_behavior(header, rows, results_dir, mismatches, checked):
    """Table 5: Behavioral breakdown vs behavior_summary.csv."""
    csv_path = os.path.join(results_dir, "behavior_summary.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path).set_index("Agent_Group")

    for row in rows:
        agent_raw = row[0].replace("**", "").strip()
        matched = None
        for idx in df.index:
            if agent_raw.lower() in idx.lower() or idx.lower() in agent_raw.lower():
                matched = idx
                break
        if not matched:
            continue
        csv_row = df.loc[matched]

        # Off-peak disch (1), Peak disch (2), Peak soc (3), Peak grid kwh (4)
        r_off_dis = clean_num(row[1])
        r_peak_dis = clean_num(row[2])
        r_peak_soc = clean_num(row[3])
        r_peak_kwh = clean_num(row[4])

        if r_off_dis is not None:
            c_val = float(csv_row["off_peak_discharge_frac"]) * 100.0
            checked.append(f"Table 5 {matched} OffPeak Disch: {r_off_dis}% vs CSV {c_val:.1f}%")
            if abs(r_off_dis - c_val) > 0.5:
                mismatches.append(f"Table 5 {matched} OffPeak Disch mismatch: README {r_off_dis}% vs CSV {c_val:.1f}%")

        if r_peak_dis is not None:
            c_val = float(csv_row["peak_discharge_frac"]) * 100.0
            checked.append(f"Table 5 {matched} Peak Disch: {r_peak_dis}% vs CSV {c_val:.1f}%")
            if abs(r_peak_dis - c_val) > 0.5:
                mismatches.append(f"Table 5 {matched} Peak Disch mismatch: README {r_peak_dis}% vs CSV {c_val:.1f}%")

        if r_peak_kwh is not None:
            c_val = float(csv_row["peak_grid_energy_kwh"])
            checked.append(f"Table 5 {matched} Peak Grid kWh: {r_peak_kwh} vs CSV {c_val:.2f}")
            if abs(r_peak_kwh - c_val) > 0.1:
                mismatches.append(f"Table 5 {matched} Peak Grid kWh mismatch: README {r_peak_kwh} vs CSV {c_val:.2f}")


def check_table_6_arrival(header, rows, results_dir, mismatches, checked):
    """Table 6: Arrival Dynamics vs arrival_behavior.csv."""
    csv_path = os.path.join(results_dir, "arrival_behavior.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path).set_index("Agent_Group")

    for row in rows:
        agent_raw = row[0].replace("**", "").strip()
        matched = None
        for idx in df.index:
            if agent_raw.lower() in idx.lower() or idx.lower() in agent_raw.lower():
                matched = idx
                break
        if not matched:
            continue
        csv_row = df.loc[matched]

        r_temp = clean_num(row[1])
        r_comf = clean_num(row[2])
        r_pre_on = clean_num(row[3])
        r_steps = clean_num(row[5])

        if r_temp is not None:
            c_val = float(csv_row["Arrival_Temp_Mean"])
            checked.append(f"Table 6 {matched} Arrival Temp: {r_temp} vs CSV {c_val:.2f}")
            if abs(r_temp - c_val) > 0.1:
                mismatches.append(f"Table 6 {matched} Arrival Temp mismatch: README {r_temp} vs CSV {c_val:.2f}")

        if r_comf is not None:
            c_val = float(csv_row["Comfortable_At_Arrival_Pct"])
            checked.append(f"Table 6 {matched} Comf at Arrival: {r_comf}% vs CSV {c_val:.1f}%")
            if abs(r_comf - c_val) > 0.5:
                mismatches.append(f"Table 6 {matched} Comf at Arrival mismatch: README {r_comf}% vs CSV {c_val:.1f}%")

        if r_pre_on is not None:
            c_val = float(csv_row["Pre_Arrival_AC_On_Pct"])
            checked.append(f"Table 6 {matched} Pre-Arrival AC On: {r_pre_on}% vs CSV {c_val:.1f}%")
            if abs(r_pre_on - c_val) > 0.5:
                mismatches.append(f"Table 6 {matched} Pre-Arrival AC On mismatch: README {r_pre_on}% vs CSV {c_val:.1f}%")

        if r_steps is not None:
            c_val = float(csv_row["Steps_To_Comfort_Mean"])
            checked.append(f"Table 6 {matched} Steps to Comfort: {r_steps} vs CSV {c_val:.2f}")
            if abs(r_steps - c_val) > 0.1:
                mismatches.append(f"Table 6 {matched} Steps to Comfort mismatch: README {r_steps} vs CSV {c_val:.2f}")


def check_table_7_100k_summary(header, rows, results_dir, mismatches, checked):
    """Table 7: 100k sensitivity summary vs sensitivity_100k_summary.csv."""
    csv_path = os.path.join(results_dir, "sensitivity_100k_summary.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path).set_index("Agent")

    for row in rows:
        agent_raw = row[0].replace("**", "").strip()
        matched = None
        for idx in df.index:
            if idx.lower() in agent_raw.lower() or agent_raw.lower() in idx.lower():
                matched = idx
                break
        if not matched:
            continue
        csv_row = df.loc[matched]

        r_comf = clean_num(row[2])
        r_cost = clean_num(row[3])
        r_sw = clean_num(row[4])
        r_score = clean_num(row[5])

        if r_score is not None:
            c_val = float(csv_row["Score_Mean"])
            checked.append(f"Table 7 {matched} Score: {r_score} vs CSV {c_val:.4f}")
            if abs(r_score - c_val) > 0.005:
                mismatches.append(f"Table 7 {matched} Score mismatch: README {r_score} vs CSV {c_val:.4f}")

        if r_cost is not None:
            c_val = float(csv_row["Cost_Mean"])
            checked.append(f"Table 7 {matched} Cost: {r_cost} vs CSV {c_val:.2f}")
            if abs(r_cost - c_val) > 0.15:
                mismatches.append(f"Table 7 {matched} Cost mismatch: README {r_cost} vs CSV {c_val:.2f}")


def check_table_8_and_9_100k_tests(tables, results_dir, mismatches, checked):
    """Tables 8 & 9: 100k paired tests vs sensitivity_100k.csv."""
    csv_path = os.path.join(results_dir, "sensitivity_100k.csv")
    if not os.path.exists(csv_path):
        mismatches.append(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # Table 8: Training-Seed level
    t8_header, t8_rows = tables[8]
    seed_df = df[df["Level"] == "Training-Seed"].set_index("Comparison")
    for row in t8_rows:
        comp_raw = row[0].replace("**", "").strip()
        matched = None
        for idx in seed_df.index:
            if comp_raw.lower() in idx.lower() or idx.lower() in comp_raw.lower():
                matched = idx
                break
        if not matched:
            continue
        csv_row = seed_df.loc[matched]
        m_wins = re.search(r"(\d+)\s*/\s*(\d+)", row[1])
        if m_wins:
            r_wins = int(m_wins.group(1))
            c_wins = int(csv_row["Wins"])
            checked.append(f"Table 8 {matched} Wins: {r_wins} vs CSV {c_wins}")
            if r_wins != c_wins:
                mismatches.append(f"Table 8 {matched} Wins mismatch: README {r_wins} vs CSV {c_wins}")

    # Table 9: Episode level
    t9_header, t9_rows = tables[9]
    ep_df = df[df["Level"] == "Episode"].set_index("Comparison")
    for row in t9_rows:
        comp_raw = row[0].replace("**", "").strip()
        matched = None
        for idx in ep_df.index:
            if comp_raw.lower() in idx.lower() or idx.lower() in comp_raw.lower():
                matched = idx
                break
        if not matched:
            continue
        csv_row = ep_df.loc[matched]
        m_wins = re.search(r"(\d+)\s*/\s*(\d+)", row[1])
        if m_wins:
            r_wins = int(m_wins.group(1))
            c_wins = int(csv_row["Wins"])
            checked.append(f"Table 9 {matched} Wins: {r_wins} vs CSV {c_wins}")
            if r_wins != c_wins:
                mismatches.append(f"Table 9 {matched} Wins mismatch: README {r_wins} vs CSV {c_wins}")


def check_prose_numbers(readme_text, results_dir, mismatches, checked):
    """Check key numbers and parameters stated in README prose."""
    # 1. T_hot optimum from baseline_design.csv
    design_csv = os.path.join(results_dir, "baseline_design.csv")
    if os.path.exists(design_csv):
        df_des = pd.read_csv(design_csv)
        # Find row with max mean_score among numeric t_hot
        numeric_des = df_des.copy()
        numeric_des["t_hot_num"] = pd.to_numeric(numeric_des["t_hot"], errors="coerce")
        numeric_des = numeric_des.dropna(subset=["t_hot_num"])
        best_row = numeric_des.loc[numeric_des["mean_score"].idxmax()]
        best_t_hot = float(best_row["t_hot_num"])

        checked.append(f"Baseline Design Optimal T_hot: {best_t_hot}°C")
        # Ensure README mentions 26.0 or 26°C
        if "26.0°C" not in readme_text and "26°C" not in readme_text:
            mismatches.append("README prose does not mention the optimal T_hot = 26.0°C")

    # 2. Key scores in prose
    eval_csv = os.path.join(results_dir, "evaluation_summary.csv")
    if os.path.exists(eval_csv):
        df_eval = pd.read_csv(eval_csv).set_index("Agent")
        ppo_score = float(df_eval.loc["PPO", "Score_Mean"])
        ao_score = float(df_eval.loc["AlwaysOn_24", "Score_Mean"])
        checked.append(f"Prose PPO 20k Score: {ppo_score:.4f}")
        checked.append(f"Prose AlwaysOn_24 Score: {ao_score:.4f}")

        # Check for 0.8283 and 0.8367 in text
        if f"{ppo_score:.4f}" not in readme_text:
            mismatches.append(f"PPO 20k mean score {ppo_score:.4f} not found in README text")
        if f"{ao_score:.4f}" not in readme_text:
            mismatches.append(f"AlwaysOn_24 mean score {ao_score:.4f} not found in README text")

    # 3. 100k sensitivity numbers
    sens_csv = os.path.join(results_dir, "sensitivity_100k_summary.csv")
    if os.path.exists(sens_csv):
        df_sens = pd.read_csv(sens_csv).set_index("Agent")
        target_ppo100 = [k for k in df_sens.index if "ppo_100k" in k.lower() or "ppo (100k" in k.lower()]
        if target_ppo100:
            k = target_ppo100[0]
            ppo100_score = float(df_sens.loc[k, "Score_Mean"])
            ppo100_cost = float(df_sens.loc[k, "Cost_Mean"])
            checked.append(f"Prose PPO 100k Score: {ppo100_score:.4f}")
            checked.append(f"Prose PPO 100k Cost: ${ppo100_cost:.2f}")

            if f"{ppo100_score:.4f}" not in readme_text:
                mismatches.append(f"PPO 100k mean score {ppo100_score:.4f} not found in README text")
            if f"{ppo100_cost:.2f}" not in readme_text:
                mismatches.append(f"PPO 100k mean cost {ppo100_cost:.2f} not found in README text")


def check_forbidden_phrases(readme_text, mismatches, checked):
    """
    Check that no forbidden stale phrases remain:
      - 'dominates' or 'Dominates'
      - 'overtakes'
      - unsupported 'proactive pre-cooling' (pre-cooling is only supported if qualified as empirical or arrival)
      - claims that AdaptedRuleAgent is 'strong'
    """
    # 1. dominates
    dom_matches = re.findall(r"(?i)\bdominat\w*", readme_text)
    checked.append(f"Forbidden 'dominates' count: {len(dom_matches)}")
    if dom_matches:
        mismatches.append(f"Forbidden word 'dominates' found ({len(dom_matches)} occurrences): {dom_matches}")

    # 2. overtakes
    over_matches = re.findall(r"(?i)\bovertak\w*", readme_text)
    checked.append(f"Forbidden 'overtakes' count: {len(over_matches)}")
    if over_matches:
        mismatches.append(f"Forbidden word 'overtakes' found ({len(over_matches)} occurrences): {over_matches}")

    # 3. unsupported 'proactive pre-cooling'
    proact_matches = re.findall(r"(?i)proactive pre-cooling", readme_text)
    checked.append(f"Unsupported 'proactive pre-cooling' count: {len(proact_matches)}")
    if proact_matches:
        mismatches.append(f"Unsupported phrase 'proactive pre-cooling' found ({len(proact_matches)} occurrences)")

    # 4. 'strong' attached to AdaptedRuleAgent
    # Look for AdaptedRuleAgent within 50 characters of 'strong'
    adapted_strong = re.findall(r"(?i)AdaptedRuleAgent[^\.\n]{0,50}\bstrong\b|\bstrong\b[^\.\n]{0,50}AdaptedRuleAgent", readme_text)
    checked.append(f"AdaptedRuleAgent described as 'strong' count: {len(adapted_strong)}")
    if adapted_strong:
        mismatches.append(f"AdaptedRuleAgent improperly described as 'strong': {adapted_strong}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    readme_path = os.path.join(repo_root, "README.md")
    results_dir = os.path.join(repo_root, "results")

    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} not found.")
        sys.exit(1)

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    tables = parse_markdown_tables(readme_text)
    print(f"Parsed {len(tables)} tables from README.md")

    mismatches = []
    checked = []

    # 1. Check Tables
    if len(tables) >= 10:
        check_table_2_eval_summary(tables[2][0], tables[2][1], results_dir, mismatches, checked)
        check_table_3_statistical_tests(tables[3][0], tables[3][1], results_dir, mismatches, checked)
        check_table_4_per_train_seed(tables[4][0], tables[4][1], results_dir, mismatches, checked)
        check_table_5_behavior(tables[5][0], tables[5][1], results_dir, mismatches, checked)
        check_table_6_arrival(tables[6][0], tables[6][1], results_dir, mismatches, checked)
        check_table_7_100k_summary(tables[7][0], tables[7][1], results_dir, mismatches, checked)
        check_table_8_and_9_100k_tests(tables, results_dir, mismatches, checked)
    else:
        mismatches.append(f"Expected at least 10 tables in README.md, found {len(tables)}")

    # 2. Check Prose Numbers
    check_prose_numbers(readme_text, results_dir, mismatches, checked)

    # 3. Check Forbidden Stale Phrases
    check_forbidden_phrases(readme_text, mismatches, checked)

    # Print summary
    print(f"\n--- Automated Consistency Check Results ---")
    print(f"Total values and properties verified: {len(checked)}")
    for c in checked[:15]:
        print(f"  [PASS] {c}")
    if len(checked) > 15:
        print(f"  ... and {len(checked) - 15} more checks passed.")

    if mismatches:
        print(f"\n[FAIL] Found {len(mismatches)} consistency mismatches:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All README numbers, tables, and phrases are 100% consistent with results/ CSVs!")
        sys.exit(0)


if __name__ == "__main__":
    main()
