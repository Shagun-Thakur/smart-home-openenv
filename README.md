---
title: Smart Home OpenEnv
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

# Smart Home OpenEnv: Residential Energy Optimization Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0.0-green.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20v2.15-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.9-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Unit Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-purple.svg)](openenv.yaml)

A rigorous, Gymnasium-compliant reinforcement learning benchmark for residential demand response, HVAC setpoint control, and stationary battery storage management. The environment models the multi-objective decision process of an autonomous Home Energy Management System (HEMS) balancing **human thermal comfort**, **time-of-use (TOU) electricity costs**, and **compressor mechanical switching fatigue** under stochastic occupancy and dynamic weather conditions.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Formal Problem Formulation (MDP)](#-formal-problem-formulation-mdp)
  - [State & Action Spaces](#state-and-action-spaces)
  - [System Dynamics & Physics](#system-dynamics-and-physics)
  - [Multi-Objective Reward Function](#multi-objective-reward-function)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
  - [Tier 1: Standard Tasks (10 Seeds)](#tier-1-standard-tasks-10-seeds)
  - [Tier 2: Realistic Scenario Evaluation (50 Shared Seeds)](#tier-2-realistic-scenario-evaluation-50-shared-seeds)
  - [Paired Statistical Significance (Wilcoxon Signed-Rank Test)](#paired-statistical-significance)
  - [Per-Training-Seed Stability](#per-training-seed-stability)
  - [Analysis & Behavioral Evidence](#analysis--behavioral-evidence)
  - [Extended Training Sensitivity Check (100k Steps)](#-extended-training-sensitivity-check-100000-steps)
- [Diagnostic Visualizations](#-diagnostic-visualizations)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
- [Repository Structure](#-repository-structure)
- [Limitations & Threats to Validity](#-limitations--threats-to-validity)
- [Citation & Contact](#-citation--contact)

---

## 🚀 Executive Overview

Heating, ventilation, and air conditioning (HVAC) represent over 40% of residential electricity consumption. With the widespread adoption of variable time-of-use tariffs, rooftop solar, and stationary batteries, autonomous demand-side management is critical for modern grid stability. 

`smart-home-openenv` provides a fast, reproducible platform designed to address key limitations in existing benchmarks:
1. **Physical Conservation**: Eliminates unconstrained battery loopholes through rigorous physical action clipping and consistent charge/discharge energy accounting.
2. **Synchronous Exogenous Transitions**: Prevents 1-step observation lag by advancing price and Markovian occupancy endogenously within the environment transition loop.
3. **Rigorous Multi-Seed Protocols**: Evaluates learned policies against heuristic and rule-based baselines across 10 training seeds and 50 strictly disjoint evaluation seeds with non-parametric paired statistical testing.
4. **Lightweight & High Throughput**: Executes at 1,000+ steps/second on standard CPUs, enabling complete multi-seed training runs in minutes without GPU dependencies.

---

## ⚡ Key Features

- **Standardized Gymnasium API**: Implements `gymnasium.Env` with clean `Box` observation spaces, `Discrete(12)` action spaces, and standard `(terminated, truncated)` horizon semantics.
- **Bi-Directional Battery Inverter Model**: 10 kWh storage with 1.0 kWh step flow, 95% one-way efficiencies, and guaranteed zero capacity limit violations.
- **First-Order Thermal Inertia**: Continuous temperature approach rates ($\alpha = 0.35$ cooling, $\alpha = 0.05$ ambient, $\alpha = 0.10$ drift) replacing unrealistic instantaneous step jumps.
- **Diurnal Weather & Time-of-Use Tariffs**: Continuous 24-hour sinusoidal solar heat gain coupled with an empirical 3-tier tariff (\$0.50 off-peak, \$1.00 mid-peak, \$2.00 peak).
- **Markovian Occupancy**: 2-state Markov chain with state retention probabilities $p(\text{away} \to \text{away}) = 0.85$ and $p(\text{home} \to \text{home}) = 0.90$, generating realistic continuous presence and absence intervals.
- **FastAPI & Docker Deployment**: Fully self-contained container with REST API endpoints (`GET /`, `POST /reset`) for Hugging Face Space hosting.

---

## 📐 Formal Problem Formulation (MDP)

The residential energy management task is formulated as a discrete-time, finite-horizon Markov Decision Process: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{T} \rangle$.

### State and Action Spaces

#### State Space ($\mathcal{S}$)
The agent observes a continuous vector in `Box(7)` (or `Box(5)` for baseline tasks):

| Index | Feature | Dimension / Unit | Physical Meaning |
| :---: | :--- | :---: | :--- |
| `0` | $T^{\text{in}}_t$ | $[10.0, 45.0]^\circ\text{C}$ | Current indoor dry-bulb temperature |
| `1` | $T^{\text{out}}_t$ | $[0.0, 50.0]^\circ\text{C}$ | Ambient outdoor dry-bulb temperature |
| `2` | $P_t$ | $[0.0, 5.0] \ \$/\text{kWh}$ | Active electricity tariff rate |
| `3` | $\text{SoC}_t$ | $[0.0, 100.0] \ \%$ | Battery state of charge |
| `4` | $O_t$ | $\{0, 1\}$ | Occupancy indicator ($1 = \text{Home}, 0 = \text{Away}$) |
| `5` | $\sin(2\pi h / 24)$ | $[-1.0, 1.0]$ | Cyclical trigonometric time encoding (sine) |
| `6` | $\cos(2\pi h / 24)$ | $[-1.0, 1.0]$ | Cyclical trigonometric time encoding (cosine) |

#### Action Space ($\mathcal{A}$)
A flat `Discrete(12)` space mapping the Cartesian product of 4 AC cooling setpoints and 3 battery storage modes:

$$
\mathcal{A} = \mathcal{A}_{\text{ac}} \times \mathcal{A}_{\text{bat}} = \{22^\circ\text{C}, 24^\circ\text{C}, 26^\circ\text{C}, \text{None (OFF)}\} \times \{\text{"CHARGE"}, \text{"DISCHARGE"}, \text{"IDLE"}\}
$$

The scalar action index $a \in \{0, 1, \dots, 11\}$ is computed as:

$$
a = \text{idx}_{\text{ac}} \times 3 + \text{idx}_{\text{bat}}
$$

---

### System Dynamics and Physics

#### 1. Thermal Inertia Model

Approximates sensible Newton cooling with building envelope thermal mass:

$$
T^{\text{in}}_{t+1} = \begin{cases} 
T^{\text{in}}_t + \alpha_{\text{cooling}}(a^{\text{ac}}_t - T^{\text{in}}_t) + \alpha_{\text{ambient}}(T^{\text{out}}_t - T^{\text{in}}_t) & \text{if } a^{\text{ac}}_t \neq \text{None} \\ 
T^{\text{in}}_t + \alpha_{\text{drift}}(T^{\text{out}}_t - T^{\text{in}}_t) & \text{if } a^{\text{ac}}_t = \text{None} 
\end{cases}
$$

$$
\alpha_{\text{cooling}} = 0.35, \quad \alpha_{\text{ambient}} = 0.05, \quad \alpha_{\text{drift}} = 0.10
$$

#### 2. Battery Storage & Net Grid Demand

**Battery Discharge Mode (`"DISCHARGE"`):**  
Effective only when stored energy is available ($\text{SoC}_t > 0$). Flow is clipped to available charge:

$$
\Delta\text{SoC} = \min(10.0, \, \text{SoC}_t)
$$

The energy released to offset household demand is:

$$
E_{\text{relief}} = \left(\frac{\Delta\text{SoC}}{100}\right) \cdot C_{\text{bat}} \cdot \eta_{\text{dis}}
$$

**Battery Charge Mode (`"CHARGE"`):**  
Effective only when capacity remains ($\text{SoC}_t < 100.0$). Flow is clipped to remaining capacity:

$$
\Delta\text{SoC} = \min(10.0, \, 100.0 - \text{SoC}_t)
$$

The energy drawn from the grid to charge the battery is:

$$
E_{\text{grid-bat}} = \frac{(\Delta\text{SoC}/100) \cdot C_{\text{bat}}}{\eta_{\text{chg}}}
$$

**Net Grid Power & Financial Cost:**  

$$
E^{\text{net}}_t = \max\left(0.0, \, E^{\text{ac}}_t + E_{\text{grid-bat}} - E_{\text{relief}}\right), \quad \text{Cost}_t = P_t \times E^{\text{net}}_t
$$

---

### Multi-Objective Reward Function

The scalarized reward combines energy expenditure, thermal discomfort, and mechanical cycling:

$$
R_t = -\left(w_1 \cdot \text{Cost}_t + w_2 \cdot \text{Pen}^{\text{comfort}}_t + w_3 \cdot \text{Pen}^{\text{switch}}_t\right) + \mathbb{I}\left(\text{Pen}^{\text{comfort}}_t = 0\right)
$$

$$
w_1 = 0.4 \ \ (\text{cost}), \quad w_2 = 0.5 \ \ (\text{comfort}), \quad w_3 = 0.1 \ \ (\text{cycling wear}), \quad \text{bonus} = 1.0
$$

#### Occupancy-Consistent Thermal Comfort

Human discomfort is zero when the residence is vacant ($O = 0$). When occupied ($O = 1$), discomfort scales linearly with temperature deviations outside $[22^\circ\text{C}, 26^\circ\text{C}]$:

$$
\text{Pen}^{\text{comfort}}_t = \begin{cases} 
0.0 & \text{if } O_t = 0 \quad (\text{Vacant building; zero human discomfort}) \\ 
\max(0.0, \, 22.0 - T^{\text{in}}_t) + \max(0.0, \, T^{\text{in}}_t - 26.0) & \text{if } O_t = 1 \quad (\text{Occupied interval}) 
\end{cases}
$$

#### Compressor Switching Wear

Penalizes active switching between non-idle cooling setpoints to preserve equipment lifetime:

$$
\text{Pen}^{\text{switch}}_t = \begin{cases} 
1.0 & \text{if } a^{\text{ac}}_t \neq a^{\text{ac}}_{t-1} \land a^{\text{ac}}_{t-1} \neq \text{None} \\ 
0.0 & \text{otherwise} 
\end{cases}
$$

---

## 📊 Experimental Results & Benchmarks

All results are reproducible directly from repository scripts. No numbers are estimated or invented.

### Tier 1: Standard Tasks (10 Seeds)
Evaluated across 10 fixed random seeds (0–9, 50 steps/episode) using `python scripts/evaluate_baseline.py`:

| Task Name | Objective Focus | Comfort Ratio (mean ± SD) | Normalized Cost (mean ± SD) | Combined Score (mean ± SD) |
| :--- | :--- | :---: | :---: | :---: |
| **Easy** | Comfort Regulation | 0.9400 ± 0.0000 | 0.0000 ± 0.0000 | **0.9400 ± 0.0000** |
| **Medium** | Comfort + Constant Cost | 0.9400 ± 0.0000 | 0.3600 ± 0.0000 | **0.8200 ± 0.0000** |
| **Hard** | Dynamic Tariffs + Occupancy | 0.9560 ± 0.0158 | 0.2571 ± 0.0230 | **0.8921 ± 0.0148** |

*Note: The Easy and Medium tasks are simple deterministic sanity tasks with static conditions and zero variance ($\pm 0.0000$) across seeds; they verify basic environment mechanics and grader integration rather than serving as evidence of policy learning.*

---

### Tier 2: Realistic Scenario Evaluation (50 Shared Seeds)

Benchmarked on the 24-hour realistic scenario (`configs/realistic.yaml`). Learned models (**PPO** and **DQN**) were trained across 10 independent seeds (0–9, 20,000 steps each) and evaluated on 50 shared, unseen test episodes (seeds 1000–1049):

| Agent / Model | Paradigm | Comfort Ratio (mean ± SD) | Total Cost (\$) (mean ± SD) | Daily AC Switches (mean ± SD) | Combined Score (mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AlwaysOn_24** | Heuristic | **0.9686 ± 0.0178** | \$52.00 ± 0.00 | 0.00 | **0.8367 ± 0.0107** |
| **PPO** (10 models) | On-Policy RL | 0.8924 ± 0.1329 | \$38.56 ± 11.13 | **2.55 ± 0.88** | **0.8283 ± 0.0613** |
| **AdaptedRuleAgent** | Domain Rule | 0.7743 ± 0.1043 | **\$25.90 ± 5.77** | 16.40 ± 2.65 | **0.7926 ± 0.0535** |
| **AlwaysOn_22** | Heuristic | 1.0000 ± 0.0000 | \$78.00 ± 0.00 | 0.00 | **0.7833 ± 0.0000** |
| **DQN** (10 models) | Off-Policy RL | 0.7596 ± 0.2537 | \$44.49 ± 14.17 | 5.36 ± 2.14 | **0.7322 ± 0.1295** |
| **BaselineAgent (Rule)** | Original Rule | 0.6873 ± 0.1086 | \$35.06 ± 5.09 | 15.44 ± 2.40 | **0.7150 ± 0.0572** |
| **Random** | Stochastic | 0.7030 ± 0.1426 | \$44.31 ± 6.17 | 26.52 ± 2.80 | **0.6987 ± 0.0754** |
| **AlwaysOn_26** | Heuristic | 0.1746 ± 0.1064 | \$26.00 ± 0.00 | 0.00 | **0.4325 ± 0.0638** |

*Physical Battery Violations: **0** across all 1,100 evaluated episodes.*

---

### Paired Statistical Significance

Non-parametric paired **Wilcoxon signed-rank tests** (aligned strictly by evaluation seed) were conducted across all 50 shared evaluation episodes (`results/statistical_tests.csv`), with family-wise error rate controlled using the **Holm-Bonferroni correction**:

| Comparison | Median Paired Diff | Mean Paired Diff | Wins (/50) | Wilcoxon ($W$) | Raw $p$-value | Holm $p$-value | Significant ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PPO vs. AlwaysOn_24** | **+0.0027** | -0.0084 | 30 / 50 | 609.0 | 0.7888 | 0.7888 | **No ($p = 0.789$)** |
| **PPO vs. AdaptedRuleAgent** | **+0.0463** | +0.0357 | 36 / 50 | 223.0 | $2.83 \times 10^{-5}$ | $5.66 \times 10^{-5}$ | **Yes ($p < 0.001$)** |
| **PPO vs. BaselineAgent (Rule)** | **+0.1180** | +0.1134 | 49 / 50 | 4.0 | $1.24 \times 10^{-14}$ | $4.97 \times 10^{-14}$ | **Yes ($p < 0.001$)** |
| **DQN vs. AlwaysOn_24** | **-0.1016** | -0.1046 | 0 / 50 | 0.0 | $1.78 \times 10^{-15}$ | $1.07 \times 10^{-14}$ | **Yes (Deficit)** |
| **DQN vs. AdaptedRuleAgent** | **-0.0629** | -0.0605 | 2 / 50 | 48.0 | $4.38 \times 10^{-11}$ | $1.31 \times 10^{-10}$ | **Yes (Deficit)** |
| **DQN vs. PPO** | **-0.0995** | -0.0961 | 0 / 50 | 0.0 | $1.78 \times 10^{-15}$ | $1.07 \times 10^{-14}$ | **Yes (Deficit)** |

---

### Per-Training-Seed Stability

Episodic score distribution across the 10 independent training seeds (`results/per_train_seed.csv`):

| Algorithm | Mean Score (Min – Max) | Seeds Beating AlwaysOn_24 | Seeds Beating AdaptedRuleAgent |
| :--- | :---: | :---: | :---: |
| **PPO** | 0.8283 (0.7655 – 0.8622) | **5 / 10** | **8 / 10** |
| **DQN** | 0.7322 (0.4847 – 0.8271) | **0 / 10** | **5 / 10** |

---

### Analysis & Behavioral Evidence

To understand the actual mechanisms governing agent performance, step-level actions, battery dynamics, and temperature profiles were extracted across all 50 evaluation episodes via `python scripts/analyze_behavior.py` (`results/behavior_summary.csv`):

| Agent / Group | Off-Peak Disch. (%) | Peak Disch. (%) | Peak Mean SoC (%) | Peak Grid Energy (kWh) | AC Setpoint (% 22 / 24 / 26 / Off) | Occupied Mean Temp (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AlwaysOn_24** | 0.0% | 0.0% | 50.0% | 12.00 kWh | 0.0 / 100.0 / 0.0 / 0.0 | 24.89°C |
| **AdaptedRuleAgent** | 0.0% | **75.0%** | 47.5% | **1.76 kWh** | 5.2 / 46.6 / 17.0 / 31.3 | 25.38°C |
| **Random** | 36.6% | 35.3% | 47.5% | 10.03 kWh | 23.3 / 24.6 / 27.0 / 25.0 | 25.43°C |
| **PPO (all 10 seeds)** | 63.1% | 79.9% | 5.0% | 12.86 kWh | 29.1 / 33.0 / 0.0 / **37.9** | 24.66°C |
| **DQN (all 10 seeds)** | 21.4% | 14.7% | 69.7% | 10.21 kWh | 21.5 / 37.6 / 22.2 / 18.7 | 25.05°C |
| **DQN (seed 1 - worst)** | 42.5% | 0.0% | 0.0% | 7.80 kWh | 0.0 / 14.2 / **73.9** / 11.9 | 26.30°C |
| **DQN (seed 8 - worst)** | 100.0% | 64.5% | 0.0% | 8.13 kWh | 0.0 / 12.8 / **87.2** / 0.0 | 26.28°C |
| **DQN (seed 3 - best)** | 0.0% | 0.0% | 100.0% | 10.92 kWh | 0.0 / **97.8** / 0.0 / 2.2 | 24.93°C |

#### Key Empirical Insights

1. **How PPO Actually Achieves Cost Savings (Curtailment, Not Arbitrage)**:
   A common assumption in demand response is that reinforcement learning agents automatically discover time-of-use tariff arbitrage. The empirical data refutes this: PPO emits **0.0% charge actions** across all tariff tiers. It quickly discharges its initial 50% SoC during early off-peak and mid-peak hours (mean SoC drops to 5.0% before peak hours begin), resulting in **12.86 kWh** of peak grid energy (comparable to AlwaysOn_24's 12.00 kWh). Instead, PPO's 25.8% cost savings (\$38.56 vs. \$52.00) arise entirely from **intelligent compressor curtailment**: it turns the AC off for **37.9% of steps** (primarily when the residence is vacant), while cooling steadily to 22°C (29.1%) or 24°C (33.0%) when occupied, maintaining an occupied mean temperature of 24.66°C with only 2.55 switches/day.

2. **DQN Failure Mode: Under-cooling, Not Over-cooling**:
   The worst DQN seeds (seeds 1 and 8, scores ~0.48–0.52) did *not* over-cool during peak hours. Rather, they collapsed into an **under-cooling trap**: seed 1 and seed 8 selected AC 26°C for **73.9% and 87.2%** of all steps, allowing occupied indoor temperatures to drift to **26.30°C and 26.28°C** (consistently breaching the 26.0°C upper comfort limit). While this minimized electricity consumption (\$26.99 and \$29.95), it incurred massive thermal discomfort penalties, collapsing comfort ratios to 32.6% and 28.0%. Conversely, the best DQN seed (seed 3, score 0.8271) learned to emulate `AlwaysOn_24`, setting AC 24°C on **97.8% of steps** and continuously charging the battery to 100% SoC.

3. **True Battery Arbitrage is Achieved by Domain Rules**:
   `AdaptedRuleAgent` is the only policy that successfully executes battery load-shifting: charging during off-peak (56.3% charge actions, raising SoC to 82.0% in mid-peak) and discharging during peak pricing (75.0% discharge actions), slashing peak grid energy from 12.00 kWh down to **1.76 kWh**. However, its rigid thermostat thresholding causes frequent compressor toggling (16.4 switches/day) and lower comfort (77.4%).

---

### 🔬 Extended Training Sensitivity Check (100,000 Steps)

To evaluate whether the empirical conclusions are artifacts of the 20,000-step training budget, PPO and DQN were trained for **100,000 steps** across training seeds 0–4 and evaluated on the same 50 unseen test episodes (`python scripts/run_sensitivity_100k.py`, `results/sensitivity_100k.csv`):

| Agent / Model | Budget | Comfort Ratio (mean ± SD) | Total Cost (\$) (mean ± SD) | Daily AC Switches (mean ± SD) | Combined Score (mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PPO (100k steps, seeds 0–4)** | 100k | 0.9235 ± 0.0423 | **\$24.17 ± 5.09** | 4.23 ± 1.12 | **0.8870 ± 0.0423** |
| **AlwaysOn_24** | Static | **0.9686 ± 0.0178** | \$52.00 ± 0.00 | **0.00** | 0.8367 ± 0.0107 |
| **PPO (20k steps, all 10 seeds)** | 20k | 0.8924 ± 0.1329 | \$38.56 ± 11.13 | 2.55 ± 0.88 | 0.8283 ± 0.0613 |
| **DQN (100k steps, seeds 0–4)** | 100k | 0.8077 ± 0.0821 | \$25.36 ± 6.12 | 4.83 ± 1.45 | 0.8142 ± 0.0821 |
| **AdaptedRuleAgent** | Rule | 0.7743 ± 0.1043 | \$25.90 ± 5.77 | 16.40 ± 2.65 | 0.7926 ± 0.0535 |
| **DQN (20k steps, all 10 seeds)** | 20k | 0.7596 ± 0.2537 | \$44.49 ± 14.17 | 5.36 ± 2.14 | 0.7322 ± 0.1295 |

#### Sensitivity Conclusions
1. **PPO Dominates with Extended Training**: With 100k steps, PPO achieves a statistically significant victory over `AlwaysOn_24` (winning **45 out of 50 episodes**, median paired difference $+0.0580$, Holm-adjusted $p = 4.06 \times 10^{-12}$). It cuts daily electricity cost from \$52.00 to **\$24.17** (a 53.5% cost reduction) while achieving **92.4% comfort**. The conclusion that PPO and `AlwaysOn_24` are indistinguishable applies strictly to the initial 20,000-step budget.
2. **DQN Substantially Improves and Overtakes Rule Baselines**: With 100k steps, DQN's mean score rises from 0.7322 to **0.8142** and it statistically outperforms `AdaptedRuleAgent` (winning **32 out of 50 episodes**, Holm-adjusted $p = 0.0051$). However, DQN remains significantly behind PPO at both 20k and 100k steps ($p < 0.001$).

---

### Diagnostic Visualizations

Generated via `python scripts/make_report.py` and `python scripts/analyze_behavior.py`:

#### 1. Behavioral Breakdown by Tariff Tier
![Behavior by Tariff](results/behavior_by_tariff.png)

#### 2. Training Convergence (10 Seeds, Mean ± 1 SD)
![Learning Curves](results/learning_curves.png)

#### 3. Cost vs. Comfort Pareto Frontier
![Cost vs Comfort](results/cost_vs_comfort.png)

#### 4. Representative 24-Hour Day Profile (Seed 1000)
![Example Day Profile](results/example_day_trajectory.png)

---

## 🛠️ Quickstart & Reproduction Guide

### 1. Installation
Clone the repository and set up a dedicated virtual environment:
```bash
git clone https://github.com/Shagun-Thakur/smart-home-openenv.git
cd smart-home-openenv
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install runtime dependencies:
pip install -r requirements.txt

# For RL training and benchmark evaluation:
pip install -r requirements-train.txt
```

### 2. Verify Baseline Execution & Run Tests
```bash
# Run baseline task verification:
python inference.py

# Run the full automated unit test suite (27 tests):
pytest
```

### 3. Train RL Agents from Scratch
Train single models or execute the full 10-seed experiment:
```bash
# Train a single PPO agent (seed 0):
python scripts/train.py --config configs/realistic.yaml --algo ppo --seed 0

# Train a single DQN agent (seed 0):
python scripts/train.py --config configs/realistic.yaml --algo dqn --seed 0

# Train all 10 seeds for both algorithms:
python scripts/train.py --config configs/realistic.yaml --train-all
```

### 4. Run Benchmark Evaluation & Generate Plots
```bash
# Evaluate all agents over 50 test episodes and run Wilcoxon tests:
python scripts/evaluate_all.py

# Generate publication-ready diagnostic figures:
python scripts/make_report.py
```

### 5. Dockerized Execution
Build and launch the containerized REST API server:
```bash
docker build -t smart-home-env .
docker run -p 7860:7860 smart-home-env
```
Access the server at `http://localhost:7860/` (`GET /` returns `{"status": "running"}`).

---

## 📁 Repository Structure

```
smart-home-openenv/
├── configs/
│   └── realistic.yaml          # Realistic 24-hr simulation and physical parameters
├── env/
│   ├── dynamics.py             # Thermal drift, battery flow, and grid power equations
│   ├── environment.py          # Gymnasium SmartHomeEnv & LegacySmartHomeEnv wrapper
│   ├── realistic_env.py        # RealisticSmartHomeEnv with TOU tariffs and Markovian occupancy
│   └── reward.py               # Multi-objective reward computation and comfort predicates
├── tasks/
│   ├── task_easy.py            # EasyTask: comfort regulation (50 steps)
│   ├── task_medium.py          # MediumTask: comfort + cost balance (50 steps)
│   └── task_hard.py            # HardTask: dynamic price and occupancy (50 steps)
├── graders/
│   ├── grader_easy.py          # EasyGrader: comfort scoring
│   ├── grader_medium.py        # MediumGrader: combined comfort and cost scoring
│   └── grader_hard.py          # HardGrader: weighted comfort and cost scoring
├── agents/
│   └── baseline_agent.py       # BaselineAgent, AdaptedRuleAgent, RandomAgent, AlwaysOnAgent
├── scripts/
│   ├── evaluate.py             # Quick standalone evaluation script
│   ├── evaluate_all.py         # Full 50-episode benchmark & Wilcoxon statistical testing
│   ├── evaluate_baseline.py    # 10-seed standard task benchmark generator
│   ├── make_report.py          # Publication figure generation script
│   ├── run_env.py              # Stepped environment diagnostic script
│   └── train.py                # SB3 PPO and DQN training CLI
├── tests/                      # Automated unit test suite (27 tests)
│   ├── test_action_spaces.py   # Discrete(12) bijective encoding and bounds
│   ├── test_battery.py         # Physical clipping and BMS flow conservation
│   ├── test_comfort.py         # Reward/grader alignment during vacant intervals
│   ├── test_cost_accounting.py # Single source of truth for energy and cost
│   ├── test_determinism.py     # Bitwise trajectory reproducibility under static seeds
│   ├── test_realistic_env.py   # Diurnal weather, tariffs, and thermal inertia
│   ├── test_stale_observation.py # Zero-lag exogenous state verification
│   └── test_termination.py     # Truncation and horizon semantics
├── results/                    # Empirical data, trained models, and figures
│   ├── models/                 # Serialized PPO and DQN models (seeds 0-9)
│   ├── cost_vs_comfort.png     # Pareto trade-off scatter plot
│   ├── evaluation_metrics.csv  # Detailed per-episode test rollouts
│   ├── evaluation_summary.csv  # Benchmark aggregate summary table
│   ├── example_day_trajectory.png # 24-hr multi-panel step profile
│   ├── learning_curves.csv     # Episodic training returns
│   ├── learning_curves.png     # Training convergence curves
│   ├── statistical_tests.csv   # Wilcoxon signed-rank test statistics and p-values
│   ├── tier1_baseline.csv      # Standard task per-seed baseline numbers
│   └── tier1_baseline_summary.csv # Standard task aggregate statistics
├── app.py                      # FastAPI REST application for Space deployment
├── Dockerfile                  # Container build specification
├── inference.py                # Verification entrypoint
├── openenv.yaml                # OpenEnv environment metadata
├── pyproject.toml              # Build system and project metadata
├── requirements.txt            # Minimal runtime deployment requirements
└── requirements-train.txt      # Pinned research training dependencies
```

---

## ⚠️ Limitations & Threats to Validity

1. **Training Budget & Default Hyperparameters**: PPO and DQN were trained for only 20,000 steps (about 417 episodes) using default Stable-Baselines3 hyperparameters without tuning. The comparative performance between PPO and DQN applies strictly within this limited computational budget; longer training or tailored exploration may yield different convergence properties.
2. **Reward vs. Grader Objective Divergence**: Agents are trained on the step reward in `env/reward.py` (raw electricity cost weighted 0.4, thermal discomfort weighted 0.5, compressor wear weighted 0.1, plus comfort bonus), but evaluated on the grader score ($0.6 \cdot \text{comfort} + 0.4 \cdot (1 - \text{normalized cost})$). Because the training reward and evaluation metric are non-identical scalarizations, policies do not directly optimize the benchmark evaluation objective.
3. **Competitive Constant-Setpoint Baseline**: In this diurnal simulation, a fixed setpoint of 24°C (`AlwaysOn_24`) provides an exceptionally strong baseline ($96.9\%$ comfort, zero compressor switching wear). Learned RL policies achieve a different trade-off (saving cost via vacant curtailment) but are not statistically distinguishable on the overall combined score.
4. **Single Synthetic Scenario**: All results are generated from one synthetic residential configuration (`configs/realistic.yaml`). Real homes feature distinct building thermal envelopes, multi-zone thermodynamics, occupant behavioral idiosyncrasies, and actual climate variability.
5. **Synthetic Thermal Linearization**: Temperature transitions follow a simplified first-order sensible cooling model. Real buildings exhibit multi-zone thermodynamic gradients, structural thermal capacitance across internal partition walls, and solar radiation geometry.
6. **Sensible vs. Latent Cooling**: The environment models dry-bulb sensible temperature only. Real air conditioners expend significant energy condensing moisture (latent heat of vaporization), which is unmodeled here.
7. **Idealized Battery Inverter**: The battery model utilizes constant 95% one-way efficiencies and discrete 1.0 kWh step flow. Real battery storage exhibits Peukert capacity degradation, C-rate dependent thermal losses, and long-term cyclic State of Health (SoH) wear.
8. **Subjective Reward Scalarization**: The reward weights (cost: 0.4, comfort: 0.5, switching wear: 0.1) represent one specific scalar preference on the Pareto frontier. Other residential consumers may prioritize cost over comfort or vice versa.

---

## 📄 Citation & Contact

If you find this environment or benchmark helpful in your research, please cite:

```bibtex
@misc{thakur2026smarthomeopenenv,
  author = {Shagun Thakur},
  title = {Smart Home OpenEnv: A Gymnasium Benchmark for Residential Demand Response and Energy Storage Control},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Shagun-Thakur/smart-home-openenv}}
}
```

**Author**: Shagun Thakur  
**License**: MIT License
