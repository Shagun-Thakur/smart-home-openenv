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
  - [Analysis & Scientific Honesty on Negative Results](#scientific-honesty--analysis-of-negative-results)
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
- **First-Order Thermal Inertia**: Continuous temperature approach rates ($\alpha_{\text{cooling}}=0.35, \alpha_{\text{ambient}}=0.05, \alpha_{\text{drift}}=0.10$) replacing unrealistic instantaneous step jumps.
- **Diurnal Weather & Time-of-Use Tariffs**: Continuous 24-hour sinusoidal solar heat gain coupled with an empirical 3-tier tariff (\$0.50 off-peak, \$1.00 mid-peak, \$2.00 peak).
- **Markovian Occupancy**: 2-state Markov chain ($p_{\text{stay-away}}=0.85, \, p_{\text{stay-home}}=0.90$) generating realistic continuous presence and absence intervals.
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

$$\mathcal{A} = \mathcal{A}_{\text{ac}} \times \mathcal{A}_{\text{bat}} = \{22^\circ\text{C}, 24^\circ\text{C}, 26^\circ\text{C}, \text{None (OFF)}\} \times \{\text{"CHARGE"}, \text{"DISCHARGE"}, \text{"IDLE"}\}$$

$$a = \text{idx}_{\text{ac}} \times 3 + \text{idx}_{\text{bat}}, \quad a \in \{0, 1, \dots, 11\}$$

---

### System Dynamics and Physics

#### 1. Thermal Inertia Model
Approximates sensible Newton cooling with building envelope thermal mass:
$$T^{\text{in}}_{t+1} = \begin{cases} T^{\text{in}}_t + \alpha_{\text{cooling}}(a^{\text{ac}}_t - T^{\text{in}}_t) + \alpha_{\text{ambient}}(T^{\text{out}}_t - T^{\text{in}}_t) & \text{if } a^{\text{ac}}_t \neq \text{None} \\ T^{\text{in}}_t + \alpha_{\text{drift}}(T^{\text{out}}_t - T^{\text{in}}_t) & \text{if } a^{\text{ac}}_t = \text{None} \end{cases}$$

#### 2. Battery Storage & Net Grid Demand
- **Discharge ($a^{\text{bat}}_t = \text{"DISCHARGE"}$)**: Effective only if $\text{SoC}_t > 0$. Flow is clipped to available charge: $\Delta\text{SoC} = \min(10.0, \text{SoC}_t)$. Released energy $E_{\text{relief}} = (\Delta\text{SoC}/100) \cdot C_{\text{bat}} \cdot \eta_{\text{dis}}$.
- **Charge ($a^{\text{bat}}_t = \text{"CHARGE"}$)**: Effective only if $\text{SoC}_t < 100.0$. Flow is clipped to remaining capacity: $\Delta\text{SoC} = \min(10.0, 100.0 - \text{SoC}_t)$. Grid draw $E_{\text{grid-bat}} = [(\Delta\text{SoC}/100) \cdot C_{\text{bat}}] / \eta_{\text{chg}}$.
- **Grid Power & Financial Cost**:
  $$E^{\text{net}}_t = \max\left(0.0, \, E^{\text{ac}}_t + E^{\text{grid-bat}}_t - E_{\text{relief}}\right), \quad \text{Cost}_t = P_t \times E^{\text{net}}_t$$

---

### Multi-Objective Reward Function

The scalarized reward combines energy expenditure, thermal discomfort, and mechanical cycling:
$$R_t = -\left(w_1 \cdot \text{Cost}_t + w_2 \cdot \text{Pen}^{\text{comfort}}_t + w_3 \cdot \text{Pen}^{\text{switch}}_t\right) + \mathbb{I}\left(\text{Pen}^{\text{comfort}}_t = 0\right)$$
- **Weights**: $w_1 = 0.4$, $w_2 = 0.5$, $w_3 = 0.1$, comfort bonus $= 1.0$.
- **Occupancy-Consistent Comfort**:
  $$\text{Pen}^{\text{comfort}}_t = \begin{cases} 0.0 & \text{if } O_t = 0 \quad (\text{Vacant building; zero human discomfort}) \\ \max(0.0, 22.0 - T^{\text{in}}_t) + \max(0.0, T^{\text{in}}_t - 26.0) & \text{if } O_t = 1 \quad (\text{Occupied interval}) \end{cases}$$
- **Compressor Wear**: $\text{Pen}^{\text{switch}}_t = 1.0$ if $a^{\text{ac}}_t \neq a^{\text{ac}}_{t-1} \land a^{\text{ac}}_{t-1} \neq \text{None}$, else $0.0$.

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

Non-parametric paired **Wilcoxon signed-rank tests** were conducted against the strong `AdaptedRuleAgent` across all 50 shared test episodes (`results/statistical_tests.csv`):

| Comparison | Median Paired Diff | Mean Paired Diff | Wilcoxon Statistic ($W$) | $p$-value | Significant ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PPO vs. AdaptedRuleAgent** | **+0.0335** | **+0.0357** | 258.0 | **$1.49 \times 10^{-4}$** | **Yes ($p < 0.001$)** |
| **DQN vs. AdaptedRuleAgent** | **-0.0645** | **-0.0605** | 109.0 | **$1.91 \times 10^{-8}$** | **Yes (Deficit)** |

---

### Scientific Honesty & Analysis of Negative Results

1. **Why PPO Outperformed the Rule Baseline**:
   PPO learned to coordinate thermal inertia with time-of-use tariffs. Instead of aggressively toggling the compressor around setpoint boundaries like the threshold-based rule agent (16.4 switches/day), PPO maintains steady cooling states and shifts battery discharge into peak pricing, achieving higher comfort (**89.2%** vs **77.4%**) with an 84% reduction in mechanical switching wear (**2.55 switches/day**).
2. **Why DQN Underperformed**:
   DQN underperformed both PPO and the rule baseline. In an environment with 12 discrete compound actions, time-varying tariffs, and delayed battery state transitions, standard $\epsilon$-greedy exploration in Q-learning suffers from high variance and poor sample efficiency within a 20,000-step budget. Several DQN seeds fell into suboptimal local attractors (over-cooling during peak price hours or failing to utilize the battery effectively), resulting in high cross-seed variance ($SD = 0.1295$).

---

## 📈 Diagnostic Visualizations

Generated via `python scripts/make_report.py`:

### 1. Training Convergence (10 Seeds, Mean ± 1 SD)
![Learning Curves](results/learning_curves.png)

### 2. Cost vs. Comfort Pareto Frontier
![Cost vs Comfort](results/cost_vs_comfort.png)

### 3. Representative 24-Hour Day Profile (Seed 1000)
![Example Day Profile](results/example_day_trajectory.png)

---

## 🛠️ Quickstart & Reproduction Guide

### 1. Installation
Clone the repository and set up a dedicated virtual environment:
```bash
git clone https://github.com/ShagunThakur/smart-home-openenv.git
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

1. **Synthetic Thermal Linearization**: Temperature transitions follow a simplified first-order sensible cooling model. Real buildings exhibit multi-zone thermodynamic gradients, structural thermal capacitance across internal partition walls, and solar radiation geometry.
2. **Sensible vs. Latent Cooling**: The environment models dry-bulb sensible temperature only. Real air conditioners expend significant energy condensing moisture (latent heat of vaporization), which is unmodeled here.
3. **Idealized Battery Inverter**: The battery model utilizes constant 95% one-way efficiencies and discrete 1.0 kWh step flow. Real battery storage exhibits Peukert capacity degradation, C-rate dependent thermal losses, and long-term cyclic State of Health (SoH) wear.
4. **Subjective Reward Scalarization**: The reward weights ($w_1=0.4, w_2=0.5, w_3=0.1$) represent one specific scalar preference on the Pareto frontier. Other residential consumers may prioritize cost over comfort or vice versa.

---

## 📄 Citation & Contact

If you find this environment or benchmark helpful in your research, please cite:

```bibtex
@misc{thakur2026smarthomeopenenv,
  author = {Shagun Thakur},
  title = {Smart Home OpenEnv: A Gymnasium Benchmark for Residential Demand Response and Energy Storage Control},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ShagunThakur/smart-home-openenv}}
}
```

**Author**: Shagun Thakur  
**License**: MIT License
