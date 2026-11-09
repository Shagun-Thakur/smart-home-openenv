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

### Analysis & Empirical Findings

1. **PPO vs. Rule Baselines**:
   PPO statistically outperforms both threshold-based rule agents (`AdaptedRuleAgent`, $p = 5.66 \times 10^{-5}$; `BaselineAgent`, $p = 4.97 \times 10^{-14}$). The rule agents frequently toggle the compressor around fixed setpoint boundaries (15.4–16.4 switches/day), incurring switching penalties and lower comfort, whereas PPO maintains steady states ($2.55 \pm 0.88$ switches/day).

2. **PPO vs. Constant Setpoint (AlwaysOn_24)**:
   On the scalar combined score, PPO is **not statistically distinguishable** from `AlwaysOn_24` (median paired difference $+0.0027$, Holm $p = 0.789$, 30 wins in 50 episodes). Rather than dominating `AlwaysOn_24`, PPO reaches a **different cost/comfort trade-off**:
   - `AlwaysOn_24` provides near-perfect comfort ($0.9686 \pm 0.0178$) at a constant high energy cost (**\$52.00 ± \$0.00**).
   - `PPO` achieves a **25.8% cost reduction** (**\$38.56 ± \$11.13**) while maintaining reasonable comfort ($0.8924 \pm 0.1329$).
   Whether PPO is preferred over `AlwaysOn_24` depends on the occupant's subjective willingness to trade comfort for energy savings.

3. **DQN Underperformance**:
   DQN does not beat `AlwaysOn_24` (0 wins out of 50, Holm $p = 1.07 \times 10^{-14}$) and underperforms PPO across all seeds. In this 12-discrete-action environment with delayed battery dynamics and varying tariffs, standard $\epsilon$-greedy exploration exhibited high cross-seed variance ($SD = 0.1295$). While 5 DQN seeds achieved scores around 0.80–0.83, two seeds (seeds 1 and 8) collapsed to scores around 0.48–0.52 due to poor setpoint control.

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
4. **Subjective Reward Scalarization**: The reward weights (cost: 0.4, comfort: 0.5, switching wear: 0.1) represent one specific scalar preference on the Pareto frontier. Other residential consumers may prioritize cost over comfort or vice versa.

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
