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

# smart-home-openenv
A real-world reinforcement learning environment for optimizing energy usage in smart homes. The agent must balance **occupant comfort** and **electricity cost** under dynamic conditions such as changing occupancy and electricity pricing.

---

## 🚀 Overview

This project implements a complete OpenEnv-compatible environment with:
- Custom environment (`step`, `reset`, `state`)
- Real -world inspired dynamics (temperature, energy, occupancy)
- Multi-level tasks (easy -> medium -> hard)
- Automated graders (0.0 - 1.0 scoring)
- Baseline adaptive agent
- Dockerized execution

---

## Problem Setting

The goal is to control:
- **AC temperature settings**
- **Battery uasge**
Based on:
- Indoor & outdoor temperature
- Electricity price
- Occupancy
- Battery state

---

## ⚙️ Environment Design

### State Space
```python
[indoor_temp, outdoor_temp, electricity_price, battery_soc, occupancy]
```
### Action Space
```python
(ac_action, battery_action)
```
- ac_action: {22, 24, 26, None}
- battery_action: {"CHARGE", "DISCHARGE", "IDLE"}

### Transition Logic
- AC moves temperature toward setpoint(1°C per step).
- OFF -> temp drifts toward outdoor(0.5°C).
- Battery changes +5% or -5% per step.
- Occupancy and price vary dynamically.

### Reward Function
```math
R_t = - (w1 * Cost + w2 * ComfortPenalty + w3 * SwitchingPenalty)
```
Encourages:
- staying within comfort bounds
- minimizing energy cost
- avoiding excessive switching

---

## 📊Tasks & Evaluation

### Easy
- Maintain comfort
- Score = comfort

### Medium
- balance comfort + cost

### Hard
- Adapt to dynamic conditions(price + occupancy)

---

## 🤖 Baseline Agent

A rule-based agent that:
- anticipates temperature drift
- reacts to electricity price
- avoids unnecessary cooling when safe

---

## 📈Sample Output

```
comfort_ratio: 0.94
normalized_cost: 0.25
score: ~0.88
```
---

## ▶️ How to Run

### Option 1: Docker (recommended)
```bash
docker build -t smart-home-env .
docker run smart-home-env
```
### Option 2: Local
```bash
python inference.py
```
---

## 📁Project Structure
```
env/ -> environment logic
tasks/ -> task definitions
graders/ -> scoring logic
agents/ -> baseline agent
scripts/ -> execution scripts
```
---

## 👤Author

Shagun Thakur
