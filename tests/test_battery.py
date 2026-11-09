import pytest
from env.dynamics import update_battery, compute_energy
from env.environment import SmartHomeEnv

def test_discharge_at_zero_soc_cannot_reduce_energy():
    # AC=24 consumes base 2.0 energy
    # At soc=0, DISCHARGE must not reduce energy to 1.0; it must remain 2.0
    energy = compute_energy(24, "DISCHARGE", soc=0)
    assert energy == 2.0

def test_discharge_with_positive_soc_reduces_energy():
    # At soc=50, DISCHARGE reduces AC=24 (2.0) by 50% -> 1.0
    energy = compute_energy(24, "DISCHARGE", soc=50)
    assert energy == 1.0

def test_charge_at_full_soc_consumes_no_extra_energy():
    # At soc=100, charging is ineffective and must not consume extra energy
    energy = compute_energy(24, "CHARGE", soc=100)
    assert energy == 2.0

def test_charge_with_ac_off_draws_grid_energy():
    # When AC is off (base energy 0), charging battery must draw non-zero grid power
    energy = compute_energy(None, "CHARGE", soc=50)
    assert energy == 1.0

def test_charge_with_ac_on_increases_energy():
    # AC=24 (2.0), charging increases energy by 20% -> 2.4
    energy = compute_energy(24, "CHARGE", soc=50)
    assert energy == pytest.approx(2.4)

def test_battery_soc_clipping_bounds():
    assert update_battery(100, "CHARGE") == 100
    assert update_battery(98, "CHARGE") == 100
    assert update_battery(0, "DISCHARGE") == 0
    assert update_battery(3, "DISCHARGE") == 0
    assert update_battery(50, "IDLE") == 50
    assert update_battery(50, "CHARGE") == 55
    assert update_battery(50, "DISCHARGE") == 45

def test_environment_battery_depletion_in_steps():
    env = SmartHomeEnv()
    env.reset()
    env.battery_soc = 5  # Only 1 discharge step available
    
    # Step 1: discharges 5 -> 0, gets discount
    state, reward, terminated, truncated, info = env.step((24, "DISCHARGE"))
    assert env.battery_soc == 0
    assert info.get("energy", 1.0) == 1.0
    
    # Step 2: already 0, cannot discharge, energy must be full base (2.0)
    state, reward, terminated, truncated, info = env.step((24, "DISCHARGE"))
    assert env.battery_soc == 0
    assert info.get("energy", 2.0) == 2.0
