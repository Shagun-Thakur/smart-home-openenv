import numpy as np

def update_temperature(indoor_temp, outdoor_temp, ac_action):
    if ac_action is not None:
        target = ac_action
        diff = target - indoor_temp
        change = np.clip(diff, -1.0, 1.0)
    else:
        diff = outdoor_temp - indoor_temp
        change = np.clip(diff, -0.5, 0.5)
    return float(indoor_temp + change)

def update_battery(soc, action):
    if action == "CHARGE":
        return min(100, soc + 5)
    elif action == "DISCHARGE":
        return max(0, soc - 5)
    return soc

def compute_energy(ac_action, battery_action, soc=50):
    """
    Compute electricity demand from grid based on AC setting and battery state.
    - AC demand: 22 -> 3.0, 24 -> 2.0, 26 -> 1.0, None -> 0.0.
    - DISCHARGE: requires soc > 0; halves AC energy. If soc <= 0, ineffective (clipped to base).
    - CHARGE: requires soc < 100; adds 20% to active AC energy or draws 1.0 baseline when AC is off.
      If soc >= 100, ineffective (clipped to base).
    - IDLE: normal AC energy.
    """
    if ac_action == 22:
        base_energy = 3.0
    elif ac_action == 24:
        base_energy = 2.0
    elif ac_action == 26:
        base_energy = 1.0
    else:
        base_energy = 0.0

    if battery_action == "DISCHARGE":
        if soc > 0:
            energy = base_energy * 0.5
        else:
            energy = base_energy
    elif battery_action == "CHARGE":
        if soc < 100:
            if base_energy > 0:
                energy = base_energy * 1.2
            else:
                energy = 1.0
        else:
            energy = base_energy
    else:
        energy = base_energy

    return float(energy)
