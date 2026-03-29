import numpy as np
def update_temperature(indoor_temp, outdoor_temp, ac_action):
    if ac_action is not None:
        target = ac_action
        diff = target - indoor_temp
        change = np.clip(diff, -1, 1)
    else:
        diff = outdoor_temp - indoor_temp
        change = np.clip(diff, -0.5, 0.5)
    return indoor_temp + change

def update_battery(soc, action):
    if action == "CHARGE":
        soc = min(100, soc + 5)
    elif action == "DISCHARGE":
        soc = max(0, soc - 5)
    return soc

def compute_energy(ac_action, battery_action):
    if ac_action == 22:
        energy = 3
    elif ac_action == 24:
        energy = 2
    elif ac_action == 26:
        energy = 1
    else:
        energy = 0

    if battery_action == "DISCHARGE":
        energy *= 0.5
    elif battery_action == "CHARGE":
        energy *= 1.2
    return energy
