def compute_comfort_penalty(temp, cmin, cmax, occupancy=1):
    """
    Compute thermal discomfort penalty.
    Discomfort only applies when occupants are present (occupancy == 1).
    When unoccupied (occupancy == 0), penalty is strictly 0.0.
    """
    if occupancy == 0:
        return 0.0
    if temp < cmin:
        return float(cmin - temp)
    elif temp > cmax:
        return float(temp - cmax)
    return 0.0

def is_comfortable(temp, cmin, cmax, occupancy=1):
    """
    Check if current indoor temperature is comfortable.
    Unoccupied states are comfortable by definition (no occupant discomfort).
    """
    if occupancy == 0:
        return True
    return cmin <= temp <= cmax

def compute_reward(cost, comfort_penalty, switching_penalty):
    w1, w2, w3 = 0.4, 0.5, 0.1
    reward = - (w1 * cost + w2 * comfort_penalty + w3 * switching_penalty)
    if comfort_penalty == 0:
        reward += 1.0    # bonus for comfort
    return float(reward)
