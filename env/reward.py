def compute_comfort_penalty(temp, cmin, cmax):
    if temp < cmin:
        return cmin - temp
    elif temp > cmax:
        return temp - cmax
    return 0

def compute_reward(cost, comfort_penalty, switching_penalty):
    w1, w2, w3 = 0.4, 0.5, 0.1
    reward = - (w1 * cost + w2 * comfort_penalty + w3 * switching_penalty)
    if comfort_penalty == 0:
        reward += 1    # bonus for comfort
    return reward
