import numpy as np

class BaselineAgent:
    """
    Original rule-based agent: adapts AC setpoint based on price and occupancy.
    Keeps battery IDLE.
    """
    def act(self, state):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state[:5]
        # ------ 1. If no one is home --------
        if occupancy == 0:
            if indoor_temp > 26:
                return (26, "IDLE")
            else:
                return (None, "IDLE") # turn off AC
        # ------ 2. If someone is home --------
        # Too hot -> act aggressively
        if indoor_temp > 26:
            if price < 1.0:
                return (22, "IDLE")
            else:
                return (24, "IDLE")
       
        # Moderately hot -> balanced
        elif indoor_temp > 25:
            if price < 1.0:
                return (24, "IDLE")
            else: 
                return (26, "IDLE")
        # if price is high
        elif price > 1.2 and indoor_temp <= 25:
            return (26, "IDLE")    
        # Comfortable -> maintain efficiently
        else:
            return (24, "IDLE")


class AdaptedRuleAgent:
    """
    Enhanced rule-based agent for realistic TOU tariffs and Markovian occupancy.
    Actively manages battery storage:
    - Charges during off-peak hours (price <= 0.5)
    - Discharges during peak hours (price >= 2.0)
    - Conserves energy during vacant periods
    """
    def act(self, state):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state[:5]

        # Battery arbitrage
        if price <= 0.5:
            battery_action = "CHARGE" if battery_soc < 95.0 else "IDLE"
        elif price >= 1.5:
            battery_action = "DISCHARGE" if battery_soc > 10.0 else "IDLE"
        else:
            battery_action = "IDLE"

        # AC control
        if occupancy == 0:
            ac_action = 26 if indoor_temp > 28.0 else None
        else:
            if indoor_temp > 26.0:
                ac_action = 22 if price < 1.5 else 24
            elif indoor_temp > 25.0:
                ac_action = 24 if price < 1.5 else 26
            elif indoor_temp < 22.0:
                ac_action = None
            else:
                ac_action = 24 if price < 1.0 else 26

        return (ac_action, battery_action)


class RandomAgent:
    """Selects actions uniformly at random."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        ac = self.rng.choice([22, 24, 26, None])
        bat = self.rng.choice(["CHARGE", "DISCHARGE", "IDLE"])
        return (ac, bat)


class AlwaysOnAgent:
    """Maintains a fixed AC setpoint at all times with battery IDLE."""
    def __init__(self, setpoint=24):
        self.setpoint = setpoint

    def act(self, state):
        return (self.setpoint, "IDLE")
