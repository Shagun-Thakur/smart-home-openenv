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
    @staticmethod
    def battery_act(price: float, battery_soc: float) -> str:
        """Arbitrage battery based on Time-of-Use electricity price."""
        if price <= 0.5:
            return "CHARGE" if battery_soc < 95.0 else "IDLE"
        elif price >= 1.5:
            return "DISCHARGE" if battery_soc > 10.0 else "IDLE"
        else:
            return "IDLE"

    def act(self, state):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state[:5]

        # Battery arbitrage
        battery_action = self.battery_act(price, battery_soc)

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


class OccupancyAware_24:
    """
    Occupancy-aware baseline agent:
    - Sets AC to 24°C when occupied.
    - Turns AC off (None) when vacant.
    - Battery remains IDLE at all times.
    """
    def act(self, state, as_index: bool = False):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state[:5]
        is_occupied = int(round(occupancy)) == 1
        ac_action = 24 if is_occupied else None
        battery_action = "IDLE"
        if as_index:
            from env.environment import encode_action
            return encode_action(ac_action, battery_action)
        return (ac_action, battery_action)

    def act_index(self, state) -> int:
        return self.act(state, as_index=True)


class OccupancyAware_Precool:
    """
    Occupancy-aware pre-cooling baseline agent:
    - Sets AC to 24°C when occupied.
    - When vacant: runs AC at 26°C if indoor temperature exceeds T_hot (default 26.0°C,
      empirically selected via grid search on design seeds 2000-2049) to prevent
      excessive thermal mass heat buildup; otherwise turns AC off (None).
    - Battery remains IDLE at all times.
    """
    def __init__(self, t_hot: float = 26.0):
        self.t_hot = float(t_hot)

    def act(self, state, as_index: bool = False):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state[:5]
        is_occupied = int(round(occupancy)) == 1
        if is_occupied:
            ac_action = 24
        else:
            ac_action = 26 if indoor_temp > self.t_hot else None
        battery_action = "IDLE"
        if as_index:
            from env.environment import encode_action
            return encode_action(ac_action, battery_action)
        return (ac_action, battery_action)

    def act_index(self, state) -> int:
        return self.act(state, as_index=True)


class OccupancyAware_TOU:
    """
    Occupancy-aware TOU agent:
    - Reuses OccupancyAware_24 AC logic (24°C when occupied, off when vacant).
    - Reuses AdaptedRuleAgent battery logic (charge off-peak <= 0.5, discharge peak >= 1.5).
    """
    def act(self, state, as_index: bool = False):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state[:5]
        is_occupied = int(round(occupancy)) == 1
        ac_action = 24 if is_occupied else None
        battery_action = AdaptedRuleAgent.battery_act(price, battery_soc)
        if as_index:
            from env.environment import encode_action
            return encode_action(ac_action, battery_action)
        return (ac_action, battery_action)

    def act_index(self, state) -> int:
        return self.act(state, as_index=True)


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
