class BaselineAgent:
    def act(self, state):
        indoor_temp, outdoor_temp, price, battery_soc, occupancy = state
        # ------ 1. If no one is home --------
        if occupancy == 0:
            if indoor_temp > 26:
                return (26, "IDLE")
            else:
                return(None, "IDLE") # turn off AC
        # ------ 2. If someone is home --------
        # Too hot -> act aggressively
        if indoor_temp > 26:
            if price < 1.0:
                return(22, "IDLE")
            else:
                return(24, "IDLE")
       
        # Moderately hot -> balanced
        elif indoor_temp > 25:
            if price < 1.0:
                return (24, "IDLE")
            else: 
                return (26, "IDLE")
        # if price if high
        elif price > 1.2 and indoor_temp <= 25:
            return(26, "IDLE")    
        # Comfortable -> maintain efficiently
        else:
            return (24, "IDLE")