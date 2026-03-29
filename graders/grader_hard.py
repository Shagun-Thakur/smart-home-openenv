class HardGrader:
    def grade(self, result):
        comfort = result["comfort_ratio"]
        cost = result["normalized_cost"]
        # stronger penalty for bad comfort
        score = 0.7 * comfort + 0.3 * (1 - cost)
        return float(score)