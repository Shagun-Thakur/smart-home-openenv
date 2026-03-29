# combine: comfort and cost
class MediumGrader:
    def grade(self, result):
        comfort = result["comfort_ratio"]
        cost = result["normalized_cost"]
        score = 0.6 * comfort + 0.4 * (1 - cost)
        return float(score)