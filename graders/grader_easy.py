# Only cares about comfort
class EasyGrader:
    def grade(self, result):
        comfort_ratio = result["comfort_ratio"]
        # direct mapping
        score = comfort_ratio
        return float(score)