from tasks.task_hard import HardTask
from graders.grader_hard import HardGrader
from agents.baseline_agent import BaselineAgent

agent = BaselineAgent
task = HardTask()
result = task.run_episode(agent())
grader = HardGrader()
score = grader.grade(result)

print("Result:", result)
print("Score:", score)