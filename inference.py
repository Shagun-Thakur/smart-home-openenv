from agents.baseline_agent import BaselineAgent

from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask

from graders.grader_easy import EasyGrader
from graders.grader_medium import MediumGrader
from graders.grader_hard import HardGrader

def run_task(task, grader, agent, name):
    result = task.run_episode(agent)
    score = grader.grade(result)
    print(f"\n--- {name} ---")
    print("Result: ", result)
    print("Score: ", score)
    return score

def main():
    agent = BaselineAgent()
    # Easy
    run_task(EasyTask(), EasyGrader(), agent, "Easy Task")
    # Medium
    run_task(MediumTask(), MediumGrader(), agent, "Medium Task" )
    # Hard
    run_task(HardTask(), HardGrader(), agent, "Hard Task")

if __name__ == "__main__":
    main()