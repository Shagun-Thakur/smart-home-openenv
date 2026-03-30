import os
def call_openai_stub(state):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping LLM call.")
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key = api_key)
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this smart home state briefly: {state}"
                }
            ],
            max_tokens= 50
        )
        print("LLM Response: ", response.choice[0].message.content)

    except Exception as e:
        print("LLM call failed: ", str(e))

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
    dummy_state = [30, 35, 1, 50, 1]
    call_openai_stub(dummy_state)
    # Easy
    run_task(EasyTask(), EasyGrader(), agent, "Easy Task")
    # Medium
    run_task(MediumTask(), MediumGrader(), agent, "Medium Task" )
    # Hard
    run_task(HardTask(), HardGrader(), agent, "Hard Task")

if __name__ == "__main__":
    main()