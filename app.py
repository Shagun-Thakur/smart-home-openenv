from fastapi import FastAPI
from env.environment import SmartHomeEnv

app = FastAPI()
env = SmartHomeEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset():
    obs, _ = env.reset()
    return {"state": obs.tolist()}
