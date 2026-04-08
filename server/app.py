"""
FastAPI server for the Email Triage OpenEnv environment (v0.4.0).
"""
import sys, os, threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict
from env import EmailTriageEnv
from models import Action, Observation, State

_env = EmailTriageEnv()
_env_lock = threading.Lock()

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]

app = FastAPI(
    title="Email Triage OpenEnv", version="0.4.0",
    description="Real-world B2B SaaS email triage with CSAT decay, consequence chains, escalation budget, prompt injection, confidence calibration, and 4 difficulty tiers.",
)

@app.get("/")
def read_root():
    return JSONResponse({"message": "Email Triage OpenEnv v0.4.0", "status": "online", "documentation": "/docs",
        "endpoints": {"reset": "/reset (POST)", "step": "/step (POST)", "state": "/state (GET)",
            "health": "/health (GET)", "metadata": "/metadata (GET)", "schema": "/schema (GET)"}})

@app.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else "easy") or "easy"
    seed = req.seed if req else None
    episode_id = req.episode_id if req else None
    try:
        with _env_lock:
            obs = _env.reset(seed=seed, episode_id=episode_id, task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return JSONResponse({"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done})

@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    with _env_lock:
        obs = _env.step(action)
    return JSONResponse({"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done, "info": {}})

@app.get("/state")
def state():
    with _env_lock:
        s = _env.state
    return JSONResponse(s.model_dump())

@app.get("/health")
def health():
    return JSONResponse({"status": "healthy", "env": "email-triage-env"})

@app.get("/metadata")
def metadata():
    return JSONResponse(_env.get_metadata().model_dump())

@app.get("/schema")
def schema():
    return JSONResponse({"action": Action.model_json_schema(), "observation": Observation.model_json_schema(), "state": State.model_json_schema()})

@app.get("/schema/action")
def schema_action():
    return JSONResponse(Action.model_json_schema())

@app.get("/schema/observation")
def schema_observation():
    return JSONResponse(Observation.model_json_schema())

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
