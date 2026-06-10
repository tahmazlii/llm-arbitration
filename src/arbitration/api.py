import json
import sqlite3
import uuid
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graph import graph
from .models import Verdict

# --- Request/response shapes -------------------------------------------------

class ArbitrationRequest(BaseModel):
    question: str
    output: str

class ArbitrationResponse(BaseModel):
    id: str
    verdict: Verdict

# --- Storage (SQLite) --------------------------------------------------------

DB_PATH = "arbitrations.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS arbitrations ("
            "id TEXT PRIMARY KEY, "
            "question TEXT, "
            "output TEXT, "
            "verdict TEXT)"  # verdict stored as JSON text
        )

def save_arbitration(arbitration_id: str, question: str, output: str, verdict: Verdict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO arbitrations (id, question, output, verdict) VALUES (?, ?, ?, ?)",
            (arbitration_id, question, output, verdict.model_dump_json()),
        )

def load_arbitration(arbitration_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, verdict FROM arbitrations WHERE id = ?",
            (arbitration_id,),
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "verdict": Verdict.model_validate_json(row[1])}

# --- App ---------------------------------------------------------------------

app = FastAPI(title="LLM Output Arbitration System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # fine for local dev; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

@app.post("/arbitrate", response_model=ArbitrationResponse)
async def arbitrate(request: ArbitrationRequest):
    result = await run_in_threadpool(
        graph.invoke,
        {"question": request.question, "output": request.output, "critiques": []},
    )
    verdict = result["verdict"]
    arbitration_id = str(uuid.uuid4())
    save_arbitration(arbitration_id, request.question, request.output, verdict)
    return ArbitrationResponse(id=arbitration_id, verdict=verdict)

@app.get("/arbitrations/{arbitration_id}", response_model=ArbitrationResponse)
async def get_arbitration(arbitration_id: str):
    record = load_arbitration(arbitration_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Arbitration not found")
    return ArbitrationResponse(id=record["id"], verdict=record["verdict"])