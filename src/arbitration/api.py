import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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

DB_PATH = os.getenv("DB_PATH", "arbitrations.db")

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
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

FRONTEND_PATH = Path(__file__).parent / "frontend.html"

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_PATH)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

def upstream_error(exc: BaseException) -> anthropic.APIStatusError | None:
    """Find an Anthropic API error in the exception chain.

    LangGraph and instructor both re-raise, so the useful error (bad key, no
    credit, rate limit) is buried several `raise ... from ...` levels down.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, anthropic.APIStatusError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None

@app.post("/arbitrate", response_model=ArbitrationResponse)
async def arbitrate(request: ArbitrationRequest):
    try:
        result = await run_in_threadpool(
            graph.invoke,
            {"question": request.question, "output": request.output, "critiques": []},
        )
    except Exception as exc:
        api_error = upstream_error(exc)
        if api_error is None:
            raise HTTPException(status_code=500, detail=f"Arbitration failed: {exc}")
        detail = api_error.body.get("error", {}).get("message") if isinstance(api_error.body, dict) else str(api_error)
        if api_error.status_code == 401:
            detail = f"{detail} Check that ANTHROPIC_API_KEY in your .env is set and valid."
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {detail}")

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