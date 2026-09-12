"""FastAPI application providing the AI Support Agent REST API and Evaluation endpoints."""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import SELECTED_BRAND, INTENT_TAXONOMY, REPORTS_DIR, GOLDEN_DATA_PATH
from src.pipeline import SupportAgentPipeline
from backend.schemas import (
    CustomerQueryRequest, AgentResponse, EvaluationReport
)

app = FastAPI(
    title=f"Hiver AI Support Agent - @{SELECTED_BRAND}",
    description="Evidence-Grounded AI Support Agent with semantic intent classification, historical precedent retrieval, grounded reply drafting, and risk-aware escalation.",
    version="1.0.0"
)

# Enable CORS for frontend web experience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

# Global Pipeline Instance
pipeline = SupportAgentPipeline()


@app.on_event("startup")
def startup_event():
    """Warm up models and FAISS retrieval index on startup."""
    pipeline.initialize()


@app.get("/")
def serve_index():
    """Serves the interactive web triage console and evaluation dashboard."""
    if FRONTEND_FILE.exists():
        from fastapi.responses import HTMLResponse
        with open(FRONTEND_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return {"message": "Hiver AI Support Agent API running. Visit /docs for Swagger UI."}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "brand": SELECTED_BRAND,
        "is_initialized": pipeline.is_initialized
    }


@app.get("/api/taxonomy")
def get_taxonomy():
    """Returns the brand-specific intent taxonomy definition and risk levels."""
    return {
        "brand": SELECTED_BRAND,
        "taxonomies": INTENT_TAXONOMY
    }


@app.post("/api/agent/analyze", response_model=AgentResponse)
def analyze_customer_message(request: CustomerQueryRequest):
    """Core Agent Endpoint: Understands intent, retrieves historical precedent evidence,

    evaluates evidence sufficiency, drafts a grounded reply, and makes an auditable auto-handle/escalate decision.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Customer message cannot be empty.")

    response = pipeline.process(request.message, context=request.conversation_context)
    return response


@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    """Returns benchmark results across Baselines and Proposed System."""
    results_file = REPORTS_DIR / "results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run evaluation first.")

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/evaluation/golden")
def get_golden_samples(limit: int = 50):
    """Returns sample golden evaluation records with ground-truth labels."""
    if not GOLDEN_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Golden set not found.")

    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


@app.get("/api/evaluation/failures")
def get_failure_cases():
    """Returns top analyzed failure cases with hypotheses and expected behaviors."""
    failures_file = REPORTS_DIR / "failure_analysis.md"
    if failures_file.exists():
        with open(failures_file, "r", encoding="utf-8") as f:
            return {"markdown": f.read()}
    return {"markdown": "Failure analysis report in progress."}
