"""
main.py — Mini Compliance Copilot API
FastAPI server exposing individual module endpoints + the unified pipeline.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os

from app.modules.policy_analyzer import PolicyAnalyzer
from app.modules.risk_agent import RiskAgent
from app.modules.decision_agent import DecisionAgent
from app.orchestrator.graph import CompliancePipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mini Compliance Copilot API",
    description="AI-driven compliance analysis pipeline: RAG → Risk → Decision",
    version="1.0.0",
)

# Allow the frontend (any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Module initialization (done once at startup)
# ---------------------------------------------------------------------------
print("[Startup] Loading PolicyAnalyzer (this may take a moment)...")
analyzer    = PolicyAnalyzer()          # builds FAISS index from policies/
risk_agent  = RiskAgent()
agent       = DecisionAgent()
pipeline    = CompliancePipeline(analyzer, risk_agent, agent)
print("[Startup] All modules ready ✓")

# Serve the frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class CopilotRequest(BaseModel):
    query: str

class ComplianceRequest(BaseModel):
    risk_analysis: Dict[str, Any]
    policy_context: Optional[str] = None

class PolicyQuery(BaseModel):
    query: str
    top_k: int = 3

class RiskRequest(BaseModel):
    policy: str
    context: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health check — confirms the server is running."""
    return {"message": "Mini Compliance Copilot API is running ✓", "version": "1.0.0"}


# ── Serve frontend ──────────────────────────────────────────────────────────
@app.get("/ui", tags=["Frontend"])
async def serve_frontend():
    """Serve the compliance copilot UI."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not found. Place index.html in the frontend/ folder."}


# ── MAIN PIPELINE ENDPOINT (Member 4) ──────────────────────────────────────
@app.post("/run-copilot", tags=["Pipeline"])
async def run_copilot(request: CopilotRequest):
    """
    Full compliance analysis pipeline.
    Chains: PolicyAnalyzer → RiskAgent → DecisionAgent

    POST { "query": "Does our policy require MFA for all employees?" }

    Returns:
        {
          "query": "...",
          "policy": "...",
          "context": "...",
          "issue": "...",
          "risk": "HIGH | MEDIUM | LOW",
          "reason": "...",
          "action_required": true,
          "action": "...",
          "approval_required": true
        }
    """
    try:
        result = pipeline.run(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Individual module endpoints (for testing each module separately) ────────
@app.post("/api/policy-analyze", tags=["Modules"])
async def policy_analyze(request: PolicyQuery):
    """
    Member 1 — RAG retrieval only.
    POST { "query": "...", "top_k": 3 }
    """
    try:
        results = analyzer.retrieve(request.query, top_k=request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk", tags=["Modules"])
async def analyze_risk(request: RiskRequest):
    """
    Member 2 — Risk analysis only.
    POST { "policy": "...", "context": "..." }
    """
    try:
        result = risk_agent.analyze({
            "policy":  request.policy,
            "context": request.context,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decision", tags=["Modules"])
async def get_decision(request: ComplianceRequest):
    """
    Member 3 — Decision agent only.
    POST { "risk_analysis": { "issue": "...", "risk": "HIGH", "reason": "..." }, "policy_context": "..." }
    """
    try:
        result = agent.generate_decision(
            risk_analysis=request.risk_analysis,
            policy_context=request.policy_context,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)