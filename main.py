from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from app.modules.decision_agent import DecisionAgent
from app.modules.policy_analyzer import PolicyAnalyzer
from app.modules.risk_agent import RiskAgent

app = FastAPI(title="Mini Compliance Copilot API")

# Enable CORS so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
agent = DecisionAgent()
analyzer = PolicyAnalyzer()  # loads + indexes policies at startup
risk_agent = RiskAgent()

class ComplianceRequest(BaseModel):
    risk_analysis: Dict[str, Any]
    policy_context: Optional[str] = None

class PolicyQuery(BaseModel):
    query: str
    top_k: int = 3
# Risk Request Model
class RiskRequest(BaseModel):
    policy: str
    context: str

@app.get("/")
async def root():
    return {"message": "Compliance Copilot API is running"}

@app.post("/api/decision")
async def get_decision(request: ComplianceRequest):
    try:
        result = agent.generate_decision(
            risk_analysis=request.risk_analysis,
            policy_context=request.policy_context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------
# Policy Analyzer endpoint (Member 1 — RAG)
# -----------------------------------------------------------------
@app.post("/api/policy-analyze")
async def policy_analyze(request: PolicyQuery):
    """
    RAG retrieval endpoint.
    POST { "query": "...", "top_k": 3 }
    Returns list of { policy, context, score } objects.
    """
    try:
        results = analyzer.retrieve(request.query, top_k=request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# API Endpoint for Risk Analysis
@app.post("/api/risk")
async def analyze_risk(request: RiskRequest):
    try:
        result = risk_agent.analyze({
            "policy": request.policy,
            "context": request.context
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))