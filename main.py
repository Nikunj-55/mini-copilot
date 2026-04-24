from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from app.modules.decision_agent import DecisionAgent

app = FastAPI(title="Mini Compliance Copilot API")

# Enable CORS so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Agent
agent = DecisionAgent()

class ComplianceRequest(BaseModel):
    risk_analysis: Dict[str, Any]
    policy_context: Optional[str] = None

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
