import os
import json
import re
from typing import Dict, Any, Optional
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class DecisionResponse(BaseModel):
    action_required: bool
    action: str

class DecisionAgent:
    """
    Policy-Aware Agentic Decision Agent module (Member 3)
    Evaluates risk analysis in the context of the actual policy text.
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[DecisionAgent] WARNING: ANTHROPIC_API_KEY not found. Will use fallback logic.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

    def generate_decision(self, risk_analysis: Dict[str, Any], policy_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Agentic process with Policy Cross-Referencing:
        1. Compares risk vs policy — drafts a decision.
        2. Reflects and finalizes as JSON.
        """

        if not self.client:
            return self._fallback_logic(risk_analysis)

        issue  = risk_analysis.get('issue', 'Unknown')
        risk   = risk_analysis.get('risk', 'LOW')
        reason = risk_analysis.get('reason', 'N/A')

        policy_info = f"\nRelevant Policy Context:\n{policy_context}" if policy_context else "\nNo specific policy context provided."

        # PHASE 1: Draft a plain-English recommendation
        initial_prompt = f"""You are a Compliance Expert. Based on the risk report and company policy below, decide what action to take.

{policy_info}

Risk Analysis Report:
- Issue: {issue}
- Risk Level: {risk}
- Reason: {reason}

Draft your recommendation:
1. Is action required? (yes/no and why)
2. What specific action should be taken? (cite the policy section if applicable)
Be practical, specific, and accurate."""

        try:
            # First pass: get a plain-English draft
            draft_response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": initial_prompt}]
            )
            draft = draft_response.content[0].text
            print(f"[DecisionAgent] Draft: {draft[:150]}...")

            # PHASE 2: Convert draft to strict JSON using prefill trick
            reflection_prompt = f"""Convert this compliance recommendation into a JSON object.

Recommendation:
\"\"\"{draft}\"\"\"

Rules:
- action_required must be a boolean (true or false)
- action must be a single clear sentence with the recommended action"""

            final_response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=300,
                messages=[
                    {"role": "user",      "content": reflection_prompt},
                    {"role": "assistant", "content": '{"action_required":'},  # ← prefill trick
                ],
            )

            # Reconstruct the full JSON string
            raw = '{"action_required":' + final_response.content[0].text.strip()
            print(f"[DecisionAgent] Raw JSON: {raw[:200]}")

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"[DecisionAgent] JSON parse error: {e} | Raw: {raw[:300] if 'raw' in dir() else 'N/A'}")
            return self._fallback_logic(risk_analysis)
        except Exception as e:
            print(f"[DecisionAgent] API error: {e}")
            return self._fallback_logic(risk_analysis)

    def _fallback_logic(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        risk_level = risk_analysis.get("risk", "LOW").upper()
        issue      = risk_analysis.get("issue", "the identified compliance issue")
        action_required = risk_level in ["HIGH", "MEDIUM"]

        if not action_required:
            return {
                "action_required": False,
                "action": f"No immediate action needed. Continue monitoring for: {issue}.",
            }

        urgency = "Immediately remediate" if risk_level == "HIGH" else "Schedule a review to address"
        return {
            "action_required": True,
            "action": f"{urgency} the following issue: {issue}. Escalate to the compliance team for policy alignment.",
        }
