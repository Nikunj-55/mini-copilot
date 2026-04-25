import os
import json
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
            print("Warning: ANTHROPIC_API_KEY not found in environment. Agent will use fallback logic.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

    def generate_decision(self, risk_analysis: Dict[str, Any], policy_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Agentic process with Policy Cross-Referencing:
        1. Compares risk vs policy.
        2. Drafts a decision.
        3. Reflects and finalizes JSON output.
        """
        
        if not self.client:
            return self._fallback_logic(risk_analysis)

        issue = risk_analysis.get('issue', 'Unknown')
        risk = risk_analysis.get('risk', 'LOW')
        reason = risk_analysis.get('reason', 'N/A')
        
        policy_info = f"\nRelevant Policy Context:\n{policy_context}" if policy_context else "\nNo specific policy context provided."

        # PHASE 1: Generate Initial Decision with Policy Awareness
        initial_prompt = f"""
        You are a Compliance Expert. Your task is to decide on an action based on a risk analysis report and the relevant company policy.

        1. Company Policy Context:
        {policy_info}

        2. Risk Analysis Report:
        - Issue: {issue}
        - Risk Level: {risk}
        - Reason: {reason}

        Analyze the risk. If it violates the policy context provided, be strict in your recommendation.
        Draft a decision:
        1. Is action required?
        2. What is the specific recommendation? (Cite the policy if applicable)
        """
        
        try:
            # First pass: Drafting
            draft_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=600,
                messages=[{"role": "user", "content": initial_prompt}]
            )
            draft = draft_response.content[0].text

            # PHASE 2: Reflection and Finalization
            reflection_prompt = f"""
            You previously drafted this policy-aware recommendation:
            "{draft}"

            Review this draft. Ensure it is accurately mapped to the policy and provides a practical, legally sound instruction.

            Now, provide the FINAL output in strictly valid JSON format.
            JSON Schema:
            {{
                "action_required": (boolean),
                "action": (string)
            }}
            """

            final_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": initial_prompt},
                    {"role": "assistant", "content": draft},
                    {"role": "user", "content": reflection_prompt}
                ]
            )
            
            final_text = final_response.content[0].text
            start_index = final_text.find('{')
            end_index = final_text.rfind('}') + 1
            json_str = final_text[start_index:end_index]
            
            return json.loads(json_str)

        except Exception as e:
            print(f"Agentic Error: {e}")
            return self._fallback_logic(risk_analysis)

    def _fallback_logic(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        risk_level = risk_analysis.get("risk", "LOW").upper()
        action_required = risk_level in ["HIGH", "MEDIUM"]
        
        return {
            "action_required": action_required,
            "action": f"Fallback: Manual review required for {risk_analysis.get('issue')}. (API or parsing error)"
        }
