# app/modules/risk_agent.py

import json
import os
import re
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RiskAgent:

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[RiskAgent] WARNING: ANTHROPIC_API_KEY not set. Will use fallback.")
            self.client = None
        else:
            self.client = Anthropic(api_key=api_key)

    def analyze(self, data: dict) -> dict:
        context = data.get("context", "")
        policy  = data.get("policy", "")
        query   = data.get("query", "")

        if not self.client:
            return self._smart_fallback(context, policy, "No API key configured")

        prompt = f"""You are a compliance risk analyzer. A user is asking about a compliance scenario. Your job is to assess the risk level based on the policy and the user's query.

User Query / Scenario:
{query}

Policy Name: {policy}

Relevant Policy Context:
{context}

Based on the user's query and the policy above, respond with a JSON object containing:
- "issue": A specific compliance issue identified (string, or null if fully compliant)
- "risk": Risk level — must be exactly one of: "LOW", "MEDIUM", or "HIGH"
- "reason": A clear explanation tying the user's scenario to the specific policy requirement

Assign risk levels as follows:
- HIGH: Clear violation, mandatory control missed, or immediate threat
- MEDIUM: Partial compliance, ambiguous adherence, or requires prompt action
- LOW: Compliant or minor/informational observation only

Respond with ONLY the JSON object, no explanation outside it."""

        try:
            # Prefill the assistant response with '{' — forces Claude to output raw JSON
            # with no preamble, no markdown fences, guaranteed parseable
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=400,
                temperature=0,
                messages=[
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": '{"issue":'},   # ← prefill trick
                ],
            )

            # Claude completes from '{"issue": ...' — prepend it back
            raw = '{"issue":' + response.content[0].text.strip()
            print(f"[RiskAgent] Raw response: {raw[:200]}")

            result = json.loads(raw)

            # Normalize: Claude sometimes uses 'risk_level' instead of 'risk'
            if "risk_level" in result and "risk" not in result:
                result["risk"] = result.pop("risk_level")

            # Ensure risk is always uppercase and valid
            result["risk"] = result.get("risk", "LOW").upper()
            if result["risk"] not in ("LOW", "MEDIUM", "HIGH"):
                result["risk"] = "LOW"

            return result

        except json.JSONDecodeError as e:
            print(f"[RiskAgent] JSON parse error: {e} | Raw: {raw[:300]}")
            return self._smart_fallback(context, policy, str(e))
        except Exception as e:
            print(f"[RiskAgent] API error: {e}")
            return self._smart_fallback(context, policy, str(e))

    def _smart_fallback(self, context: str, policy: str, error: str = "") -> dict:
        """
        Last-resort fallback using simple keyword logic.
        Only used when the Claude API call completely fails.
        """
        ctx = context.lower()

        if "shared" in ctx and ("account" in ctx or "admin" in ctx):
            return {
                "issue": "Shared accounts detected — violates individual accountability policy",
                "risk": "HIGH",
                "reason": "Policy requires unique user accounts; shared credentials prevent audit trails",
                "note": error,
            }
        if "mfa" in ctx or "multi-factor" in ctx:
            return {
                "issue": "Multi-factor authentication gap identified",
                "risk": "HIGH",
                "reason": "MFA is mandatory per policy but may not be enforced",
                "note": error,
            }
        if "retain" in ctx or "retention" in ctx:
            return {
                "issue": "Data retention compliance review needed",
                "risk": "MEDIUM",
                "reason": "Retention periods must align with policy requirements",
                "note": error,
            }
        if "incident" in ctx or "breach" in ctx:
            return {
                "issue": "Incident response process requires review",
                "risk": "MEDIUM",
                "reason": "Ensure incident procedures match policy mandates",
                "note": error,
            }

        return {
            "issue": "Manual compliance review required",
            "risk": "LOW",
            "reason": "Unable to perform automated analysis — API unavailable",
            "note": error,
        }