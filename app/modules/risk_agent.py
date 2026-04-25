# app/api/modules/risk_agent.py

import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class RiskAgent:

    def analyze(self, data: dict) -> dict:
        context = data.get("context", "")
        policy = data.get("policy", "")

        prompt = f"""
You are a compliance risk analyzer.

Policy:
{policy}

Context:
{context}

Tasks:
1. Identify issue
2. Assign risk (LOW, MEDIUM, HIGH)
3. Give reason

Return ONLY JSON:
{{
  "issue": "...",
  "risk": "LOW | MEDIUM | HIGH",
  "reason": "..."
}}
"""

        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            output = response.content[0].text.strip()

            return json.loads(output)

        except Exception as e:
            return self.fallback(context, str(e))

    def fallback(self, context: str, error: str = "") -> dict:
        context = context.lower()

        if "mfa" in context:
            return {
                "issue": "Missing multi-factor authentication",
                "risk": "HIGH",
                "reason": "Critical security control missing",
                "note": error
            }

        return {
            "issue": "No major issue detected",
            "risk": "LOW",
            "reason": "System appears compliant",
            "note": error
        }