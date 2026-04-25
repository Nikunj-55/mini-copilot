"""
app/orchestrator/graph.py
Member 4 — Orchestration Layer

Chains the three AI modules into a single sequential pipeline:
  PolicyAnalyzer (RAG) → RiskAgent → DecisionAgent → Final Response
"""

from typing import Dict, Any


class CompliancePipeline:
    """
    Orchestrates the full compliance analysis workflow.

    Flow:
        1. PolicyAnalyzer  — retrieves relevant policy chunks via RAG
        2. RiskAgent       — identifies compliance issues & assigns risk level
        3. DecisionAgent   — generates a policy-aware recommended action
        4. Combiner        — merges all outputs into a single response dict
    """

    def __init__(self, policy_analyzer, risk_agent, decision_agent):
        """
        Accept pre-initialized module instances so the expensive
        PolicyAnalyzer FAISS index is only built once at server startup.
        """
        self.policy_analyzer = policy_analyzer
        self.risk_agent = risk_agent
        self.decision_agent = decision_agent

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the full compliance analysis pipeline for a natural-language query.

        Args:
            query: The compliance question, e.g.
                   "Does our policy require MFA for remote access?"

        Returns:
            A dict with keys:
                query, policy, context, issue, risk, reason,
                action_required, action, approval_required
        """
        print(f"\n[Pipeline] Starting analysis for query: '{query}'")

        # ── Step 1: RAG retrieval ───────────────────────────────────────
        print("[Pipeline] Step 1/3 — PolicyAnalyzer (RAG retrieval)...")
        policy_results = self.policy_analyzer.retrieve(query, top_k=3)
        policy_text    = self.policy_analyzer.retrieve_as_text(query, top_k=3)

        # Pick the highest-ranked policy name for display
        top_policy_name = (
            policy_results[0]["policy"] if policy_results else "Unknown Policy"
        )
        print(f"[Pipeline]   ↳ Top match: '{top_policy_name}'")

        # ── Step 2: Risk analysis ───────────────────────────────────────
        print("[Pipeline] Step 2/3 — RiskAgent (compliance issue detection)...")
        risk_result = self.risk_agent.analyze({
            "policy":  top_policy_name,
            "context": policy_text,
            "query":   query,
        })
        print(f"[Pipeline]   ↳ Risk level: {risk_result.get('risk', 'UNKNOWN')}")

        # ── Step 3: Decision generation ─────────────────────────────────
        print("[Pipeline] Step 3/3 — DecisionAgent (recommendation)...")
        decision_result = self.decision_agent.generate_decision(
            risk_analysis=risk_result,
            policy_context=policy_text,
        )
        print(f"[Pipeline]   ↳ Action required: {decision_result.get('action_required')}")

        # ── Step 4: Combine into final response ─────────────────────────
        print("[Pipeline] Done ✓\n")
        return {
            # Query
            "query":            query,
            # From PolicyAnalyzer (M1)
            "policy":           top_policy_name,
            "context":          policy_text,
            # From RiskAgent (M2)
            "issue":            risk_result.get("issue",   "No issue identified"),
            "risk":             risk_result.get("risk",    "LOW"),
            "reason":           risk_result.get("reason",  ""),
            # From DecisionAgent (M3)
            "action_required":  decision_result.get("action_required", False),
            "action":           decision_result.get("action", "No action required."),
            # Convenience alias used by frontend
            "approval_required": decision_result.get("action_required", False),
        }
