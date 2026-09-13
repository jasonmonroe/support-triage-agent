# evaluation/visa_agent.py
# +---------------------------------------------------------------------------+
# |                              VISA AGENT                                   |
# +---------------------------------------------------------------------------+

# Python Libraries
from typing import List

# Vendor Libraries
from langchain_core.documents import Document

# Local Libraries
from agents.support_agent import SupportAgent
from src.enums import Status
from src.utils import log_chat_transcript


class VisaAgent(SupportAgent):
    def __init__(
        self, row_index, ticket_df, chroma_model, support_agent_model
    ):
        super().__init__(
            row_index, ticket_df, chroma_model, support_agent_model
        )
        self.title = "🤖 Visa Agent"
        self.company = "Visa"
        self._set_agent_model_title(f"{self.title} Model")

    def _query(self, query_str: str) -> List[Document]:
        if not self._chroma_model:
            raise ValueError("🚨 Chroma Model needs to be defined!")

        return self._chroma_model.query(
            query_str=query_str, company=self.company
        )

    def _assess_risk(self, request: str) -> str:
        """
        Overrides SupportAgent._assess_risk(). The base class's global
        CRITICAL_RISK_TERMS/HIGH_RISK_TERMS lists include "fraud" and
        "unauthorized" — for Claude or HackerRank those words almost
        always mean "escalate, no KB article could cover this." For Visa
        they show up in tickets the KB is specifically designed to
        answer (e.g. "my card was stolen, what do I do" has a fully
        documented process with phone numbers). This override should
        distinguish "reporting a known incident via a documented
        process" (answerable, not automatically high-risk) from
        "disputing/contesting a specific charge's legitimacy" or
        requesting a policy exception (still needs escalation) — likely
        by checking for dispute-specific language rather than treating
        every fraud/unauthorized mention the same way the base class
        does.

        Input:
            request (str): the ticket's issue text (same input the base
                class's version takes).

        Output:
            str — one of the Risk enum values (low/high/critical), same
            contract as SupportAgent._assess_risk().
        """
        pass

    def _verify_grounded_response(self, draft: dict, documents: list) -> bool:
        """Overrides base verification to add PCI-DSS compliance and financial policy checks."""

        # Base citation verification check
        is_verified = super()._verify_grounded_response(draft, documents)
        if not is_verified:
            return False

        draft_text = (
            draft.get("response", "")
            if isinstance(draft, dict)
            else str(draft)
        )

        # Financial Compliance & PII Check Prompt
        system_prompt = """
        You are a Visa Compliance & Safety Auditor.
        Evaluate whether the drafted support response violates PCI-DSS compliance or financial policy rules.

        ## DRAFTED RESPONSE
        {draft}

        ## FORBIDDEN ACTIONS & VIOLATIONS:
        1. Disclosing sensitive payment data (e.g., credit card numbers, CVVs, bank account credentials).
        2. Guaranteeing or promising financial refunds, chargeback reversals, or transaction overrides without specialist approval.

        ## OUTPUT SPECIFICATION:
        Return ONLY a JSON object:
        ```json
        {{
        "is_compliant": true,
        "reasoning": "Explanation of compliance findings."
        }}""".strip().format(
            draft=draft_text,
        )

        compliance_response = self._model.get_response(
            system_prompt, self.row_index
        )
        log_chat_transcript("VISA_COMPLIANCE_CHECK", compliance_response)

        if not compliance_response or not isinstance(
            compliance_response, dict
        ):
            self.status = Status.ESCALATED
            return False

        return compliance_response.get("is_compliant", False)
