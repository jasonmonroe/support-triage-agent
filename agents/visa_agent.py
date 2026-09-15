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

    def _get_verify_hook(self):
        return self._compliance_check

    def _compliance_check(self, draft: dict) -> bool:
        """Plugged into RagAgent's verification gate (see

        SupportAgent._get_verify_hook()) to add PCI-DSS compliance and
        financial policy checks on top of base citation verification.
        """
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
        log_chat_transcript(
            "🤖 VISA_AGENT", f"Visa Compliance Check: {compliance_response}"
        )

        if not compliance_response or not isinstance(
            compliance_response, dict
        ):
            return False

        return compliance_response.get("is_compliant", False)
