# agents/hackerrank_agent.py
# +---------------------------------------------------------------------------+
# |                           HACKERRANK AGENT                                |
# +---------------------------------------------------------------------------+

# Python Libraries
from typing import List

# Vendor Libraries
from langchain_core.documents import Document

# Local Libraries
from agents.support_agent import SupportAgent
from src.enums import Status
from src.utils import log_chat_transcript


class HackerrankAgent(SupportAgent):
    def __init__(
        self, row_index, ticket_df, chroma_model, support_agent_model
    ):
        super().__init__(
            row_index, ticket_df, chroma_model, support_agent_model
        )

        self.title = "🤖 HackerRank Agent"
        self.company = "Hackerrank"
        self._set_agent_model_title(f"{self.title} Model")

    def _query(self, query_str: str) -> List[Document]:
        if not self._chroma_model:
            raise ValueError("🚨 Chroma Model needs to be defined!")

        return self._chroma_model.query(
            query_str=query_str, company=self.company
        )

    def _forbidden_terms(self) -> list:
        return [
            "increase your score",
            "increase score",
            "re-grade",
            "regrade",
            "override score",
            "change score",
            "bypass evaluation",
            "grant score",
        ]

    def _verify_grounded_response(self, draft: dict, documents: list) -> bool:
        """Overrides SupportAgent._verify_grounded_response() to add a

        business-rule check on top of base citation verification.
        """
        # 1. Run base citation verification gate first
        is_verified = super()._verify_grounded_response(draft, documents)
        if not is_verified:
            return False

        # 2. Extract response text safely
        response_text = (
            draft.get("response", "").lower()
            if isinstance(draft, dict)
            else ""
        )

        # 3. Check for unauthorized HackerRank policy promises
        forbidden_terms = self._forbidden_terms()

        for term in forbidden_terms:
            if term in response_text:
                log_chat_transcript(
                    "🤖 HACKERRANK_AGENT",
                    f"{self.title}: Forbidden promise detected: '{term}'",
                )
                self.status = Status.ESCALATED
                return False

        return True
