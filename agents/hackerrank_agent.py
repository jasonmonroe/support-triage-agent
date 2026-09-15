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

    def _get_verify_hook(self):
        return self._check_forbidden_terms

    def _check_forbidden_terms(self, draft: dict) -> bool:
        """Plugged into RagAgent's verification gate (see

        SupportAgent._get_verify_hook()) to block unauthorized HackerRank
        score-manipulation promises on top of base citation verification.
        """
        response_text = (
            draft.get("response", "").lower()
            if isinstance(draft, dict)
            else ""
        )

        for term in self._forbidden_terms():
            if term in response_text:
                log_chat_transcript(
                    "🤖 HACKERRANK_AGENT",
                    f"{self.title}: Forbidden promise detected: '{term}'",
                )
                return False

        return True
