# evaluation/claude_agent.py
# +---------------------------------------------------------------------------+
# |                              CLAUDE AGENT                                 |
# +---------------------------------------------------------------------------+

# Python Libraries
from typing import List

# Vendor Libraries
from langchain_core.documents import Document

# Local Libraries
from agents.support_agent import SupportAgent


class ClaudeAgent(SupportAgent):
    def __init__(
        self, row_index, ticket_df, chroma_model, support_agent_model
    ):
        super().__init__(
            row_index, ticket_df, chroma_model, support_agent_model
        )

        self.title = "🤖 Claude Agent"
        self.company = "Claude"
        self._set_agent_model_title(f"{self.title} Model")

    def _query(self, query_str: str) -> List[Document]:
        if not self._chroma_model:
            raise ValueError("🚨 Chroma Model needs to be defined!")

        return self._chroma_model.query(
            query_str=query_str, company=self.company
        )
