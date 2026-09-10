# evaluation/claude_agent.py
# +---------------------------------------------------------------------------+
# |                              CLAUDE AGENT                                 |
# +---------------------------------------------------------------------------+

from agents.support_agent import SupportAgent


class ClaudeAgent(SupportAgent):
    def __init__(
        self, row_index, ticket_df, chroma_model, support_agent_model
    ):
        super().__init__(
            row_index, ticket_df, chroma_model, support_agent_model
        )

        self.title = "Claude Agent"
        self.company = "Claude"

    def ground(self, documents: list):
        pass
