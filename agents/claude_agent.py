# evaluation/claude_agent.py
# +---------------------------------------------------------------------------+
# |                              CLAUDE AGENT                                 |
# +---------------------------------------------------------------------------+

from agents.support_agent import SupportAgent


class ClaudeAgent(SupportAgent):
    def __init__(self, ticket_df, chroma_model=None):
        super().__init__(ticket_df, chroma_model)

        self.title = "Claude Agent"
        self.company = "claude"
