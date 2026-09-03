# evaluation/claude_agent.py
# +---------------------------------------------------------------------------+
# |                              CLAUDE AGENT                                 |
# +---------------------------------------------------------------------------+

from agents.support_agent import SupportAgent


class ClaudeAgent(SupportAgent):
    def __init__(self):
        super().__init__()
