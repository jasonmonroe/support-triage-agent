# evaluation/visa_agent.py
# +---------------------------------------------------------------------------+
# |                              VISA AGENT                                   |
# +---------------------------------------------------------------------------+

from agents.support_agent import SupportAgent


class VisaAgent(SupportAgent):
    def __init__(self):
        super().__init__()

        self.title = "Visa Agent"

    def _query(self):
        # Override Parent _query()
        pass
