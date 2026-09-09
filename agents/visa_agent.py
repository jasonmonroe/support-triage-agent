# evaluation/visa_agent.py
# +---------------------------------------------------------------------------+
# |                              VISA AGENT                                   |
# +---------------------------------------------------------------------------+

from agents.support_agent import SupportAgent


class VisaAgent(SupportAgent):
    def __init__(self, ticket_df, chroma_model=None):
        super().__init__(ticket_df, chroma_model)

        self.title = "Visa Agent"
        self.company = "visa"
