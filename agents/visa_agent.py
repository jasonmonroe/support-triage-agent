# evaluation/visa_agent.py
# +---------------------------------------------------------------------------+
# |                              VISA AGENT                                   |
# +---------------------------------------------------------------------------+

from agents.support_agent import SupportAgent


class VisaAgent(SupportAgent):
    def __init__(
        self, row_index, ticket_df, chroma_model, support_agent_model
    ):
        super().__init__(
            row_index, ticket_df, chroma_model, support_agent_model
        )
        self.title = "Visa Agent"
        self.company = "Visa"
        self._title_agent_model(f"{self.title} Model")

    def ground(self, documents: list, ticket_response: str | None):
        pass
