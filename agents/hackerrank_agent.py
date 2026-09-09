# evaluation/hackerrank_agent.py
# +---------------------------------------------------------------------------+
# |                           HACKERRANK AGENT                                |
# +---------------------------------------------------------------------------+


# Python Libraries
import json

# Local Libraries
from agents.support_agent import SupportAgent


class HackerrankAgent(SupportAgent):
    def __init__(self, ticket_df, chroma_model=None):
        super().__init__(ticket_df, chroma_model)

        self.title = "HackerRank Agent"
        self.company = "hackerrank"

    def ground(self, documents: list, ticket_response: str | None):
        prompt = f"""
        Analyze this prompt to ensure that the context is about Visa Incorporated a financial company that processes digital payments and to prevent
        hullicinations from the model from fabricating non-existent information.  Your response must be analyzed by the retrieved documents.
        Then compared to the ticket reponse to see if it's accurate.  If so use the ticket response 
        or update it with the appropriate answer as a response. 
        
        ## Documents
        {documents}

        ## Ticket Response
        {ticket_response}
        Also if the context is accurate assess the risk level for the issue.  Assign it as `low`, `medium`,  `high`, `critical`.

        Output JSON with keys: risk_level: string, response: string
         
        """.strip()

        response = self._model.get_response(prompt)

        grounded_response = json.load(response)
        # Compare against ticket response
        self._risk_level = grounded_response["risk_level"]
        self.response = grounded_response["response"]

    def _assess_risk(self):
        """
        Visa is about finance, financial documents
        """

        grounding_prompt = """
       
        """

        self._risk_level = None
