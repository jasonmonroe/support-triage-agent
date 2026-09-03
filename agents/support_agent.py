# src/support_agent.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries

from abc import ABC


class SupportAgent(ABC):
    """
    A class to handle support agent operations.
    """

    def __init__(self):
        """
        Initialize the SupportAgent class.
        """

        # Data files for the company
        self._support = []

        # Read only
        self.issue = ""
        self.subject = ""
        self.company = ""

        # Outputs (override)
        self.response = ""
        self.product_area = ""
        self.status = ""
        self.request_type = ""

        # Outputs
        self.justification = ""

    def get_request_type(self, request: str) -> str:
        """
        Identify the request type from the request.
        """
        return "request type"

    def classify_issue(self, request: str) -> str:
        """
        Classify the issue from the request.
        """
        return "issue classification"

    def _assess_urgency(self, request: str) -> str:
        """
        Assess the urgency of the request.
        """
        return "urgency assessment"

    def _assess_risk(self, request: str) -> str:
        """
        Assess the risk of the request.
        """
        return "risk assessment"

    def _make_decision(self, request: str) -> str:
        """
        Make a decision based on the request.
        """
        return "decision"

    def _get_relevant_knowledge(self, request: str) -> str:
        """
        Get the relevant knowledge from the request via the /data/ directory.
        This will be a RAG retrieval process.
        """
        return "relevant knowledge"

    def export(self):
        """
        Returns output.  Use three inputs and 5 outputs (status, product_area, response, justificiation, request_ type)
        to create the output.csv row
        """

        return "output"
