# src/support_agent.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries

from abc import ABC

import pandas as pd

CRITICAL_RISK_TERMS = (
    "fraud",
    "unauthorized",
    "breach",
    "hacked",
    "security vulnerability",
    "data leak",
    "legal action",
    "lawsuit",
    "subpoena",
)

HIGH_RISK_TERMS = (
    "refund",
    "chargeback",
    "billing dispute",
    "cancel my account",
    "delete my account",
    "gdpr",
)

URGENT_TERMS = (
    "down",
    "outage",
    "cannot access",
    "can't access",
    "blocked",
    "urgent",
    "asap",
    "immediately",
)

COMPANY_KEYWORDS = {
    "claude": ("claude", "anthropic"),
    "hackerrank": ("hackerrank", "test", "candidate", "interview"),
    "visa": ("visa", "card", "payment", "merchant"),
}


class SupportAgent(ABC):
    """
    A class to handle support agent operations.
    """

    def __init__(self, ticket_df: pd.DataFrame, chroma_model=None):
        """
        Initialize the SupportAgent class.
        """
        self.title = "Support Agent"
        self.chroma_model = chroma_model
        self._model = None  # lazily set to a SupportAgentModel, if used

        # Read only
        self.issue = None
        self.subject = None
        self.company = None

        # Outputs (override)
        self.response = None
        self.product_area = None
        self.status = None  # Replied or Escalated
        self.request_type = None

        # Outputs
        self.justification = None

        self._risk_level = None  # low, high, critical
        self._urgency = None  # normal, high

        self._set_attrs(ticket_df)

        self.company = self._normalize_company(self.company)
        if not self.company:
            self.company = self._find_company()

    def _set_attrs(self, row) -> None:
        # itertuples() rows are namedtuples (._asdict()); iterrows()/dict
        # rows expose .items() directly.
        has_asdict = hasattr(row, "_asdict")
        items = row._asdict().items() if has_asdict else row.items()
        for column, value in items:
            key = column.replace(" ", "_").lower()
            if hasattr(self, key):
                setattr(self, key, value)

    def _normalize_company(self, company: str | None) -> str | None:
        if not company or company.strip().lower() == "none":
            return None
        return company.strip().lower()

    def get_request_type(self) -> str:
        """
        Identify the request type from the request.
        """
        return self.request_type

    def classify(self) -> None:
        """
        Classify and assess request type, product area, risk.
        request_type/product_area are still finalized by the LLM's
        structured output later (real classification, not keyword
        matching) — this pass only extracts cheap, deterministic
        risk/urgency signals so escalation doesn't depend solely on the
        model's judgment.
        """
        if not self.issue or len(self.issue) < 10:
            self.request_type = "invalid"

        self._risk_level = self._assess_risk(self.issue)
        self._urgency = self._assess_urgency(self.issue)

    def classify_issue(self, request: str) -> str:
        """
        Classify the issue from the request.
        """
        return "issue classification"

    def _assess_urgency(self, request: str) -> str:
        """
        Assess the urgency of the request.
        """
        text = (request or "").lower()
        is_urgent = any(term in text for term in URGENT_TERMS)
        return "high" if is_urgent else "normal"

    def _assess_risk(self, request: str) -> str:
        """
        Assess the risk of the request.
        """
        text = (request or "").lower()

        if any(term in text for term in CRITICAL_RISK_TERMS):
            return "critical"

        if any(term in text for term in HIGH_RISK_TERMS):
            return "high"

        return "low"

    def make_decision(self, request: str) -> str:
        """
        Make a decision based on the request: reply or escalate, based purely
        on the risk level from classify() (pre-retrieval).
        """
        is_risky = self._risk_level in ("high", "critical")
        return "escalated" if is_risky else "replied"

    def retrieve_relevant_documents(self) -> list:
        """
        Get the relevant knowledge from the request via the /data/ directory.
        This is the RAG retrieval process.
        """
        subject = f"Subject: {self.subject}\n" if self.subject else ""
        query = f"{subject}Issue: {self.issue}".strip()

        return self._query(query)

    def ground(self, documents: list) -> None:
        """
        Decide reply-vs-escalate using the risk-based decision plus retrieval
        confidence: even a low-risk ticket gets escalated if nothing in the
        knowledge base grounds an answer, so the LLM is never asked to answer
        ungrounded. Also lifts product_area from the top-matching chunk's
        metadata rather than asking the LLM to guess it freehand.
        """
        self.status = self.make_decision(self.issue)

        if self.status == "escalated":
            self.justification = (
                f"Escalated: ticket matched '{self._risk_level}' risk signals."
            )
            return

        if not documents:
            self.status = "escalated"
            self.justification = (
                "Escalated: no knowledge base match found to ground a"
                " response."
            )
            return

        self.product_area = documents[0].metadata.get("product_area")

    def export(self) -> dict:
        "export data needed"
        return self.__dict__

    def output(self, llm_response):
        """
        Returns output.  Use three inputs and 5 outputs (status, product_area, response, justificiation, request_ type)
        to create the output.csv row
        """

        return "output"

    def _query(self, input_str: str) -> list:
        if not self._chroma_model:
            return []
        return self._chroma_model.query(self.company, input_str)

    def _find_company(self) -> str | None:
        """
        If company is not defined look for context clues to identify it.  If still not
        found return blank and treat the search as global.
        """
        text = f"{self.subject or ''} {self.issue or ''}".lower()

        for company, keywords in COMPANY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return company

        return None
