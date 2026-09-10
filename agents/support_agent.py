# src/support_agent.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries

from abc import ABC

# Vendor Libraries
import pandas as pd

from src.constants import CRITICAL_RISK_TERMS, HIGH_RISK_TERMS, URGENT_TERMS
from src.enums import Company, RequestType, Risk, Status, Urgency
from src.utils import match_company_by_keywords, row_to_dict


class SupportAgent(ABC):
    """
    A class to handle support agent operations.
    """

    def __init__(
        self,
        row_index: int,
        ticket_df: pd.DataFrame,
        chroma_model,
        support_agent_model,
    ):
        """
        Initialize the SupportAgent class.
        """
        self.title = "Support Agent"
        self.row_index = row_index
        self._chroma_model = chroma_model
        self._model = support_agent_model
        self._risk_level = None  # low, high, critical
        self._urgency = None  # normal, high

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

        self._set_attrs(ticket_df)

        self.company = self._get_company(self.company)

    def _set_attrs(self, row) -> None:
        for column, value in row_to_dict(row).items():
            key = column.title().replace(" ", "_").lower()
            print(f"key = {key}")
            if hasattr(self, key):
                setattr(self, key, value)

    def _get_company(self, company: str | None) -> str | None:
        if not company or company.strip().lower() == Company.NONE.lower():
            return self._find_company()
        return company.strip().lower()

    # @TODO - defunct
    @staticmethod
    def resolve_company(row) -> str | None:
        """
        Determine a ticket's company straight from its raw row data,
        without constructing a full agent instance. Lets the caller pick
        the right SupportAgent subclass up front instead of building a
        throwaway base instance just to inspect `.company`.
        """
        data = row_to_dict(row)
        fields = {}
        for column, value in data.items():
            key = column.title().replace(" ", "_").lower()
            fields[key] = value

        company = fields.get("company")
        if company and str(company).strip().lower() != Company.NONE.lower():
            return str(company).strip().lower()

        text = f"{fields.get('subject') or ''} {fields.get('issue') or ''}"
        return match_company_by_keywords(text)

    # @TODO - defunct
    @staticmethod
    def normalized_columns(row) -> list[str]:
        """
        Field names for a ticket row, normalized the same way _set_attrs
        maps them to agent attributes — lets export() filter against real
        attribute names instead of raw, mixed-case CSV column names.
        """
        return [
            column.title().replace(" ", "_").lower()
            for column in row_to_dict(row).keys()
        ]

    def _title_agent_model(self, title: str) -> None:
        self._model.title = title

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
            self.request_type = RequestType.INVALID

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
        return Risk.HIGH if is_urgent else Urgency.NORMAL

    def _assess_risk(self, request: str) -> str:
        """
        Assess the risk of the request.
        """
        text = (request or "").lower()

        if any(term in text for term in CRITICAL_RISK_TERMS):
            return Risk.CRITICAL

        if any(term in text for term in HIGH_RISK_TERMS):
            return Risk.HIGH

        return Risk.LOW

    def make_decision(self, request: str) -> str:
        """
        Make a decision based on the request: reply or escalate, based purely
        on the risk level from classify() (pre-retrieval).
        """
        is_risky = self._risk_level in (Risk.HIGH, Risk.CRITICAL)
        return Status.ESCALATED if is_risky else Status.REPLIED

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

        if self.status == Status.ESCALATED:
            self.justification = (
                f"Escalated: ticket matched '{self._risk_level}' risk signals."
            )
            return

        if not documents:
            self.status = Status.ESCALATED
            self.justification = (
                f"{Status.ESCALATED}: no knowledge base match found to ground a"
                " response."
            )
            return

        self.product_area = documents[0].metadata.get("product_area")

    def export(self, ticket_columns: list) -> dict:

        class_dict = self.__dict__

        export_dict = {}
        for key, value in class_dict.items():
            if key in ticket_columns:
                export_dict[key] = value

        return export_dict

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
        text = f"{self.subject or ''} {self.issue or ''}"
        return match_company_by_keywords(text)
