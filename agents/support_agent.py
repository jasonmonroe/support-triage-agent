# src/support_agent.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
from abc import ABC

# Vendor Libraries
import pandas as pd

# Local Libraries
from agents.rag_agent import RagAgent
from models.chroma_model import ChromaModel
from models.support_agent_model import SupportAgentModel
from src.constants import (
    CRITICAL_RISK_TERMS,
    HIGH_RISK_TERMS,
    TICKET_ISSUE_STRLEN,
    URGENT_TERMS,
)
from src.enums import Company, RequestType, Risk, Status, Urgency
from src.utils import (
    match_company_by_keywords,
    row_to_dict,
)


class SupportAgent(ABC):
    """
    A class to handle support agent operations.
    """

    def __init__(
        self,
        row_index: int,
        ticket_df: pd.DataFrame,
        chroma_model: ChromaModel,
        support_agent_model: SupportAgentModel,
    ):
        """
        Initialize the SupportAgent class.
        """
        self.title = "🤖 Support Agent"
        self.row_index = row_index
        self._chroma_model = chroma_model
        self._model = support_agent_model
        self._risk_level = None
        self._urgency = None

        self.issue = None
        self.subject = None
        self.company = None
        self.response = None
        self.product_area = None
        self.status = None
        self.request_type = None
        self.justification = None

        self._set_attrs(ticket_df)

        self.company = self._get_company(self.company)
        self.rag_agent = self._get_rag_agent()

    def _set_attrs(self, row) -> None:
        for column, value in row_to_dict(row).items():
            key = column.title().replace(" ", "_").lower()
            if hasattr(self, key):
                """
                A blank CSV cell comes through pandas as float('nan'), not
                None/"" — and NaN is truthy in Python, so leaving it as-is
                breaks every downstream `if self.subject:` / `.strip()` call
                that assumes "truthy means it's a string." Normalize once here
                instead of guarding every call site.
                """

                if pd.isna(value):
                    value = None
                setattr(self, key, value)

    def _get_rag_agent(self):
        dataset = {
            "model": self._model,
            "row_index": self.row_index,
            "subject": self.subject,
            "issue": self.issue,
            "status": self.status,
            "verify_hook": self._get_verify_hook(),
        }
        return RagAgent(dataset)

    def _get_verify_hook(self):
        """Hook for subclasses to plug a company-specific compliance check
        into RagAgent's verification gate. None means no extra check."""
        return None

    def _get_company(self, company: str | None) -> str | None:
        if not company or company.strip().lower() == Company.NONE.lower():
            return self._find_company()
        return company.strip().lower()

    def _set_agent_model_title(self, title: str) -> None:
        self._model.title = title

    def classify_issue(self) -> None:
        """
        Classify and assess request type, product area, risk.
        request_type/product_area are still finalized by the LLM's structured
        output later (real classification, not keyword matching) — this pass
        only extracts cheap, deterministic risk/urgency signals so escalation
        doesn't depend solely on the model's judgment.
        """

        # If no issue or it's too short invalidate the request type...
        if not self.issue or len(self.issue) <= TICKET_ISSUE_STRLEN:
            self.request_type = RequestType.INVALID

        self._risk_level = self._assess_risk(self.issue)
        self._urgency = self._assess_urgency(self.issue)

    def _assess_urgency(self, issue: str) -> str:
        """
        Assess the urgency of the request.
        """
        text = (issue or "").lower()
        is_urgent = any(term in text for term in URGENT_TERMS)

        return Risk.HIGH if is_urgent else Urgency.NORMAL

    def _assess_risk(self, issue: str) -> str:
        """
        Assess the risk of the request.
        """
        text = (issue or "").lower()

        if any(term in text for term in CRITICAL_RISK_TERMS):
            return Risk.CRITICAL

        if any(term in text for term in HIGH_RISK_TERMS):
            return Risk.HIGH

        return Risk.LOW

    def make_decision(self) -> str:
        """
        Make a decision based on the request: reply or escalate, based purely
        on the risk level from classify() (pre-retrieval).

        Only CRITICAL risk hard-escalates pre-retrieval. HIGH_RISK_TERMS
        (e.g. "delete my account", "refund") are often legitimate, documented
        self-service flows — auto-escalating those skips retrieval entirely and
        can block a perfectly answerable FAQ (confirmed: "delete my account" has
        a complete, on-point KB article, but the old HIGH-inclusive gate never
        let retrieval run to find it). Let HIGH risk flow through to retrieval +
        groundness like any other ticket instead.
        """

        if self._risk_level == Risk.CRITICAL:
            self.status = Status.ESCALATED
            self.justification = f"{self.status}: Ticket matched '{self._risk_level}' risk signals."

        else:
            self.status = Status.REPLIED

            self.justification = (
                f"{self.status}: No knowledge base match found to ground a"
                " response."
            )

        # If no product area, lets flag it...
        if not self.product_area:
            self.justification += " Product Area is also unknown at this time."
            print("🚩 Product Area is undefined!")

    def retrieve_relevant_documents(self) -> list:
        """
        Get the relevant knowledge from the request via the /data/ directory.
        This is the RAG retrieval process.

        """
        subject = f"Subject: {self.subject}\n" if self.subject else ""
        query = f"{subject}Issue: {self.issue}".strip()

        return self._query(query)

    def export(self, ticket_columns: list) -> dict:
        class_dict = self.__dict__
        export_dict = {}
        for key, value in class_dict.items():
            if key in ticket_columns or key == "justification":
                export_dict[key] = value

        return export_dict

    def _query(self, input_str: str) -> list:
        """
        Run similarity search (optionally hybrid with keyword matching)
        against your knowledge base for the ticket's issue/subject text. Pull
        back more candidates than you'll actually use (e.g., top 10) so the
        filtering step in #2 has something to filter.
        """

        if not self._chroma_model:
            raise ValueError("🚨 Chroma Model needs to be defined!")

        return self._chroma_model.query(input_str)

    def _find_company(self) -> str | None:
        """
        If company is not defined look for context clues to identify it.  If
        still not found return blank and treat the search as global.
        """
        text = f"{self.subject or ''} {self.issue or ''}"
        return match_company_by_keywords(text)

    def groundness(self, documents: list) -> dict:
        # Get grounding results
        results = self.rag_agent.grounding(documents)
        self.status = results.get("status")
        self.response = results.get("response", self.response)
        self.justification = results.get("justification", self.justification)

        return results
