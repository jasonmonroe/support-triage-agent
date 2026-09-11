# evaluation/visa_agent.py
# +---------------------------------------------------------------------------+
# |                              VISA AGENT                                   |
# +---------------------------------------------------------------------------+

from typing import List

from langchain_core.documents import Document

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

    def _query(self, query_str: str) -> List[Document]:
        if not self._chroma_model:
            raise ValueError("Chroma Model needs to be defined!")

        return self._chroma_model.query(self.company, query_str)

    def _assess_risk(self, request: str) -> str:
        """
        Overrides SupportAgent._assess_risk(). The base class's global
        CRITICAL_RISK_TERMS/HIGH_RISK_TERMS lists include "fraud" and
        "unauthorized" — for Claude or HackerRank those words almost
        always mean "escalate, no KB article could cover this." For Visa
        they show up in tickets the KB is specifically designed to
        answer (e.g. "my card was stolen, what do I do" has a fully
        documented process with phone numbers). This override should
        distinguish "reporting a known incident via a documented
        process" (answerable, not automatically high-risk) from
        "disputing/contesting a specific charge's legitimacy" or
        requesting a policy exception (still needs escalation) — likely
        by checking for dispute-specific language rather than treating
        every fraud/unauthorized mention the same way the base class
        does.

        Input:
            request (str): the ticket's issue text (same input the base
                class's version takes).

        Output:
            str — one of the Risk enum values (low/high/critical), same
            contract as SupportAgent._assess_risk().
        """
        pass

    def _verify_grounded_response(self, draft: dict, documents: list) -> bool:
        """
        Overrides SupportAgent._verify_grounded_response() with stricter
        numeric-matching, since Visa tickets routinely need to convey
        phone numbers, dollar amounts, and timeframes — and a
        hallucinated phone number in a fraud-reporting context is the
        single most dangerous failure mode across all three companies:
        it sends someone to a wrong (or malicious) number during a
        financial emergency.

        Should call super()._verify_grounded_response(draft, documents)
        for the baseline citation check, then additionally extract any
        phone numbers/dollar amounts/timeframes from `draft["response"]`
        and confirm each one appears verbatim in the cited chunk(s) —
        not just a loose keyword/topic overlap.

        Input:
            draft (dict): output of _draft_filtered_response().
            documents (list): the filtered documents the draft was
                supposedly grounded against.

        Output:
            bool — True only if the base citation check passes AND every
            number-like claim in the response exactly matches text in
            its cited source.
        """
        pass
