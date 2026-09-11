# src/support_agent.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries

from abc import ABC

# Vendor Libraries
import pandas as pd

from models.chroma_model import ChromaModel
from models.support_agent_model import SupportAgentModel
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
        chroma_model: ChromaModel,
        support_agent_model: SupportAgentModel,
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
        """
        Run similarity search (optionally hybrid with keyword matching)
        against your knowledge base for the ticket's issue/subject text. Pull
        back more candidates than you'll actually use (e.g., top 10) so the
        filtering step in #2 has something to filter.
        """

        if not self._chroma_model:
            return []
        return self._chroma_model.query_all(input_str)

    def _find_company(self) -> str | None:
        """
        If company is not defined look for context clues to identify it.  If still not
        found return blank and treat the search as global.
        """
        text = f"{self.subject or ''} {self.issue or ''}"
        return match_company_by_keywords(text)

    def groundness(self, documents: list) -> dict:
        """
        Orchestrator for the grounding pipeline. Runs a fixed sequence of
        steps against the retrieved documents and stops at the first one
        that fails, rather than looping to force a passing result:

            1. _filter_by_relevance(documents)
               Score each document against this ticket and drop anything
               below the relevance threshold. If nothing survives, stop
               here: return {"grounded": False, "precise": False,
               "response": None, "documents": [], "reasoning": "..."} —
               there's nothing left to draft from, so no LLM call is
               spent trying.

            2. _draft_filtered_response(filtered_documents)
               Build the citation-tagged, documents-only prompt from
               step 1's survivors and call the model for a draft
               response + cited chunk_idx list + its own sufficiency
               opinion.

            3. _verify_grounded_response(draft, filtered_documents)
               Independently check the draft's citations against the
               real chunk text — don't take the model's own "grounded"
               claim on faith. If this returns False, stop here: return
               {"grounded": False, ...} with the reasoning set to why
               verification failed (missing/fabricated citation).

            4. _check_precision(draft)
               Separately check whether the draft actually answers *this*
               ticket, not just whether it's factually traceable to a
               source — a response can pass step 3 while still answering
               the wrong part of the ticket (grounded facts about test
               settings for a ticket that asked about billing). If this
               returns False, stop here too: return {"grounded": True,
               "precise": False, ...} so ground() can tell this apart
               from a groundedness failure in its justification text.

            5. All four steps passed: return {"grounded": True, "precise":
               True, "response": draft's response, "cited_chunks": [...],
               "documents": filtered_documents, "reasoning": "..."}.

        groundness() itself never talks to the model or the vector
        store directly — every side effect happens inside the four steps
        above, which is what makes them independently overridable.
        Company-specific subclasses should override individual steps
        (_filter_by_relevance / _draft_filtered_response /
        _verify_grounded_response / _check_precision) and call
        super().<step>() to extend the shared behavior, rather than
        overriding groundness() itself and reimplementing the sequence.

        This method is called from ground() *after* the risk and
        empty-document hard gates already in ground() have passed — a
        ticket already escalating on risk keywords or on zero retrieved
        documents never reaches here, so no LLM call is spent grounding a
        response nobody will read. There is deliberately no retry/refine
        loop within this sequence either: any failed gate escalates
        immediately in a single pass. At this project's per-ticket batch
        scale, looping to force a response over a score threshold spends
        API calls/quota a straightforward escalation doesn't, and
        escalation is already a correct, safe, designed outcome here —
        unlike an interactive agent with no human-handoff option, where
        looping to eventually produce *some* answer is the only path
        forward.

        Input:
            documents (list): raw retrieved chunks from
                retrieve_relevant_documents() — already company-filtered
                via Chroma's metadata filter, but not yet scored for
                actual relevance to this specific ticket.

        Output:
            dict — the vetted grounding result:
                {
                    "grounded": bool,          # step 1-3 all passed?
                    "precise": bool,           # step 4 passed?
                    "response": str | None,    # citation-backed draft,
                                               # or None if any gate failed
                    "cited_chunks": list[int], # chunk_idx values used
                    "documents": list,         # step 1's survivors — hand
                                               # this to the FINAL prompt's
                                               # <retrieved_context>, not
                                               # the raw unfiltered
                                               # retrieval output
                    "reasoning": str,          # which step failed and why,
                                               # or why it succeeded
                }
            ground() reads "grounded" and "precise" together to decide
            status/justification (escalate on either being False), and
            uses "documents"/"response" to know what's trustworthy enough
            to carry into the final USER_PROMPT_TEMPLATE analysis pass.
        """
        pass

    def _filter_by_relevance(self, documents: list) -> list:
        """
        Step 2 of the grounding pipeline — the first hard gate. Scores
        each retrieved chunk's actual relevance to this ticket (e.g. via
        Chroma's own similarity/distance score, or a re-ranker) and drops
        anything below a threshold. Similarity search always returns
        *something* for the requested top-k, even when nothing in the
        knowledge base truly covers the ticket — this is the step that
        catches that case rather than treating "it came back from search"
        as proof of relevance.

        Input:
            documents (list): raw documents from retrieve_relevant_documents().

        Output:
            list — the subset of `documents` that cleared the relevance
            threshold. Can be empty; an empty list is groundness()'s
            signal that nothing in the knowledge base covers this ticket,
            and the pipeline should stop here rather than proceed to
            drafting a response.
        """

        # If empty, Escalate
        pass

    def _draft_filtered_response(self, documents: list) -> dict:
        """
        Steps 3-5 of the grounding pipeline, combined into one drafting
        call: renders the relevance-filtered documents with their source
        attribution (source/chunk_idx — see _retrieved_context() in
        prompt_builder.py for the existing tagging pattern), builds a
        constrained documents-only prompt instructing the model to answer
        strictly from what's shown and cite which chunk_idx backs each
        claim, then calls the support-agent model and parses its
        structured response. The model must have an explicit way to
        report insufficient context instead of only ever choosing between
        "answer confidently" and "answer confidently but wrong."

        Input:
            documents (list): the filtered documents from
                _filter_by_relevance() — only chunks already confirmed
                relevant, never raw unfiltered retrieval output.

        Output:
            dict — the model's raw structured draft, e.g.:
                {
                    "grounded": bool,
                    "response": str,
                    "cited_chunks": list[int],
                    "reasoning": str,
                }
            Not yet trusted — _verify_grounded_response() checks this
            against the real source chunks before anything here is
            treated as fact.
        """
        return {}

    def _verify_grounded_response(self, draft: dict, documents: list) -> bool:
        """
        Step 6 of the grounding pipeline — the second hard gate. Checks
        the model's own claim of being "grounded" against reality: do the
        cited chunk_idx values actually exist among `documents`, and does
        each cited chunk's real text actually support the claim made? A
        model that already hallucinated a fact can't be reliably trusted
        to catch itself doing so in the same breath, so this is a
        separate check, not a re-ask of the same model.

        Input:
            draft (dict): output of _draft_filtered_response().
            documents (list): the same filtered documents sent to
                _draft_filtered_response(), used as the source of truth
                to verify citations against.

        Output:
            bool — True if every citation checks out and the draft can be
            trusted; False if any citation is missing, fabricated, or
            unsupported by the text it claims to come from. On False,
            groundness() should treat this the same as "grounded": False
            — hard fallback to escalation, never a partially-trusted
            response.
        """

        # If False, Escalate

        return False

    def _check_precision(self, draft: dict) -> bool:
        """
        Third gate of the grounding pipeline, separate from
        _verify_grounded_response(). Groundedness asks "is every claim in
        this response backed by a real cited chunk"; precision asks a
        different question entirely — "does this response actually
        address what THIS ticket asked." A response can pass groundedness
        perfectly while failing precision, e.g. citing real, accurate
        documentation about test expiration when the ticket actually
        asked about billing — every claim traceable to a real source,
        but answering the wrong question.

        Should compare `draft["response"]` against `self.issue` /
        `self.subject` (not against the retrieved documents — that's
        groundedness's job) and judge whether the response is on-topic
        and actually responsive to the ticket, not just factually
        accurate about something adjacent.

        This is a single check, not a refine-and-retry loop: on failure,
        groundness() should escalate rather than attempt to rewrite the
        response and re-check. At this project's per-ticket batch scale,
        looping to force a response over a precision threshold spends
        API calls a straightforward escalation doesn't, and — unlike an
        interactive agent with no human handoff — escalation is already
        a correct, safe, designed outcome here.

        Input:
            draft (dict): output of _draft_filtered_response(), the same
                draft _verify_grounded_response() checked.

        Output:
            bool — True if the response is precise/on-topic for this
            ticket; False if it drifts off-topic or answers a different
            question than the one asked. On False, groundness() should
            escalate, the same as a groundedness failure.
        """
        # If False, Escalate

        return False

    def _is_company(self, company: str) -> bool:
        return any(company == c.value for c in Company)
