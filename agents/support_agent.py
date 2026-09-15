# src/support_agent.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
import json
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
    MAX_RELEVANCE_DISTANCE,
    MIN_SEARCH_SCORE,
    RESP_PRECISION_THRESHOLD,
    TICKET_ISSUE_STRLEN,
    URGENT_TERMS,
)
from src.enums import Company, RequestType, Risk, Status, Urgency
from src.utils import (
    log_chat_transcript,
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
                # A blank CSV cell comes through pandas as float('nan'),
                # not None/"" — and NaN is truthy in Python, so leaving it
                # as-is breaks every downstream `if self.subject:` /
                # `.strip()` call that assumes "truthy means it's a
                # string." Normalize once here instead of guarding every
                # call site.
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
        }
        return RagAgent(dataset)

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

    def evaluate_groundness(self, results: dict) -> None:
        """Evaluates the grounding and precision results dictionary and updates

        the agent's status, response, and justification accordingly.
        """
        print(f"evaluate_groundness()\ngrounding_results-> {results}")
        is_grounded = results.get("grounded", False)
        is_precise = results.get("precise", False)
        reasoning = results.get("reasoning", "No justification provided.")

        # Escalate if EITHER grounding OR precision fails
        if not is_grounded or not is_precise:
            self.status = Status.ESCALATED
            self.response = (
                "Your request has been escalated to a support specialist for"
                " further review."
            )
            self.justification = f"Escalated support ticket: {reasoning}."
        else:
            # Both gates passed successfully
            self.status = Status.REPLIED
            self.response = results.get("response", self.response)
            self.justification = (
                f"Answered using grounded documentation: '{reasoning}'."
            )

    def groundness(self, documents: list) -> dict:
        # Get grounding results
        results = self.rag_agent.grounding(documents)
        self.status = results.get("status")

        return results

    # @TODO - Testing Agent_Rag.grounding()
    # This works but want to refactor some more.
    def groundness2(self, documents: list) -> dict:

        print("\n\ngroundess()")
        filtered_documents = self._filter_by_relevance(documents)

        if not filtered_documents:
            return {
                "grounded": False,
                "precise": False,
                "response": None,
                "cited_chunks": [],
                "documents": [],
                "reasoning": (
                    "No retrieved documents cleared the relevance threshold."
                ),
            }

        draft = self._draft_filtered_response(filtered_documents)
        cited_chunks = draft.get("cited_chunks", []) if draft else []

        if not draft or not isinstance(draft, dict):
            return {
                "grounded": False,
                "precise": False,
                "response": None,
                "cited_chunks": [],
                "documents": filtered_documents,
                "reasoning": "Failed to generate a valid structured draft.",
            }

        is_verified = self._verify_grounded_response(draft, filtered_documents)

        if not is_verified:
            return {
                "grounded": False,
                "precise": False,
                "response": None,
                "cited_chunks": cited_chunks,
                "documents": filtered_documents,
                "reasoning": (
                    "Verification failed: Citations were missing, fabricated,"
                    " or unsupported."
                ),
            }

        is_query_precise = self._check_precision(draft)

        if not is_query_precise:
            return {
                "grounded": True,
                "precise": False,
                "response": None,
                "cited_chunks": cited_chunks,
                "documents": filtered_documents,
                "reasoning": (
                    "Precision check failed: Response is factually grounded but"
                    " off-topic for this ticket."
                ),
            }

        # All gates have passed
        self.response = draft.get("response")

        return {
            "grounded": True,
            "precise": True,
            "response": self.response,
            "cited_chunks": cited_chunks,
            "documents": filtered_documents,
            "reasoning": draft.get("reasoning"),
        }

    def _filter_by_relevance(self, documents: list) -> list:

        valid_documents = []

        if not documents:
            self.status = Status.ESCALATED
            return valid_documents

        for document, score in documents:
            if score <= MAX_RELEVANCE_DISTANCE:
                valid_documents.append(document)
        print(f"valid_documents={valid_documents}")
        return valid_documents

    def _draft_filtered_response(self, documents: list) -> dict:

        # Format document list for system prompt.
        formatted_docs = json.dumps(
            [
                {
                    "chunk_idx": doc.metadata.get("chunk_idx", idx),
                    "content": doc.page_content,
                }
                for idx, doc in enumerate(documents)
            ],
            indent=2,
        )

        system_prompt = """
        You are an AI Support Response Specialist. Your task is to draft a user-facing response to a support ticket using ONLY the provided retrieved context documents.

        ## SUPPORT TICKET
        Subject: {subject}
        Issue: {issue}

        ## RETRIEVED CONTEXT DOCUMENTS
        {documents}

        ## TASK INSTRUCTIONS:
        1. Answer the support ticket issue using ONLY facts present in the context documents above. Do not assume or extrapolate policies[span_0](start_span)[span_0](end_span).
        2. Cite the specific `chunk_idx` backing each claim in your response.
        3. If the context does not contain enough information to answer the ticket, set `grounded` to false and state what information is missing in `reasoning`.

        ## OUTPUT REQUIREMENTS:
        Return ONLY a valid JSON object wrapped inside a markdown code block (```json ... ```) matching this schema:

        ```json
        {{
        "grounded": true,
        "response": "Detailed support response grounded strictly in the documentation.",
        "cited_chunks": [0],
        "reasoning": "Concise justification for why the context is sufficient or insufficient."
        }}
        """.strip().format(
            subject=self.subject,
            issue=self.issue,
            documents=formatted_docs,
        )

        log_chat_transcript(
            "SUPPORT_AGENT", f"_draft_filtered_response()\n{system_prompt}"
        )

        # Parse response, update attributes
        draft_response = self._model.get_response(
            system_prompt, self.row_index
        )

        if not draft_response:
            log_chat_transcript(
                "SUPPORT_AGENT",
                "No response returned in _draft_filtered_response().",
            )

        return draft_response

    def _verify_grounded_response(self, draft: dict, documents: list) -> bool:

        formatted_draft = json.dumps(draft, indent=2)
        formatted_docs = json.dumps(
            [
                {
                    "chunk_idx": doc.metadata.get("chunk_idx", idx),
                    "content": doc.page_content,
                }
                for idx, doc in enumerate(documents)
            ],
            indent=2,
        )

        system_prompt = """
        You are an AI Quality Assurance Specialist evaluating RAG groundedness.
        Verify whether the proposed draft response is factually supported by the referenced context documents.

        ## DRAFT RESPONSE TO VERIFY
        {draft}

        ## REFERENCE CONTEXT DOCUMENTS
        {documents}

        ## CRITERIA FOR VERIFICATION:
        1. Verify that every `cited_chunks` index in the draft actually exists in the reference context documents.
        2. Verify that the text in the referenced chunks explicitly supports every claim made in `response`.
        3. Return `is_grounded = true` ONLY if every citation checks out and no claims are fabricated or hallucinated.
        4. Return `is_grounded = false` if any citation is missing, fabricated, or unsupported by the text.

        ## OUTPUT SPECIFICATION:
        Return ONLY a valid JSON object wrapped inside a markdown code block (```json ... ```) matching this schema:

        ```json
        {{
        "is_grounded": true,
        "reasoning": "Explanation of why citations pass or fail validation."
        }}
        """.strip().format(
            draft=formatted_draft,
            documents=formatted_docs,
        )

        log_chat_transcript(
            "🤖 SUPPORT_AGENT",
            f"Verify Grounded System Prompt: {system_prompt}",
        )

        grounded_response = self._model.get_response(
            system_prompt, self.row_index
        )
        log_chat_transcript(
            "🤖 SUPPORT_AGENT",
            f"Verify Grounded Response: {grounded_response}",
        )

        is_grounded = (
            bool(grounded_response.get("is_grounded"))
            if isinstance(grounded_response, dict)
            else False
        )

        if not is_grounded:
            self.status = Status.ESCALATED

        return is_grounded

    def _check_precision(self, draft: dict) -> bool:

        draft_text = (
            draft.get("response", "")
            if isinstance(draft, dict)
            else str(draft)
        )

        subject_text = (
            f"## TICKET SUBJECT\n{self.subject.strip()}"
            if self.subject
            else ""
        )

        system_prompt = """
        You are an AI Support Supervisor evaluating response precision.
        Determine whether the drafted response directly and accurately addresses the user's support ticket issue and subject[span_0](start_span)[span_0](end_span).

        ## TICKET ISSUE
        {issue}

        {subject}

        ## DRAFTED RESPONSE
        {draft}

        ## EVALUATION CRITERIA:
        - Focus solely on the relationship between the ticket issue/subject and the drafted response[span_1](start_span)[span_1](end_span).
        - Do NOT evaluate factual grounding (that has already been verified)[span_2](start_span)[span_2](end_span).
        - Does the response actually answer what the user asked, or does it answer an adjacent/unrelated question?[span_3](start_span)[span_3](end_span)

        ## SCORING SCALE (0.0 to 1.0):
        - 1.0: The response directly and completely answers the ticket issue[span_4](start_span)[span_4](end_span).
        - 0.0: The response is ambiguous, off-topic, or answers a different question entirely[span_5](start_span)[span_5](end_span).

        ## OUTPUT SPECIFICATION:
        Return ONLY a valid JSON object wrapped inside a markdown code block (```json ... ```) matching this schema:

        ```json
        {{
        "precision_score": 0.95,
        "reasoning": "Concise explanation of why the response is precise or imprecise for this ticket."
        }}""".strip().format(
            issue=self.issue.strip(),
            subject=subject_text,
            draft=draft_text,
        )

        precision_response = self._model.get_response(
            system_prompt, self.row_index
        )

        log_chat_transcript(
            "🤖 SUPPORT_AGENT",
            f"Check Precision: {system_prompt}\nCheck Precision: {precision_response}",
        )

        if not precision_response:
            self.status = Status.ESCALATED
            return False

        precision_score = (
            precision_response.get("precision_score", MIN_SEARCH_SCORE)
            if isinstance(precision_response, dict)
            else MIN_SEARCH_SCORE
        )

        return precision_score >= RESP_PRECISION_THRESHOLD
