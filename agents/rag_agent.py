# agents/rag_agent.py
# +---------------------------------------------------------------------------+
# |                                RAG AGENT                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
import json

# Local Libraries
from src.constants import (
    FILTER_DOC_PROMPT,
    GROUND_RESPONSE_PROMPT,
    MAX_RELEVANCE_DISTANCE,
    MIN_SEARCH_SCORE,
    PRECISION_RESPONSE_PROMPT,
    RESP_PRECISION_THRESHOLD,
)
from src.enums import Status
from src.utils import log_chat_transcript


class RagAgent:
    def __init__(self, dataset: dict):
        self.grounded = None
        self.precise = None
        self.response = None
        self.cited_chunks = None
        self.documents = None
        self.reasoning = None

        self.status = dataset.get("status", None)
        self._issue = dataset.get("issue", None)
        self._subject = dataset.get("subject", None)

        self._model = dataset.get("model")
        self._row_index = dataset.get("row_index")
        self._verify_hook = dataset.get("verify_hook")
        self._reset()

    def _reset(self):
        self.grounded = False
        self.precise = False
        self.response = None
        self.cited_chunks = []
        self.documents = []
        self.reasoning = None

    def grounding(self, documents: list):
        filtered_documents = self._filter_by_relevance(documents)

        if not filtered_documents:
            return self.evaluate_groundness(self.__dict__)

        draft = self._draft_filtered_response(filtered_documents)

        if not draft:
            return self.evaluate_groundness(self.__dict__)

        is_verified = self._verify_grounded_response(draft, filtered_documents)

        if not is_verified:
            return self.evaluate_groundness(self.__dict__)
        self.grounded = True

        is_query_precise = self._check_precision(draft)

        if not is_query_precise:
            return self.evaluate_groundness(self.__dict__)
        self.precise = True

        self.response = draft.get("response")
        self.reasoning = draft.get("reasoning")

        return self.evaluate_groundness(self.__dict__)

    def _filter_by_relevance(self, documents: list) -> list:
        valid_documents = []

        if not documents:
            self.status = Status.ESCALATED
            return valid_documents

        for document, score in documents:
            if score <= MAX_RELEVANCE_DISTANCE:
                valid_documents.append(document)

        if not valid_documents:
            self._reset()
            self.reasoning = (
                "No retrieved documents cleared the relevance threshold."
            )

        return valid_documents

    def _draft_filtered_response(self, documents: list) -> list:
        system_prompt = FILTER_DOC_PROMPT.format(
            subject=self._subject,
            issue=self._issue,
            documents=self._format_documents_xml(documents),
        )

        response = self._model.get_response(system_prompt, self._row_index)

        log_chat_transcript(
            "🤖 RAG_AGENT",
            f"_draft_filtered_response()\n{system_prompt}\nResponse:\n{response}",
        )

        if not response:
            self._reset()
            self.documents = documents
            self.reasoning = "Failed to generate a valid structured draft."
        else:
            # If there is a draft response update cited chunks data here!
            self.cited_chunks = response.get("cited_chunks", [])

        return response

    def _verify_grounded_response(self, draft: dict, documents: list):
        system_prompt = GROUND_RESPONSE_PROMPT.format(
            draft=json.dumps(draft, indent=2),
            documents=self._format_documents_xml(documents),
        )

        grounded_response = self._model.get_response(
            system_prompt, self._row_index
        )

        log_chat_transcript(
            "🤖 RAG_AGENT",
            f"Verify Grounded Response: {grounded_response}",
        )

        is_grounded = (
            bool(grounded_response.get("is_grounded"))
            if isinstance(grounded_response, dict)
            else False
        )

        reasoning = (
            "Verification failed: Citations were missing, fabricated,"
            " or unsupported."
        )

        # Company-specific compliance/business-rule gate (e.g. HackerRank's
        # forbidden score-manipulation terms, Visa's PCI-DSS check). Only
        # runs once citations are already verified — nothing here can
        # un-fail a response that's already ungrounded.
        if is_grounded and self._verify_hook and not self._verify_hook(draft):
            is_grounded = False
            reasoning = (
                "Verification failed: Response did not pass "
                "company-specific compliance checks."
            )

        if not is_grounded:
            self.status = Status.ESCALATED
            cited_chunks = (
                self.cited_chunks.copy()
            )  # copy this first then reset
            self._reset()
            self.cited_chunks = cited_chunks
            self.documents = documents
            self.reasoning = reasoning

        return is_grounded

    def _check_precision(self, draft: dict) -> bool:
        draft_text = (
            draft.get("response", "")
            if isinstance(draft, dict)
            else str(draft)
        )

        subject_text = (
            f"## TICKET SUBJECT\n{self._subject.strip()}"
            if self._subject
            else ""
        )

        system_prompt = PRECISION_RESPONSE_PROMPT.format(
            issue=self._issue.strip(),
            subject=subject_text,
            draft=draft_text,
        )

        precision_response = self._model.get_response(
            system_prompt, self._row_index
        )

        if not precision_response:
            self.status = Status.ESCALATED
            return False

        precision_score = (
            precision_response.get("precision_score", MIN_SEARCH_SCORE)
            if isinstance(precision_response, dict)
            else MIN_SEARCH_SCORE
        )

        log_chat_transcript(
            "🤖 RAG_AGENT",
            f"Check Precision: {system_prompt}\nCheck Precision: {precision_response}",
        )

        is_query_precise = precision_score >= RESP_PRECISION_THRESHOLD

        if not is_query_precise:
            self._reset()
            self.grounded = True
            self.reasoning = (
                "Precision check failed: Response is factually grounded but"
                " off-topic for this ticket."
            )

        return is_query_precise

    def _format_documents_xml(self, documents) -> str:
        if len(documents) == 0:
            return ""

        document_xml = "<retrieved_context_documents>\n"

        for idx, document in enumerate(documents):
            chunk_idx = document.metadata.get("chunk_idx", idx)
            content = document.page_content
            document_xml += (
                f"\t<document id='{chunk_idx}'>{content}</document>"
            )

        document_xml += "\n</retrieved_context_documents>"

        return document_xml

    def evaluate_groundness(self, results: dict) -> dict:
        """
        Evaluates the grounding and precision results dictionary and updates
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

        return self.__dict__
