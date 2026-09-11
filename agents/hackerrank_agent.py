# evaluation/hackerrank_agent.py
# +---------------------------------------------------------------------------+
# |                           HACKERRANK AGENT                                |
# +---------------------------------------------------------------------------+


from typing import List

from langchain_core.documents import Document

# Local Libraries
from agents.support_agent import SupportAgent


class HackerrankAgent(SupportAgent):
    def __init__(
        self, row_index, ticket_df, chroma_model, support_agent_model
    ):
        super().__init__(
            row_index, ticket_df, chroma_model, support_agent_model
        )

        self.title = "HackerRank Agent"
        self.company = "Hackerrank"
        self._title_agent_model(f"{self.title} Model")

    def _query(self, query_str: str) -> List[Document]:
        if not self._chroma_model:
            raise ValueError("🚨 Chroma Model needs to be defined!")

        return self._chroma_model.query(self.company, query_str)

    def _verify_grounded_response(self, draft: dict, documents: list) -> bool:
        """
        Overrides SupportAgent._verify_grounded_response() to add a
        business-rule check on top of the base citation-verification
        logic. HackerRank support tickets can ask for something no KB
        article should ever legitimately grant — e.g. "the recruiter
        rejected me, increase my score" or otherwise re-grade/bypass the
        standard evaluation process. That's not a factual-grounding
        failure for citations to catch; a response could cite a real
        document perfectly and still be wrong to send, because the
        underlying request is a policy line, not a question with a
        correct documented answer.

        Should call super()._verify_grounded_response(draft, documents)
        first for the normal citation check, then additionally scan
        `draft["response"]` for score/grading/re-evaluation promises and
        fail verification (return False) if found, regardless of what
        was cited.

        Input:
            draft (dict): output of _draft_filtered_response().
            documents (list): the filtered documents the draft was
                supposedly grounded against.

        Output:
            bool — True only if both the base citation check passes AND
            no score/grading policy violation is present in the response.
        """
        pass
