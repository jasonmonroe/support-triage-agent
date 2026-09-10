# evaluation/hackerrank_agent.py
# +---------------------------------------------------------------------------+
# |                           HACKERRANK AGENT                                |
# +---------------------------------------------------------------------------+


# Local Libraries
from agents.support_agent import SupportAgent
from src.utils import log_chat_transcript


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
        print(f"Model Title: {self._model.title}")

    def ground(self, documents: list):
        log_chat_transcript(
            "GROUNDING_DOCUMENTS", f"Grounding {len(documents)} documents."
        )
        document_content = ""
        for document in documents:
            document_content += document.page_content + "\n-----------\n"

        prompt = f"""
        Evaluate the accuracy of the proposed response against the provided internal documentation to prevent model hallucinations and ensure factual grounding.
        The context should be geared toward {self.company}.  A coding development website to test software engineering concepts.

        ## Context Documents
        {document_content.strip()}

        ## Draft Response to Evaluate
        {self.response.strip()}

        ## Evaluation Tasks:
        1. Verify if the Draft Response is fully supported by the Context Documents.
        2. If the Draft Response is accurate and complete, retain it.
        3. If the Draft Response contains factual errors, missing details, or hallucinations, rewrite it so it is strictly grounded in the Context Documents.
        4. Assess the risk level of the support issue based on its severity, security implications, or potential business impact.

        Assign one of the following risk levels: `low`, `medium`, `high`, `critical`.

        ### Required Output Format:
        Return ONLY a valid JSON object wrapped in a markdown code block (```json ... ```):
        {{
        "risk_level": "<low | medium | high | critical>",
        "response": "<final corrected or original user-facing response>"
        }}
        """.strip()

        grounded_response = self._model.get_response(prompt, self.row_index)
        print(f"ground(): response={grounded_response}")

        if not grounded_response or hasattr(grounded_response, "error"):
            return

        # Compare against ticket response
        self._risk_level = grounded_response["risk_level"] or None
        self.response = grounded_response["response"] or None

    # @TODO - defunct
    def __assess_risk(self, request: str) -> str:
        """
        Visa is about finance, financial documents
        """

        grounding_prompt = """

        """

        self._risk_level = None
        return self._risk_level
