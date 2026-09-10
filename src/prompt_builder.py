# src/prompt_builder.py
# +---------------------------------------------------------------------------+
# |                              PROMPT BUILDER                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import xmltodict

# Local Libraries
from src.constants import USER_PROMPT_TEMPLATE


class PromptBuilder:
    def __init__(self, dataset: dict):

        self._ticket_dict = {
            "@index": dataset.get("row_index"),
        }
        self.prompt = self._build(dataset) or ""

    def _build(self, dataset: dict) -> str:
        print("_build()")
        """
        for key, value in dataset.items():
            if key == "document_chunk":
                continue

            method_name = f"_{key.lstrip('_')}"
            method = getattr(self, method_name, None)
            print(f"method_name={method_name}")
            # Added 'and' operator
            # Changed 'not value' to 'value' so it only unpacks if there is data
            if value and callable(method):
                self._ticket_dict.update(method(value))
        """

        for key, value in dataset.items():
            if key not in ["document_chunk", "index"] and value:
                self._ticket_dict[key] = {f"{key}", value}

            # print(f"key => {key}")

        # import sys

        # sys.exit(0)

        ticket_data = {"ticket": self._ticket_dict}

        # Build xml off document chunks
        retrieved_context_data = {
            "retrieved_context": self._retrieved_context(
                dataset.get("document_chunks")
            )
        }

        return USER_PROMPT_TEMPLATE.format(
            support_ticket_data_xml=self._convert_to_xml(ticket_data),
            retrieved_context_data_xml=self._convert_to_xml(
                retrieved_context_data
            ),
        ).strip()

    def _convert_to_xml(self, ticket_data) -> str:
        return xmltodict.unparse(ticket_data, pretty=True, full_document=False)

    """
    def _company(self, name: str | None) -> dict:
        return {"company": name}

    def _issue(self, name: str) -> dict:
        return {
            "issue": name,
        }

    def _subject(self, name: str) -> dict:
        return {
            "subject": name,
        }

    def _response(self, name: str) -> dict:
        return {
            "response": name,
        }

    def _product_area(self, name: str) -> dict:
        return {
            "product_area": name,
        }

    def _status(self, name: str) -> dict:
        return {
            "status": name,
        }

    def _request_type(self, name: str) -> dict:
        return {
            "request_type": name,
        }
    """

    def _retrieved_context(self, document_chunks: list) -> dict:
        """
        Example:
        <retrieved_context>
            <document source="{filename}">{chunk_text}</document>
            <document source="{filename}">{chunk_text}</document>
        </retrieved_context>
        """

        return {
            # Passing a list to 'document' creates multiple <document> tags
            "document": [
                {
                    "@source": document.metadata["source"],
                    # '@' creates the source="..." attribute
                    "@file_order": document.metadata["file_order"],
                    "@chunk_idx": document.metadata["chunk_idx"],
                    "#text": document.page_content,
                    # '#text' creates the inner element text
                }
                for document in document_chunks
            ]
        }
