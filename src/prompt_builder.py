# src/prompt_builder.py
# +---------------------------------------------------------------------------+
# |                              PROMPT BUILDER                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import xmltodict

# Local Libraries
from src.constants import USER_PROMPT_TEMPLATE
from utils import log_chat_transcript


class PromptBuilder:
    def __init__(self, dataset: dict):

        self._ticket_dict = {
            "@row_index": dataset.get("row_index"),
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
            if key not in ["document_chunks", "row_index"] and value:
                print(f"key -> {key}")
                print(f"value -> {value[:1024]}")
                self._ticket_dict[key] = value
                print(f"self._ticket[{key}] has a value...")

            # print(f"key => {key}")

        # import sys

        # sys.exit(0)

        ticket_data = {"ticket": self._ticket_dict}

        # Build xml off document chunks
        # document_chunks = dataset.get("document_chunks")
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

    def _convert_to_xml(self, dict_data: dict) -> str:
        try:
            return xmltodict.unparse(
                dict_data, pretty=True, full_document=False
            )
        except Exception as e:
            log_chat_transcript("XML_CONVERSION_ERROR", e)
            return ""

    def _make_ticket_dict(self) -> dict:
        pass

    def _make_retrieved_context_dict(self) -> dict:
        pass

    def _retrieved_context(self, document_chunks: list) -> dict:
        """
        Example:
        <retrieved_context>
            <document source="{filename}">{chunk_text}</document>
            <document source="{filename}">{chunk_text}</document>
        </retrieved_context>
        """

        if len(document_chunks) == 0:
            return {}

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
