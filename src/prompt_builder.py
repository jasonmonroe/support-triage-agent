# src/prompt_builder.py
# +---------------------------------------------------------------------------+
# |                              PROMPT BUILDER                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import xmltodict

# Local Libraries
from src.constants import USER_PROMPT_TEMPLATE
from src.utils import log_chat_transcript


class PromptBuilder:
    def __init__(self, dataset: dict):

        self._ticket_dict = {
            "@id": dataset.get("row_index"),
        }
        self.prompt = self._build(dataset) or ""

    def _build(self, dataset: dict) -> str:
        # Build ticket data to convert to XML
        ticket_data = self._get_ticket_data(dataset)

        # Build retreived context data to convert to XML
        retrieved_context_data = self.get_retrieved_context_data(
            dataset.get("document_chunks", [])
        )

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
            log_chat_transcript("PROMPT_BUILDER", f"Empty Data: {e}")

            # Return an empty string if dict_data is empty or invalid.
            return ""

    def _get_ticket_data(self, dataset: dict) -> dict:
        ticket_dict = {
            "@id": dataset.get("row_index", None),
        }

        for key, value in dataset.items():
            if key not in ["document_chunks", "row_index"] and value:
                ticket_dict[key] = value

        return {"ticket": ticket_dict}

    def get_retrieved_context_data(self, document_chunks: list) -> dict:
        """
        Example:
        <retrieved_context>
            <document source="{filename}">{chunk_text}</document>
            <document source="{filename}">{chunk_text}</document>
        </retrieved_context>
        """

        if len(document_chunks) == 0:
            return {}

        # Passing a list to 'document' creates multiple <document> tags
        documents_dict = {
            "document": [
                {
                    "@source": document.metadata["source"],
                    # '@' creates the source="..." attribute
                    "@file_order": document.metadata["file_order"],
                    "@chunk_idx": document.metadata["chunk_idx"],
                    "#text": document.page_content,
                    # '#text' creates the inner element text
                }
                for document, _score in document_chunks
            ]
        }

        return {"retrieved_context": documents_dict}
