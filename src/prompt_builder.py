# src/prompt_builder.py
# +---------------------------------------------------------------------------+
# |                              PROMPT BUILDER                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import xmltodict

# Local Libraries
from constants import USER_PROMPT_TEMPLATE


class PromptBuilder:
    def __init__(self, dataset: dict):

        self._ticket_dict = {}
        self.prompt = self._build(dataset) or ""

    # def _set_attrs(self, dataset: dict) -> None:
    #    for key, value in dataset.items():
    #        if hasattr(self, key):
    #            setattr(self, key, value)

    def _build(self, dataset: dict) -> str:
        for key, value in dataset.items():
            if key == "document_chunk":
                continue

            method_name = f"_{key.lstrip('_')}"
            method = getattr(self, method_name, None)

            # Added 'and' operator
            # Changed 'not value' to 'value' so it only unpacks if there is data
            if value and callable(method):
                self._ticket_dict[key] = method(**value)

        # self._ticket_dict["company"] = dataset.get("company")
        # self._ticket_dict["issue"] = self._issue(dataset.get("issue"))
        # self._ticket_dict["subject"] = self._subject(dataset.get("subject"))
        # self._ticket_dict["response"] = self._response_xml(dataset.get(""))
        # self._ticket_dict["product_area"] = self.(dataset.get("product_area"))
        # self._ticket_dict["status"] = self.(dataset.get("status"))
        # self._ticket_dict["request_type"] = self.(dataset.get("request_type"))
        # self._ticket_dict[""] = self.(dataset.get(""))
        # self._ticket_dict[""] = self.(dataset.get(""))

        ticket_data = {"ticket": self._ticket_dict}
        retrieved_context_data = {
            "retrieved_context": self._retrieved_context(
                dataset.get("document_chunks")
            )
        }

        return USER_PROMPT_TEMPLATE.format(
            support_ticket_data_xml=self._convert_to_xml(ticket_data),
            retrieved_context_xml=self._convert_to_xml(retrieved_context_data),
        ).strip()

    def _convert_to_xml(self, ticket_dict) -> str:
        return xmltodict.unparse(ticket_dict, pretty=True, full_document=False)

    def _company(self, name: str | None) -> dict:
        return {"company": name}

    def _issue(self, name: str) -> dict:
        return {
            "issue": name,
        }
        return {
            "@__": "",  # attribute: <issue attr="">
            "": "",  # nested element: <element> <issue><element>__</element>
            "#__": "",  # inner text of element: <issue>____</issue>
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
            "request_str": name,
        }

    def _retrieved_context(self, document_chunks: list) -> dict:
        """
        <retrieved_context>
            <document source="{filename}">{chunk_text}</document>
            <document source="{filename}">{chunk_text}</document>
        </retrieved_context>
        """

        return {
            # Passing a list to 'document' creates multiple <document> tags
            "document": [
                {
                    "@source": document.metadata[
                        "source"
                    ],  # '@' creates the source="..." attribute
                    "@file_order": document.metadata["file_order"],
                    "@chunk_idx": document.metadata["chunk_idx"],
                    "#text": document.page_content,  # '#text' creates the inner element text
                }
                for document in document_chunks
            ]
        }
