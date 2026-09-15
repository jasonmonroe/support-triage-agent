# tests/unit/src/test_prompt_builder.py
# +---------------------------------------------------------------------------+
# |                        PROMPT BUILDER TESTS                               |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/src/test_prompt_builder.py -v

# Local Libraries
from src.prompt_builder import PromptBuilder


class _Doc:
    def __init__(self, source, file_order, chunk_idx, content):
        self.metadata = {
            "source": source,
            "file_order": file_order,
            "chunk_idx": chunk_idx,
        }
        self.page_content = content


def test_get_ticket_data_excludes_internal_keys_and_falsy_values():
    builder = PromptBuilder(
        {"row_index": 3, "document_chunks": [], "issue": "some issue", "response": ""}
    )

    ticket_data = builder._get_ticket_data(
        {"row_index": 3, "document_chunks": [], "issue": "some issue", "response": ""}
    )

    ticket = ticket_data["ticket"]
    assert ticket["@id"] == 3
    assert ticket["issue"] == "some issue"
    assert "document_chunks" not in ticket
    assert "row_index" not in ticket
    assert "response" not in ticket  # falsy -> excluded


def test_get_retrieved_context_data_returns_empty_dict_for_no_chunks():
    builder = PromptBuilder({"row_index": 0, "document_chunks": []})

    assert builder.get_retrieved_context_data([]) == {}


def test_get_retrieved_context_data_builds_document_list():
    builder = PromptBuilder({"row_index": 0, "document_chunks": []})
    chunks = [
        (_Doc("kb1.md", 1, 0, "First chunk text"), 0.2),
        (_Doc("kb2.md", 2, 3, "Second chunk text"), 0.4),
    ]

    result = builder.get_retrieved_context_data(chunks)

    documents = result["retrieved_context"]["document"]
    assert len(documents) == 2
    assert documents[0]["@source"] == "kb1.md"
    assert documents[0]["@file_order"] == 1
    assert documents[0]["@chunk_idx"] == 0
    assert documents[0]["#text"] == "First chunk text"
    assert documents[1]["@chunk_idx"] == 3


def test_convert_to_xml_renders_a_valid_dict():
    builder = PromptBuilder({"row_index": 0, "document_chunks": []})

    xml = builder._convert_to_xml({"ticket": {"@id": 0, "issue": "x"}})

    assert "<ticket" in xml
    assert "<issue>x</issue>" in xml


def test_convert_to_xml_returns_empty_string_on_failure():
    builder = PromptBuilder({"row_index": 0, "document_chunks": []})

    # xmltodict.unparse() requires a mapping; a list has no .items() to
    # walk, so this exercises the except branch instead of the happy path.
    assert builder._convert_to_xml(["not", "a", "dict"]) == ""


def test_build_embeds_ticket_and_context_xml_in_final_prompt():
    chunks = [(_Doc("kb1.md", 1, 0, "Reset your password here."), 0.2)]
    dataset = {
        "row_index": 7,
        "document_chunks": chunks,
        "issue": "I forgot my password",
        "subject": "Password help",
    }

    builder = PromptBuilder(dataset)

    assert "I forgot my password" in builder.prompt
    assert "Reset your password here." in builder.prompt
    assert "TASK INSTRUCTIONS" in builder.prompt


def test_build_handles_no_retrieved_documents():
    dataset = {
        "row_index": 1,
        "document_chunks": [],
        "issue": "Some issue with nothing retrieved",
    }

    builder = PromptBuilder(dataset)

    assert "Some issue with nothing retrieved" in builder.prompt
    assert isinstance(builder.prompt, str)
