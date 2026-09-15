# tests/unit/conftest.py
# +---------------------------------------------------------------------------+
# |             SHARED TEST FIXTURES FOR THE WHOLE tests/unit/ TREE           |
# +---------------------------------------------------------------------------+

# Vendor Libraries
import pandas as pd
import pytest


def _make_ticket(
    index: int = 0,
    issue: str = "",
    subject: str = "",
    company: str = "None",
    response: str = "",
    product_area: str = "",
    status: str = "",
    request_type: str = "",
):
    """Builds an itertuples() row shaped exactly like what DataHandler hands
    to TicketAnalyzer/SupportAgent in production — column names already
    underscored (DataHandler._clean_data() guarantees this before itertuples()
    ever runs), so field access like `row.Product_Area` works the same way
    here as it does in the real pipeline."""

    data = {
        "Issue": issue,
        "Subject": subject,
        "Company": company,
        "Response": response,
        "Product_Area": product_area,
        "Status": status,
        "Request_Type": request_type,
    }
    df = pd.DataFrame([data], index=[index])
    return next(df.itertuples())


class Doc:
    """Minimal stand-in for a langchain Document. Carries the metadata keys
    both RagAgent (_format_documents_xml) and PromptBuilder
    (get_retrieved_context_data) read off retrieved chunks."""

    def __init__(
        self, content="doc content", chunk_idx=0, source="kb.md", file_order=1
    ):
        self.page_content = content
        self.metadata = {
            "chunk_idx": chunk_idx,
            "source": source,
            "file_order": file_order,
        }


class FakeModel:
    """Stub SupportAgentModel. Responses are keyed by a marker substring that
    must appear in the prompt, so a test can control each step of the
    draft -> verify -> precision -> final-analysis chain independently
    without needing to match the full prompt text."""

    def __init__(self, responses: dict | None = None):
        self.title = "fake"
        self._responses = responses or {}
        self.calls: list[str] = []

    def get_response(self, prompt: str, row_index: int) -> dict:
        self.calls.append(prompt)
        for marker, response in self._responses.items():
            if marker in prompt:
                return response
        return {}


class FakeChroma:
    """Stub ChromaModel. Records every query so tests can assert on the
    company filter an agent's _query() actually passed through."""

    def __init__(self, documents=None):
        self._documents = (
            documents if documents is not None else [(Doc(), 0.2)]
        )
        self.queries: list[dict] = []

    def query(self, query_str: str, company: str | None = None) -> list:
        self.queries.append({"query_str": query_str, "company": company})
        return self._documents


@pytest.fixture
def ticket():
    return _make_ticket


@pytest.fixture
def doc():
    return Doc


@pytest.fixture
def fake_model():
    return FakeModel


@pytest.fixture
def fake_chroma():
    return FakeChroma
