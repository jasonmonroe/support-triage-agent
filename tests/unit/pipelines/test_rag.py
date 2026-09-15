# tests/unit/pipelines/test_rag.py
# +---------------------------------------------------------------------------+
# |                           RAG PIPELINE TESTS                              |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/pipelines/test_rag.py -v

# Vendor Libraries
import pytest

# Local Libraries
from pipelines.rag import _ingest, _verify, run_rag_pipeline
from src.enums import RagStatus


@pytest.fixture(autouse=True)
def quiet_log(mocker):
    """Orchestration is what these tests are about, not the log file — mock
    the logger so tests don't depend on (or write to) the real log.txt."""
    return mocker.patch("pipelines.rag.log_chat_transcript")


@pytest.fixture
def doc_handle_cls(mocker):
    """A mocked DocumentHandler with count_documents() pinned to a real int
    — run_rag_pipeline()'s post-ingest re-verify always calls this, and an
    unconfigured MagicMock return value would break _verify()'s int
    comparisons."""
    cls = mocker.patch("pipelines.rag.DocumentHandler")
    cls.return_value.count_documents.return_value = 0
    return cls


# --- _verify() --------------------------------------------------------------- #


def test_verify_fails_when_collection_is_empty():
    counts = {"collection_count": 0, "document_count": 5, "chunk_count": None}
    assert _verify(counts) == RagStatus.FAIL


def test_verify_pre_ingestion_success_when_collection_exceeds_documents():
    counts = {"collection_count": 10, "document_count": 5, "chunk_count": None}
    assert _verify(counts) == RagStatus.SUCCESS


def test_verify_pre_ingestion_partial_when_collection_at_or_below_documents():
    counts = {"collection_count": 3, "document_count": 5, "chunk_count": None}
    assert _verify(counts) == RagStatus.PARTIAL


def test_verify_post_ingestion_success_when_collection_matches_chunks():
    counts = {"collection_count": 100, "document_count": 5, "chunk_count": 100}
    assert _verify(counts) == RagStatus.SUCCESS


def test_verify_post_ingestion_partial_when_collection_under_chunks():
    counts = {"collection_count": 40, "document_count": 5, "chunk_count": 100}
    assert _verify(counts) == RagStatus.PARTIAL


def test_verify_returns_none_when_collection_exceeds_chunks_unexpectedly():
    """Characterization test: once chunk_count is known, _verify() has no
    branch for collection_count > chunk_count — it falls through and
    returns None instead of SUCCESS or PARTIAL."""
    counts = {"collection_count": 150, "document_count": 5, "chunk_count": 100}
    assert _verify(counts) is None


# --- _ingest() ----------------------------------------------------------------- #


def test_ingest_returns_success_when_all_chunks_are_collected(mocker):
    mocker.patch("pipelines.rag.sum_bytes_in_dir", return_value=0)
    chroma_model = mocker.MagicMock()
    chroma_model.add_vector_documents.return_value = True
    chroma_model.get_collection_count.return_value = 3

    doc_handle = mocker.MagicMock()
    doc_handle.process.return_value = ["chunk1", "chunk2", "chunk3"]
    doc_handle.count_documents.return_value = 5
    doc_handle.count_chunks.return_value = 3

    result = _ingest(chroma_model, doc_handle)

    chroma_model.add_vector_documents.assert_called_once_with(
        ["chunk1", "chunk2", "chunk3"]
    )
    doc_handle.show.assert_called_once()
    assert result == RagStatus.SUCCESS


def test_ingest_returns_partial_when_fewer_chunks_collected_than_produced(
    mocker,
):
    mocker.patch("pipelines.rag.sum_bytes_in_dir", return_value=0)
    chroma_model = mocker.MagicMock()
    chroma_model.get_collection_count.return_value = 2  # only 2 of 3 landed

    doc_handle = mocker.MagicMock()
    doc_handle.process.return_value = ["chunk1", "chunk2", "chunk3"]
    doc_handle.count_documents.return_value = 5
    doc_handle.count_chunks.return_value = 3

    result = _ingest(chroma_model, doc_handle)

    assert result == RagStatus.PARTIAL


# --- run_rag_pipeline() --------------------------------------------------------- #


def test_refresh_flag_wipes_and_reingests(mocker, doc_handle_cls):
    mock_ingest = mocker.patch("pipelines.rag._ingest")
    chroma_model = mocker.MagicMock()
    chroma_model.get_collection_count.return_value = 10

    run_rag_pipeline({"refresh": True}, {"md_files": {}}, chroma_model)

    chroma_model.delete.assert_called_once()
    chroma_model.reload.assert_called_once()
    mock_ingest.assert_called_once()


def test_no_refresh_and_already_successful_skips_prompting(
    mocker, doc_handle_cls
):
    mock_input = mocker.patch("builtins.input")
    chroma_model = mocker.MagicMock()
    chroma_model.get_collection_count.return_value = 10  # > document_count(0)

    run_rag_pipeline({"refresh": False}, {"md_files": {}}, chroma_model)

    mock_input.assert_not_called()


def test_declining_ingestion_skips_it(mocker, doc_handle_cls):
    mocker.patch("builtins.input", return_value="N")
    mock_ingest = mocker.patch("pipelines.rag._ingest")
    chroma_model = mocker.MagicMock()
    chroma_model.get_collection_count.return_value = 0  # FAIL -> prompts

    run_rag_pipeline({"refresh": False}, {"md_files": {}}, chroma_model)

    mock_ingest.assert_not_called()


def test_accepting_ingestion_calls_ingest_and_stops_once_successful(
    mocker, doc_handle_cls
):
    mocker.patch("builtins.input", return_value="Y")
    mock_ingest = mocker.patch(
        "pipelines.rag._ingest", return_value=RagStatus.SUCCESS
    )
    chroma_model = mocker.MagicMock()
    chroma_model.get_collection_count.return_value = 0  # FAIL -> prompts

    run_rag_pipeline({"refresh": False}, {"md_files": {}}, chroma_model)

    mock_ingest.assert_called_once()


def test_max_retries_exhausted_stops_prompting(mocker, doc_handle_cls):
    mock_input = mocker.patch("builtins.input", return_value="X")  # not Y/N
    mocker.patch("pipelines.rag._ingest")
    chroma_model = mocker.MagicMock()
    chroma_model.get_collection_count.return_value = 0  # FAIL -> prompts

    run_rag_pipeline({"refresh": False}, {"md_files": {}}, chroma_model)

    # INGEST_LIMIT_RETRIES=3, itr starts at 1: loop runs at itr=1 and itr=2
    # (2 prompts) before itr reaches 3 and the loop condition stops it.
    assert mock_input.call_count == 2
