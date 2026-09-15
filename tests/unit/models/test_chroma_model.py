# tests/unit/models/test_chroma_model.py
# +---------------------------------------------------------------------------+
# |                         CHROMA MODEL TESTS                                |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/models/test_chroma_model.py -v
#
# chromadb.PersistentClient, Chroma, and HuggingFaceEmbeddings are all
# mocked out — constructing the real ones would spin up an actual on-disk
# Chroma store and load real embedding-model weights, which is far too
# heavy for a unit test.

# Python Libraries
import os

# Vendor Libraries
import pytest

# Local Libraries
from models.chroma_model import ChromaModel
from src.constants import CHROMA_COLL_NAME, CHROMA_DB_DIR


@pytest.fixture
def mocks(mocker):
    return {
        "persistent_client_cls": mocker.patch(
            "models.chroma_model.chromadb.PersistentClient"
        ),
        "chroma_cls": mocker.patch("models.chroma_model.Chroma"),
        "hf_embeddings_cls": mocker.patch(
            "models.chroma_model.HuggingFaceEmbeddings"
        ),
    }


def test_init_disables_chroma_telemetry(mocks):
    ChromaModel()

    assert os.environ["CHROMA_SERVER_NO_TELEMETRY"] == "true"


def test_load_client_points_persistent_client_at_chroma_db_dir(mocks):
    ChromaModel()

    _, kwargs = mocks["persistent_client_cls"].call_args
    assert kwargs["path"] == os.path.abspath(CHROMA_DB_DIR)
    assert kwargs["settings"].allow_reset is True


def test_get_vector_storage_raises_without_embedding_model(mocks, monkeypatch):
    monkeypatch.setattr("models.gemini_model.MODEL_EMBEDDING", None)

    with pytest.raises(ValueError):
        ChromaModel()


def test_get_vector_storage_wraps_persistent_client(mocks):
    ChromaModel()

    _, kwargs = mocks["chroma_cls"].call_args
    assert kwargs["collection_name"] == CHROMA_COLL_NAME
    assert kwargs["client"] == mocks["persistent_client_cls"].return_value


def test_get_hf_embeddings_uses_configured_model(mocks):
    ChromaModel()

    _, kwargs = mocks["hf_embeddings_cls"].call_args
    assert kwargs["model_name"] == "test-embedding-model"
    assert kwargs["encode_kwargs"]["normalize_embeddings"] is True


def test_get_collection_count_returns_underlying_count(mocks):
    model = ChromaModel()
    model.vector_storage._collection.count.return_value = 42

    assert model.get_collection_count() == 42


def test_get_collection_count_returns_zero_on_error(mocks):
    model = ChromaModel()
    model.vector_storage._collection.count.side_effect = RuntimeError("boom")

    assert model.get_collection_count() == 0


def test_query_without_company_has_no_filter(mocks):
    model = ChromaModel()
    model.vector_storage.similarity_search_with_score.return_value = [
        "result"
    ]

    result = model.query("some query")

    _, kwargs = model.vector_storage.similarity_search_with_score.call_args
    assert kwargs["query"] == "some query"
    assert "filter" not in kwargs
    assert result == ["result"]


def test_query_with_company_adds_lowercase_filter(mocks):
    model = ChromaModel()

    model.query("some query", company="HackerRank")

    _, kwargs = model.vector_storage.similarity_search_with_score.call_args
    assert kwargs["filter"] == {"company": "hackerrank"}


def test_query_with_company_none_has_no_filter(mocks):
    model = ChromaModel()

    model.query("some query", company="None")

    _, kwargs = model.vector_storage.similarity_search_with_score.call_args
    assert "filter" not in kwargs


def test_add_vector_documents_batches_by_batch_size(mocks):
    model = ChromaModel()
    model.vector_storage.add_documents.side_effect = lambda chunks: [
        f"id-{i}" for i in range(len(chunks))
    ]

    result = model.add_vector_documents(["c1", "c2", "c3"], batch_size=2)

    assert model.vector_storage.add_documents.call_count == 2
    assert result is True


def test_add_vector_documents_returns_false_when_ids_are_missing(mocks):
    model = ChromaModel()
    # Fewer ids come back than chunks were sent in -- a partial failure.
    model.vector_storage.add_documents.return_value = ["id-1"]

    result = model.add_vector_documents(["c1", "c2", "c3"], batch_size=2)

    assert result is False


def test_delete_resets_the_underlying_client(mocks):
    model = ChromaModel()

    model.delete()

    model._client.reset.assert_called_once()


def test_reload_rebuilds_client_and_vector_storage(mocks):
    model = ChromaModel()

    model.reload()

    assert mocks["persistent_client_cls"].call_count == 2
    assert mocks["chroma_cls"].call_count == 2
