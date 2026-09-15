# tests/unit/models/test_gemini_model.py
# +---------------------------------------------------------------------------+
# |                          GEMINI MODEL TESTS                               |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/models/test_gemini_model.py -v

# Vendor Libraries
import pytest

# Local Libraries
from models.gemini_model import GeminiModel, GeminiUnavailableError


class _ConcreteGeminiModel(GeminiModel):
    """GeminiModel is abstract (_load_client has no implementation) — this
    minimal subclass exercises the shared behavior it defines without
    depending on a real chat/embedding provider."""

    def _load_client(self):
        return "fake-client"


def test_cannot_instantiate_abstract_base_directly():
    with pytest.raises(TypeError):
        GeminiModel()


def test_init_reads_provider_config_from_constants():
    model = _ConcreteGeminiModel()

    assert model.name == "test-model"
    assert model.api_url == "https://example.test/v1"
    assert model.api_key == "test-api-key-1234"
    assert model.embedding_model == "test-embedding-model"
    assert model._client is None


def test_get_embeddings_uses_configured_model_and_key(mocker):
    mock_embeddings_cls = mocker.patch(
        "models.gemini_model.GoogleGenerativeAIEmbeddings"
    )
    model = _ConcreteGeminiModel()

    model._get_embeddings()

    mock_embeddings_cls.assert_called_once_with(
        model="models/test-embedding-model",
        google_api_key="test-api-key-1234",
    )


def test_safety_filters_covers_all_four_harm_categories():
    model = _ConcreteGeminiModel()

    filters = model._safety_filters()
    settings = filters["extra_body"]["google"]["safety_settings"]
    categories = {entry["category"] for entry in settings}

    assert categories == {
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    assert all(
        entry["threshold"] == "BLOCK_LOW_AND_ABOVE" for entry in settings
    )


def test_gemini_unavailable_error_is_a_runtime_error():
    assert issubclass(GeminiUnavailableError, RuntimeError)
