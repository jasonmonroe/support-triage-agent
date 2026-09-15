# tests/unit/models/conftest.py
# +---------------------------------------------------------------------------+
# |                 SHARED TEST FIXTURES FOR models/                          |
# +---------------------------------------------------------------------------+

# Vendor Libraries
import pytest


@pytest.fixture(autouse=True)
def patched_provider_constants(monkeypatch):
    """Every GeminiModel subclass reads provider config from these
    module-level names at __init__ time. Pin them to known test values so
    model tests don't depend on whatever happens to be in the real .env."""
    monkeypatch.setattr("models.gemini_model.MODEL_NAME", "test-model")
    monkeypatch.setattr(
        "models.gemini_model.MODEL_API_URL", "https://example.test/v1"
    )
    monkeypatch.setattr(
        "models.gemini_model.MODEL_API_KEY", "test-api-key-1234"
    )
    monkeypatch.setattr(
        "models.gemini_model.MODEL_EMBEDDING", "test-embedding-model"
    )
