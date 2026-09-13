# models/gemini_model.py
# +---------------------------------------------------------------------------+
# |                               GEMINI MODEL                                |
# +---------------------------------------------------------------------------+

# Python Libraries
from abc import ABC, abstractmethod

# Vendor Libraries
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Local Libraries
from src.constants import (
    MODEL_API_KEY,
    MODEL_API_URL,
    MODEL_EMBEDDING,
    MODEL_NAME,
)


class GeminiUnavailableError(RuntimeError):
    """Raised when the configured model is temporarily unavailable."""


class GeminiModel(ABC):
    def __init__(self) -> None:
        """
        Although this project is using Google's Gemini Model you can safely use
        any models.
        """

        self.name = MODEL_NAME
        self.api_url = MODEL_API_URL
        self.api_key = MODEL_API_KEY
        self.embedding_model = MODEL_EMBEDDING

        self._client = None

    @abstractmethod
    def _load_client(self):
        pass

    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{self.embedding_model}",
            google_api_key=self.api_key,
        )

    def _safety_filters(self) -> dict:
        # https://ai.google.dev/gemini-api/docs/safety-settings
        return {
            "extra_body": {
                "google": {
                    "safety_settings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_LOW_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_LOW_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_LOW_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_LOW_AND_ABOVE",
                        },
                    ],
                }
            }
        }
