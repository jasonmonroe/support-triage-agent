# models/support_agent_model.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT MODEL                            |
# +---------------------------------------------------------------------------+

# Python Libraries

# Vendor Libraries
from google import genai

from constants import MODEL_NAME

# Local Libraries


class SupportAgentModel:
    """
    A class to represent a language model.
    """

    def __init__(self, api_key: str, api_url: str, model_name: str):
        """
        Initializes the SupportAgentModel instance.

        Args:
            api_key (str): The API key for the model.
            api_url (str): The API URL for the model.
            model_name (str): The name of the model.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

        self._client = self._init_model()

    def _init_model(self):
        return genai.Client()

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the model based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.
        """

        response = self._client.models.generate_content(
            model=MODEL_NAME, contents=prompt
        )

        return self._format_response(self._filter_response(response))

    def _filter_response(response):
        return response

    def _format_response(response):
        # Output: issue
        return response

    def _ground_truth(self):
        return ""
