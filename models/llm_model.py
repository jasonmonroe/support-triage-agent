#models/llm_model.py
# +---------------------------------------------------------------------------+
# |                                MODEL                                       |
# +---------------------------------------------------------------------------+

class LlmModel:
    """
    A class to represent a language model.
    """

    def __init__(self, api_key: str, api_url: str, model_name: str):
        """
        Initializes the LlmModel instance.

        Args:
            api_key (str): The API key for the model.
            api_url (str): The API URL for the model.
            model_name (str): The name of the model.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name


    def _init_model(self):
        return

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the model based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.
        """