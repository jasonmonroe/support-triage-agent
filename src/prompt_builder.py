# src/prompt_builder.py
# +---------------------------------------------------------------------------+
# |                              PROMPT BUILDER                               |
# +---------------------------------------------------------------------------+


from constants import USER_PROMPT_TEMPLATE


class PromptBuilder:
    def __init__(self, dataset: dict):

        self.prompt = ""

        self._set_attrs(dataset)

        self.prompt = self._build()

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _build(self) -> str:

        return USER_PROMPT_TEMPLATE.format()
