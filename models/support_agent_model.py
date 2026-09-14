from __future__ import annotations

# models/support_agent_model.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT MODEL                            |
# +---------------------------------------------------------------------------+
#
# Python Libraries
import json
import time

# Vendor Libraries
from openai import InternalServerError, OpenAI, RateLimitError

# Local Libraries
from models.gemini_model import GeminiModel, GeminiUnavailableError
from src.constants import (
    MAX_TOKENS,
    RATE_LIMIT_PAUSE_TIMER,
    RATE_LIMIT_RETRIES,
    SYSTEM_INSTR_PROMPT,
)
from src.enums import Status
from src.utils import log_chat_transcript, show_banner


class SupportAgentModel(GeminiModel):
    """
    A class to represent a language model API interface for ticket triage.
    """

    def __init__(self, row_count: int = 0):
        super().__init__()

        subtitles = []
        attr_dict = self.__dict__
        for key, value in attr_dict.items():
            if "_" not in key and "_key" in key and value is None:
                raise ValueError(
                    (
                        f"🚨 {key} Credentials aren't properly being read.",
                        " Check environment file. 🚨",
                    )
                )

            display_value = (
                self._mask_secret(value) if key == "api_key" else value
            )
            subtitles.append(
                f"{key.title().replace('_', ' ')}: {display_value}"
            )

        self.title = "Support Agent Model"
        show_banner(self.title, subtitles)

        self._client = self._load_client()

    @staticmethod
    def _mask_secret(value: str | None) -> str:
        if not value or len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    def _load_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=120,  # ⏱️ Kill the connection if it hangs over 120 seconds
            max_retries=0,  # 🔄 Let custom while-loop handle retry logic explicitly
        )

    def get_response(self, prompt: str, row_index: int) -> dict:
        """
        Calls OpenAI model with instructions and prompt context and waits for
        a response.
        """
        attempt = 0
        while attempt < RATE_LIMIT_RETRIES:
            try:
                # Check if it's safe first

                response = self._client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_INSTR_PROMPT.format(
                                agent_title=self.title
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_completion_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    top_p=1.0,
                    timeout=90.0,
                )

                choice = response.choices[0] if response.choices else None
                if choice and (choice.finish_reason or "").startswith(
                    "content_filter"
                ):
                    print(
                        f"\n🚨 Idx: {row_index} | {self.title} blocked by"
                        f" content safety filter: {choice.finish_reason} 🚨"
                    )
                    log_chat_transcript(
                        "SUPPORT_AGENT_MODEL",
                        f"Content filter block ({choice.finish_reason}) at"
                        f" index {row_index}.",
                    )
                    return {
                        "status": Status.ESCALATED,
                        "justification": (
                            f"Escalated: response blocked by content safety"
                            f" filter ({choice.finish_reason})."
                        ),
                    }

                return self._format_response(self._filter_response(response))

            except InternalServerError as e:
                status_code = getattr(e, "status_code", None)

                if status_code == 503:
                    raise GeminiUnavailableError(
                        f"Gemini model {self.name!r} is unavailable due to high demand."
                    ) from e

                print(
                    f"🚨 Idx: {row_index} | {self.title} Server error encountered (503/5xx): {e} 🚨"
                )
                return {}  # Safe empty dict for downstream processing

            except RateLimitError as e:
                print(
                    f"\n🚨 Idx: {row_index} | Rate limit / Quota exceeded (429) on attempt: {attempt} 🚨"
                )

                if attempt >= RATE_LIMIT_RETRIES - 1:
                    print(
                        f"\n🚨 Idx: {row_index} | {self.title} request has exceeded the maximum retries! Returning {{'error': True}}. 🚨"
                    )
                    return {"error": True}

                body = (
                    e.body[0]
                    if isinstance(e.body, list)
                    else getattr(e, "body", {})
                )
                error_message = (
                    str(body.get("error", {}).get("message", ""))
                    if isinstance(body, dict)
                    else str(e)
                )

                delay_time = self._parse_delay_time(error_message)
                log_chat_transcript(
                    "SUPPORT_AGENT_MODEL", f"Rate Limit Error: {error_message}"
                )
                print(f"\n⏸️  Pausing for {delay_time} seconds ...")

                time.sleep(delay_time)
                attempt += 1

            except Exception as e:
                print(
                    f"\n🚨 Idx: {row_index} | {self.title} Unexpected API error occurred: {e} 🚨"
                )
                return {}

        return {}

    def _parse_delay_time(self, error_message: str) -> int | float:
        err = error_message.lower()
        anchor_str = "please retry in "
        end_char = "s"

        if anchor_str not in err or end_char not in err:
            return RATE_LIMIT_PAUSE_TIMER

        try:
            start_pos = err.find(anchor_str) + len(anchor_str)
            end_pos = err.find(end_char, start_pos)
            delay_time_str = err[start_pos:end_pos]
            delay_time = float(delay_time_str.replace("s", ""))

            if delay_time > (RATE_LIMIT_PAUSE_TIMER * 2):
                return RATE_LIMIT_PAUSE_TIMER

            return delay_time
        except ValueError:
            return RATE_LIMIT_PAUSE_TIMER

    def _filter_response(self, response) -> dict:
        content_str = ""
        try:
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                content_str = choice.message.content or ""
            elif hasattr(response, "content"):
                content_str = response.content or ""
            elif isinstance(response, str):
                content_str = response
            else:
                content_str = str(response)

            if not content_str or not content_str.strip():
                return {}

            cleaned_str = content_str.strip()
            return json.loads(cleaned_str)

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            print(f"🚨 Error occurred while parsing response: {e} 🚨")
            print(f" `repr(content_str)` was: {repr(content_str)}")
            return {}

        except Exception as e:
            log_chat_transcript(
                "SUPPORT_AGENT_MODEL", f"Filter Response Error: {e}"
            )
            return {}

    def _format_response(self, response: dict) -> dict:
        if not response:
            return {}
        return response
