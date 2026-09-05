from __future__ import annotations

# models/support_agent_model.py
# +---------------------------------------------------------------------------+
# |                            SUPPORT AGENT MODEL                            |
# +---------------------------------------------------------------------------+
# Python Libraries
import time

# Vendor Libraries
from openai import InternalServerError, OpenAI, RateLimitError

from constants import (
    MAX_TOKENS,
    MODEL_API_KEY,
    MODEL_API_URL,
    MODEL_NAME,
    RATE_LIMIT_PAUSE_TIMER,
    RATE_LIMIT_RETRIES,
    SYSTEM_INSTR_PROMPT,
)
from utils import log_chat_transcript

# Local Libraries


class SupportAgentModel:
    """
    A class to represent a language model.
    """

    def __init__(self, row_cnt: int):
        if (
            MODEL_API_URL is None
            or MODEL_NAME is None
            or MODEL_API_KEY is None
        ):
            raise ValueError(
                "🚨 Credentials aren't properly being read. Check environment file. 🚨"
            )

        subtitles = [
            f"🤖MODEL_NAME: {MODEL_NAME}",
            f"🌐️MODEL_API_URL: {MODEL_API_URL}",
            f"📄️DATA ROWS: {row_cnt}",
        ]

        self.name = "Support Agent Model"
        # show_banner(self.name, subtitles)

        self._client = self._load_model()

    def _load_model(self) -> OpenAI:
        return OpenAI(
            base_url=MODEL_API_URL,
            api_key=MODEL_API_KEY,
            timeout=120,  # ⏱️ Kill the connection if it hangs over 120 seconds
            max_retries=RATE_LIMIT_RETRIES,  # 🔄 Automatically back off and retry 3 times natively
        )

    def get_response(self, prompt: str, row_index: int) -> dict:
        """
        Calls OpenAI model with instructions and prompt context and waits for a response.  The response is then
        filtered and returned in a specific format for output.

        https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create
        """

        attempt = 0
        while attempt < RATE_LIMIT_RETRIES:
            try:
                response = self._client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTR_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_completion_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    top_p=1.0,
                    timeout=90.0,
                )

                return self._format_response(self._filter_response(response))

            except InternalServerError as e:
                print(
                    f"🚨 Idx: {row_index} | {self.name} Server error encountered (503/5xx): {e} 🚨"
                )
                return {}  # Return safe empty list so downstream code doesn't crash on None

            except RateLimitError as e:
                print(
                    f"\n🚨 Idx: {row_index} | Rate limit / Quota exceeded (429) on attempt: {attempt} 🚨"
                )

                if attempt >= RATE_LIMIT_RETRIES - 1:
                    print(
                        f"\n🚨 Idx: {row_index} | {self.name} request has exceeded the maximum amount of retries! Returning {{error: True}}. 🚨"
                    )
                    return {"error": True}

                body = e.body[0] if isinstance(e.body, list) else e.body
                error_message = body.get("error", {}).get("message", [])
                print(f"\n🚨 {error_message} 🚨")

                # You can parse the retry delay or default to a safe pause
                delay_time = self._parse_delay_time(error_message)
                log_chat_transcript("RATE LIMIT ERROR", error_message)
                print(f"\n⏸️  Pausing for {delay_time} seconds ...")

                time.sleep(delay_time)
                attempt += 1

            except Exception as e:
                print(
                    f"\n🚨 Idx: {row_index} | {self.name} Unexpected API error occurred: {e} 🚨"
                )
                return {}

    def _parse_delay_time(self, error_message: str) -> int | float:
        # Let's attempt to use the vendor's response delay time suggestion instead of our own.
        err = error_message.lower()
        anchor_str = "please retry in "
        end_char = "s"  # Safely skips the decimal point

        if anchor_str not in err or end_char not in err:
            return RATE_LIMIT_PAUSE_TIMER

        # Anchor string has been found!
        # Cherry pick their delay time by getting the start and end string positions. Then remove the `s` for seconds and convert to a float.
        start_pos = err.find(anchor_str) + len(anchor_str)
        end_pos = err.find(end_char, start_pos)

        delay_time_str = err[start_pos:end_pos]
        delay_time = float(delay_time_str.replace("s", ""))

        # Just in case the vendor's delay time is long we will override it.
        if delay_time > (RATE_LIMIT_PAUSE_TIMER * 2):
            return RATE_LIMIT_PAUSE_TIMER

        return delay_time

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

    def _filter_response(self, response):
        return response

    def _format_response(self, response):
        # Output: issue
        return response

    def _ground_truth(self):
        return ""
