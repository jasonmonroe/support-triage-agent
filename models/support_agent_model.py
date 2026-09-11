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
from src.constants import (
    LLAMA_MODEL,
    LLAMA_UNSAFE_CODES,
    MAX_TOKENS,
    MODEL_API_KEY,
    MODEL_API_URL,
    MODEL_NAME,
    RATE_LIMIT_PAUSE_TIMER,
    RATE_LIMIT_RETRIES,
    SYSTEM_INSTR_PROMPT,
)
from src.utils import log_chat_transcript, show_banner


class SupportAgentModel:
    """
    A class to represent a language model API interface for ticket triage.
    """

    def __init__(self, row_cnt: int = 0, log: bool = True):
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

        self.title = "Support Agent Model"
        self._log = log
        show_banner(self.title, subtitles)

        self._client = self._load_model()

    def _load_model(self) -> OpenAI:
        return OpenAI(
            base_url=MODEL_API_URL,
            api_key=MODEL_API_KEY,
            timeout=120,  # ⏱️ Kill the connection if it hangs over 120 seconds
            max_retries=0,  # 🔄 Let custom while-loop handle retry logic explicitly
        )

    def get_response(self, prompt: str, row_index: int) -> dict:
        """
        Calls OpenAI model with instructions and prompt context and waits for a response.
        """
        attempt = 0
        while attempt < RATE_LIMIT_RETRIES:
            try:
                response = self._client.chat.completions.create(
                    model=MODEL_NAME,
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

                return self._format_response(self._filter_response(response))

            except InternalServerError as e:
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
                print(f"\n🚨 {error_message} 🚨")

                delay_time = self._parse_delay_time(error_message)
                log_chat_transcript("RATE_LIMIT_ERROR", error_message)
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
            log_chat_transcript("FILTER_RESPONSE_ERROR", e)
            return {}

    def _format_response(self, response: dict) -> dict:
        if not response:
            return {}
        return response

    def _ground_truth(self):
        return ""

    def filter_input_with_llama_guard(self, user_input_str: str) -> str:
        """
        Filters user input using Llama Guard to ensure safety.
        """
        try:
            llama_response = self._client.chat.completions.create(
                messages=[{"role": "user", "content": user_input_str.strip()}],
                model=LLAMA_MODEL,
            )

            result = llama_response.choices[0].message.content.strip()

            if self._log:
                print("\nDEBUG --- LLAMA RESPONSE --- ")
                print(f"{llama_response}")
                print("\n# --- 🖊️  Open Guard result 🖊️ --- #")
                print(result)
                print("# --- 🖊️  Close Guard result 🖊️ --- #\n")

            return self._apply_guard(result)

        except Exception as e:
            print(f"❌ Error with Llama Guard: {e}")
            return ""

    def _apply_guard(self, result: str) -> str:
        if "unsafe" in result.lower():
            codes = result.lower().replace("unsafe ", "").strip().split(",")
            if any(
                code.strip().upper() in LLAMA_UNSAFE_CODES for code in codes
            ):
                return "BYPASS_SAFE"
            else:
                return "UNSAFE"
        else:
            return "SAFE"
