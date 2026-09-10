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
    A class to represent a language model.
    """

    def __init__(self, row_cnt: int = 0):
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
        show_banner(self.title, subtitles)

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
                return {}  # Return safe empty list so downstream code doesn't crash on None

            except RateLimitError as e:
                print(
                    f"\n🚨 Idx: {row_index} | Rate limit / Quota exceeded (429) on attempt: {attempt} 🚨"
                )

                if attempt >= RATE_LIMIT_RETRIES - 1:
                    print(
                        f"\n🚨 Idx: {row_index} | {self.title} request has exceeded the maximum amount of retries! Returning {{error: True}}. 🚨"
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
                    f"\n🚨 Idx: {row_index} | {self.title} Unexpected API error occurred: {e} 🚨"
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

    def _filter_response(self, response):

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

            # self._apply_guard(response)
            # print(f"cleaned_str={cleaned_str}")

            return json.loads(cleaned_str.strip())

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            print(f"🚨 Error occurred while parsing response: {e} 🚨")
            print(f" `repr(content_str)` was: {repr(content_str)}")
            return {}

        except Exception as e:
            log_chat_transcript("FILTER_RESPONSE_ERROR", e)
            return {}

    def _format_response(self, response):
        # Output: issue
        if not response:
            return {}

        return response

    def _ground_truth(self):
        return ""

    def filter_input_with_llama_guard(self, user_input_str: str) -> str:
        """
        Function to filter user input with Llama Guard

        Filters user input using Llama Guard to ensure it is safe.
        Whitelist "UNSAFE" codes: S6, S7, S8, S13 so that you can handle the customer query.

        Parameters:
        - user_input: The input provided by the user.
        - model: The Llama Guard model to be used for filtering (default is "meta-llama/llama-guard-4-12b").

        Returns:
        - The filtered and safe input.
        """
        # @TODO - do I need LLAMA MODEL to filter or can google do it?
        try:
            # Create a request to Llama Guard to filter the user input
            llama_response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": user_input_str.strip()}],
                model=LLAMA_MODEL,
            )

            # Return the filtered input
            result = llama_response.choices[0].message.content.strip()

            if self._log:
                print("\nDEBUG --- LLAMA RESPONSE --- ")
                print(f"{llama_response}")
                print("\n# --- 🖊️  Open Guard result 🖊️ --- #")
                print(result)
                print("# --- 🖊️  Close Guard result 🖊️ --- #\n")
                print("DEBUG --- LLAMA RESPONSE ---\n")

            return self._apply_guard(result)

        except Exception as e:
            print(f"❌ Error with Llama Guard: {e}")
            return ""

    def _apply_guard(self, result: str) -> str:
        # Added type hint for clarity
        if "unsafe" in result:
            if any(
                code.strip() in LLAMA_UNSAFE_CODES
                for code in result.replace("unsafe ", "").strip().split(",")
            ):
                return "BYPASS_SAFE"
            else:
                return "UNSAFE"
        else:
            return "SAFE"
