# tests/unit/models/test_support_agent_model.py
# +---------------------------------------------------------------------------+
# |                      SUPPORT AGENT MODEL TESTS                            |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/models/test_support_agent_model.py -v

# Python Libraries
import json
from unittest.mock import MagicMock

# Vendor Libraries
import httpx
import pytest
from openai import InternalServerError, RateLimitError

# Local Libraries
from models.gemini_model import GeminiUnavailableError
from models.support_agent_model import SupportAgentModel
from src.constants import RATE_LIMIT_PAUSE_TIMER
from src.enums import Status


def _status_error(cls, status_code, message="error", body=None):
    """Builds a real openai SDK error (RateLimitError / InternalServerError
    are just APIStatusError subclasses that derive status_code from an
    httpx.Response) — closer to what the client actually raises than a
    hand-rolled stand-in would be."""
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return cls(message, response=response, body=body)


def _make_response(content, finish_reason="stop"):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def model(mocker):
    mocker.patch("models.support_agent_model.OpenAI")
    return SupportAgentModel(row_count=1)


# --- get_response(): happy path + content filter -------------------------- #


def test_get_response_happy_path_returns_parsed_json(model):
    response = _make_response(json.dumps({"status": "Replied", "response": "ok"}))
    model._client.chat.completions.create.return_value = response

    result = model.get_response("some prompt", row_index=0)

    assert result == {"status": "Replied", "response": "ok"}


def test_get_response_blocks_on_content_filter(model):
    response = _make_response(content=None, finish_reason="content_filter_low")
    model._client.chat.completions.create.return_value = response

    result = model.get_response("some prompt", row_index=3)

    assert result["status"] == Status.ESCALATED
    assert "content_filter_low" in result["justification"]


# --- get_response(): server errors ----------------------------------------- #


def test_get_response_returns_empty_dict_on_5xx_server_error(model):
    model._client.chat.completions.create.side_effect = _status_error(
        InternalServerError, 500
    )

    result = model.get_response("some prompt", row_index=0)

    assert result == {}


def test_get_response_raises_gemini_unavailable_on_503(model):
    model._client.chat.completions.create.side_effect = _status_error(
        InternalServerError, 503
    )

    with pytest.raises(GeminiUnavailableError):
        model.get_response("some prompt", row_index=0)


# --- get_response(): rate limiting / retries ------------------------------- #


def test_get_response_retries_rate_limit_then_succeeds(model, mocker):
    sleep_mock = mocker.patch("models.support_agent_model.time.sleep")
    error = _status_error(
        RateLimitError,
        429,
        body={"error": {"message": "Please retry in 1.5s"}},
    )
    good_response = _make_response(json.dumps({"status": "Replied"}))
    model._client.chat.completions.create.side_effect = [error, good_response]

    result = model.get_response("some prompt", row_index=0)

    assert result == {"status": "Replied"}
    assert model._client.chat.completions.create.call_count == 2
    sleep_mock.assert_called_once_with(1.5)


def test_get_response_returns_error_dict_after_exhausting_retries(
    model, mocker
):
    sleep_mock = mocker.patch("models.support_agent_model.time.sleep")
    error = _status_error(
        RateLimitError, 429, body={"error": {"message": "no timing info"}}
    )
    model._client.chat.completions.create.side_effect = [error, error, error]

    result = model.get_response("some prompt", row_index=0)

    assert result == {"error": True}
    assert model._client.chat.completions.create.call_count == 3
    assert sleep_mock.call_count == 2


# --- get_response(): anything else ----------------------------------------- #


def test_get_response_returns_empty_dict_on_unexpected_exception(model):
    model._client.chat.completions.create.side_effect = ValueError("boom")

    result = model.get_response("some prompt", row_index=0)

    assert result == {}


# --- _mask_secret() --------------------------------------------------------- #


def test_mask_secret_masks_short_or_missing_values():
    assert SupportAgentModel._mask_secret("short") == "***"
    assert SupportAgentModel._mask_secret(None) == "***"


def test_mask_secret_shows_first_and_last_four_of_long_values():
    assert (
        SupportAgentModel._mask_secret("sk-1234567890abcdef") == "sk-1...cdef"
    )


# --- _parse_delay_time() ---------------------------------------------------- #


def test_parse_delay_time_extracts_seconds_from_message(model):
    assert model._parse_delay_time("Please retry in 4.2s after cooldown") == 4.2


def test_parse_delay_time_falls_back_when_pattern_missing(model):
    assert model._parse_delay_time("no timing info here") == RATE_LIMIT_PAUSE_TIMER


def test_parse_delay_time_falls_back_when_delay_is_too_large(model):
    message = f"Please retry in {RATE_LIMIT_PAUSE_TIMER * 3}s"
    assert model._parse_delay_time(message) == RATE_LIMIT_PAUSE_TIMER


# --- _filter_response() / _format_response() -------------------------------- #


def test_filter_response_parses_valid_json_from_choice_message(model):
    response = _make_response('{"a": 1}')
    assert model._filter_response(response) == {"a": 1}


def test_filter_response_returns_empty_dict_for_blank_content(model):
    response = _make_response("   ")
    assert model._filter_response(response) == {}


def test_filter_response_returns_empty_dict_for_malformed_json(model):
    response = _make_response("not valid json{")
    assert model._filter_response(response) == {}


def test_format_response_passes_through_truthy_dict(model):
    assert model._format_response({"a": 1}) == {"a": 1}


def test_format_response_returns_empty_dict_for_falsy_input(model):
    assert model._format_response({}) == {}
    assert model._format_response(None) == {}
