# tests/unit/src/test_utils.py
# +---------------------------------------------------------------------------+
# |                             UTILS TESTS                                   |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/src/test_utils.py -v
#
# Pure display/banner helpers (banner, show_banner, _create_title_banner,
# etc.) are intentionally not covered here — they're print formatting with
# no logic worth protecting, same reasoning as skipping constants.py/enums.py.

# Python Libraries
from enum import Enum

# Vendor Libraries
import pandas as pd
import pytest

# Local Libraries
from src.utils import (
    format_bytes,
    format_iso_date,
    gen_run_id,
    get_progress_bar,
    get_time,
    log_chat_transcript,
    match_company_by_keywords,
    prettify_cols,
    pretty_dict,
    row_to_dict,
    sum_bytes_in_dir,
)


# --- gen_run_id() -------------------------------------------------------------- #


def test_gen_run_id_is_five_uppercase_hex_chars():
    run_id = gen_run_id()

    assert len(run_id) == 5
    assert run_id == run_id.upper()
    int(run_id, 16)  # raises ValueError if not valid hex


def test_gen_run_id_is_not_the_same_every_call():
    assert gen_run_id() != gen_run_id()


# --- get_time() ------------------------------------------------------------------ #


def test_get_time_formats_minutes_seconds_and_milliseconds():
    result = get_time(0.0, end_time_float=65.25)

    assert result == "1m 5s 250ms"


def test_get_time_uses_now_when_end_time_omitted():
    result = get_time(__import__("time").time())

    assert result.startswith("0m 0s")


# --- row_to_dict() / prettify_cols() ---------------------------------------------- #


def test_row_to_dict_converts_itertuples_row():
    df = pd.DataFrame([{"Issue": "a", "Product_Area": "billing"}])
    row = next(df.itertuples())

    result = row_to_dict(row)

    assert result["Issue"] == "a"
    assert result["Product_Area"] == "billing"


def test_row_to_dict_passes_through_plain_mapping():
    assert row_to_dict({"a": 1}) == {"a": 1}


def test_prettify_cols_normalizes_column_names():
    df = pd.DataFrame([{"Issue": "a", "Product_Area": "billing"}])
    row = next(df.itertuples())

    result = prettify_cols(row)

    assert "product_area" in result
    assert "issue" in result


# --- match_company_by_keywords() -------------------------------------------------- #


def test_match_company_by_keywords_finds_hackerrank():
    result = match_company_by_keywords("My HackerRank test invite expired")

    assert result == "hackerrank"


def test_match_company_by_keywords_finds_visa():
    result = match_company_by_keywords("My Visa card payment was declined")

    assert result == "visa"


def test_match_company_by_keywords_returns_none_when_no_match():
    result = match_company_by_keywords("What is the capital of France?")

    assert result is None


# --- sum_bytes_in_dir() / format_bytes() ------------------------------------------ #


def test_sum_bytes_in_dir_sums_regular_files(tmp_path):
    (tmp_path / "a.md").write_text("12345")  # 5 bytes
    (tmp_path / "b.md").write_text("1234567890")  # 10 bytes
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.md").write_text("123")  # 3 bytes

    assert sum_bytes_in_dir(str(tmp_path)) == 18


def test_sum_bytes_in_dir_skips_index_md(tmp_path):
    (tmp_path / "a.md").write_text("12345")  # 5 bytes
    (tmp_path / "index.md").write_text("this should not be counted")

    assert sum_bytes_in_dir(str(tmp_path)) == 5


def test_sum_bytes_in_dir_returns_zero_for_missing_dir():
    assert sum_bytes_in_dir("/no/such/directory/exists") == 0


def test_format_bytes_examples():
    assert format_bytes(0) == "0B"
    assert format_bytes(1024) == "1KB"
    assert format_bytes(1_572_864) == "1.5MB"
    assert format_bytes(5_368_709_120) == "5GB"


def test_format_bytes_rejects_negative_values():
    with pytest.raises(ValueError):
        format_bytes(-1)


# --- format_iso_date() ------------------------------------------------------------ #


def test_format_iso_date_parses_iso_8601_with_z_suffix():
    result = format_iso_date("2026-04-15T01:46:20Z")

    assert result.year == 2026
    assert result.month == 4
    assert result.hour == 1
    assert result.tzinfo is not None


def test_format_iso_date_parses_human_readable_format():
    result = format_iso_date("Apr 15, 2026, 01:46 PM")

    assert result.year == 2026
    assert result.hour == 13
    assert result.minute == 46


def test_format_iso_date_raises_on_empty_string():
    with pytest.raises(ValueError):
        format_iso_date("")

    with pytest.raises(ValueError):
        format_iso_date("   ")


# --- pretty_dict() ----------------------------------------------------------------- #


def test_pretty_dict_serializes_enum_values():
    class Color(Enum):
        RED = "red"

    result = pretty_dict({"color": Color.RED})

    assert '"red"' in result


def test_pretty_dict_is_indented_json():
    result = pretty_dict({"a": 1})

    assert "{\n" in result
    assert '"a": 1' in result


# --- get_progress_bar() ------------------------------------------------------------- #


def test_get_progress_bar_shows_correct_percentage():
    result = get_progress_bar(0, 4)

    assert result.endswith("25.0%")


def test_get_progress_bar_marks_only_completed_steps():
    result = get_progress_bar(1, 3)
    graphic = result.split("\t")[0]

    assert graphic.count("✅️") == 2
    assert graphic.count("☑️") == 1


def test_get_progress_bar_full_at_last_index():
    result = get_progress_bar(2, 3)

    assert result.endswith("100.0%")
    assert "☑️" not in result.split("\t")[0]


# --- log_chat_transcript() ----------------------------------------------------------- #


def test_log_chat_transcript_appends_to_the_log_file(tmp_path, monkeypatch):
    log_path = tmp_path / "log.txt"
    monkeypatch.setattr("src.utils.CHAT_TRANSCRIPT_FILE", str(log_path))

    log_chat_transcript("STAGE_ONE", "first entry")
    log_chat_transcript("STAGE_TWO", "second entry")

    content = log_path.read_text()
    assert "STAGE_ONE" in content
    assert "first entry" in content
    assert "STAGE_TWO" in content
    assert "second entry" in content
    # Both entries present in order -> confirms append, not overwrite.
    assert content.index("first entry") < content.index("second entry")
