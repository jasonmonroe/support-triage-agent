# src/utils.py
# +---------------------------------------------------------------------------+
# |                                UTILITIES                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
import json
import textwrap
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

# Local Libraries
from src.constants import (
    CHAT_TRANSCRIPT_FILE,
    COMPANY_KEYWORDS,
    MSEC,
    PEP8_LINE_LEN,
    SECS_IN_MIN,
)


def gen_run_id() -> str:
    """Generates a unique ID for the current run."""
    return uuid.uuid4().hex[:5].upper()


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(
    start_time_float: float, end_time_float: float | None = None
) -> str:

    if end_time_float is None:
        end_time_float = time.time()

    diff = abs(end_time_float - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN * SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"⏱ Run Time: {get_time(start_time_int)}")


def banner(inspect) -> None:
    m = inspect.f_code.co_name.title().replace("_", " ").upper()
    show_banner(f"🏃 {m}")


def _make_top_btm_line() -> str:
    open_close_len = 2  # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    return "+" + ("-" * max_line_len) + "+"


def _create_title_banner(text: str, center_text: bool = True) -> None:
    open_close_len = 4  # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    # Trim off any chars after limit plus two spaces for blanks
    text = text[0 : max_line_len - open_close_len]
    text_len = len(text)
    padding_len = max_line_len - text_len

    if center_text:
        # If uneven padding add an extra length for the right side
        extra_len = 0 if padding_len % 2 == 0 else 1

        padding_len = padding_len // 2
        title_line = (
            "| "
            + (" " * padding_len)
            + text
            + (" " * (padding_len + extra_len))
            + " |"
        )

    else:
        # Remove last two characters to account for open/close spacing
        title_line = "| " + text + (" " * padding_len) + " |"

    top_btm_line = _make_top_btm_line()

    # Print title banner
    print("\n")
    print(top_btm_line)
    print(title_line)
    print(top_btm_line)


def _create_subtitle_banner(
    text: str | list, center_text: bool = False
) -> None:
    # Reconstructs the guard to safely catch wrong types OR empty values
    if not isinstance(text, (str, list)) or not text:
        return None

    open_close_len = 4  # open close of char `+` or `|` plus space
    max_line_len = PEP8_LINE_LEN - open_close_len
    wrapped_lines = _get_wrapped_lines(text, max_line_len)

    # Now that the data is a list format it for display.
    for line in wrapped_lines:
        # Clean up any rogue newline markers so they don"t break string length math
        line = line.replace("\n", " ").strip()
        line_len = len(line)
        padding_len = max_line_len - line_len

        if center_text:
            extra_len = 0 if padding_len % 2 == 0 else 1
            padding_len = padding_len // 2
            padded_line = (
                "| "
                + (" " * padding_len)
                + line
                + (" " * (padding_len + extra_len))
                + " |"
            )
        else:
            padded_line = "| " + line + (" " * padding_len) + " |"

        print(padded_line)

    # Close the subtitle
    if len(wrapped_lines) > 0:
        print(_make_top_btm_line())

    return None


def _get_wrapped_lines(text: str | list, max_line_len: int) -> list:
    wrapped_lines = []

    if isinstance(text, list):
        # Explicitly wrap each individual item inside the list
        for item in text:
            if isinstance(item, str):
                wrapped_lines.extend(textwrap.wrap(item, width=max_line_len))
            else:
                wrapped_lines.append(str(item))

    elif isinstance(text, str):
        wrapped_lines = textwrap.wrap(text, width=max_line_len)

    return wrapped_lines


def show_banner(
    title: str,
    subtitle: str | list | None = "",
    center_title_text: bool = True,
    center_subtitle_text: bool = False,
) -> None:
    _create_title_banner(title, center_title_text)

    if subtitle:
        _create_subtitle_banner(subtitle, center_subtitle_text)


def log_chat_transcript(stage: str, content: str) -> None:
    """
    Appends a formatted execution step or model interaction directly
    to the required evaluation log path.

    :param stage:
    :param content:
    :param filepath:
    :return:
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "-" * PEP8_LINE_LEN

    log_entry = (
        f"\n{divider}\n[{timestamp}] - STAGE: {stage}\n{divider}\n{content}\n"
    )

    print(log_entry[:1024])

    # Ensure the file appends cleanly
    with open(CHAT_TRANSCRIPT_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)


def get_progress_bar2(idx: int, total: int, batch_size: int = 0) -> str:
    """
    Displays progress of claim analysis.

    :param idx: Current index (0-indexed)
    :param total: Total number of items or batches
    :param batch_size: Size of each batch (optional)
    :return: Formatted progress bar string with percentage
    """
    # Use proper string literals instead of URL-encoded strings
    i_empty, i_full = "☑️ ", "✅ "

    # Calculate percentage based on the 0-indexed current position
    completion_pct = ((idx + 1) / total) * 100

    if batch_size == 0:
        # Number of completed and remaining steps
        completed = idx + 1
        remaining = max(0, total - completed)
        graphic = (i_full * completed) + (i_empty * remaining)
    else:
        # Calculate total steps and current progress in batch increments
        total_steps = (total + batch_size - 1) // batch_size
        completed_steps = (idx + 1) // batch_size
        remaining_steps = max(0, total_steps - completed_steps)
        graphic = (i_full * completed_steps) + (i_empty * remaining_steps)

    return f"{graphic}\t{completion_pct:.1f}%"


def get_progress_bar(idx: int, total: int) -> str:
    """
    Displays progress of claim analysis.

    :param idx:
    :param total:
    :return:
    """
    print("")

    i_empty, i_full = "☑️ ", "✅️ "
    completion_pct = ((idx + 1) / total) * 100

    graphic = ""
    for i in range(0, total):
        graphic += i_full if i <= idx else i_empty

    return graphic + f"\t{completion_pct:.1f}%"


def format_iso_date(date_str: str) -> datetime:
    """
    Parse either an ISO 8601 string (e.g. 2026-04-15T01:46:20Z)
    or a human-readable string (e.g. Apr 15, 2026, 01:46 PM)
    and return a timezone-aware datetime in UTC.
    """
    if not date_str or not date_str.strip():
        raise ValueError("Date string is empty.")

    date_str = date_str.strip()

    # Try ISO 8601 first (handles 2026-04-15T01:46:20Z)
    try:
        normalized = date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)

    except ValueError:
        # Fallback to human-readable format
        dt = datetime.strptime(date_str, "%b %d, %Y, %I:%M %p")

    # Ensure UTC-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def row_to_dict(series_row: pd.Series) -> dict:
    return (
        series_row._asdict()
        if hasattr(series_row, "_asdict")
        else dict(series_row)
    )


def prettify_cols(series_row: pd.Series) -> list[str]:
    return [
        column.title().replace(" ", "_").lower()
        for column in row_to_dict(series_row).keys()
    ]


def pretty_dict(d: dict, indent: int = 4, stage: str = "") -> str:
    """Converts a dictionary into a pretty-printed, indented JSON string.

    Handles custom types like Enums safely.
    """
    pretty = json.dumps(
        d,
        indent=indent,
        default=lambda o: o.value if isinstance(o, Enum) else str(o),
    )

    return pretty


def match_company_by_keywords(text: str) -> str | None:
    text = text.lower()
    for company, keywords in COMPANY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return company
    return None


def sum_bytes_in_dir(dir_path: str) -> int:
    """
    Sum the sizes (in bytes) of all regular files in dir_path,
    including subdirectories.
    """
    total = 0
    root = Path(dir_path)

    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                # Skip files we can't stat (permissions, etc.)
                continue

    return total


def format_bytes(num_bytes: int) -> str:
    """
    Convert an integer byte count to a compact size string.

    Uses binary units:
    1 KB = 1024 bytes
    1 MB = 1024 KB
    1 GB = 1024 MB
    1 TB = 1024 GB

    Examples:
        format_bytes(1024)            -> "1KB"
        format_bytes(1_572_864)       -> "1.5MB"
        format_bytes(5_368_709_120)   -> "5GB"
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if size.is_integer():
        return f"{int(size)}{units[unit_index]}"

    return f"{size:.2f}".rstrip("0").rstrip(".") + units[unit_index]
