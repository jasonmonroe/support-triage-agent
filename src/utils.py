# src/utils.py
# +---------------------------------------------------------------------------+
# |                                UTILITIES                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
import time
import uuid
from datetime import datetime, timezone

# Local Libraries
# from main import args
from src.constants import (
    CHAT_TRANSCRIPT_FILE,
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

    # if args.get("log"):
    # print(log_entry)

    # Ensure the file appends cleanly
    with open(CHAT_TRANSCRIPT_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)


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
    for i in range(total):
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
        # Python <3.11 does not accept 'Z'; normalize to +00:00
        normalized = date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        # Fallback to human-readable format
        dt = datetime.strptime(date_str, "%b %d, %Y, %I:%M %p")

    # Ensure UTC-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt
