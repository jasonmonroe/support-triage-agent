# src/utils.py
# +---------------------------------------------------------------------------+
# |                                UTILITIES                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
import time
import uuid
from datetime import datetime

# Local Libraries
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
