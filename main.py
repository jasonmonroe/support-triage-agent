# main.py

"""
+---------------------------------------------------------------------------+
|                       HACKERRANK ORCHESTRATE CHALLENGE                    |
+---------------------------------------------------------------------------+

Builds a terminal-based support triage agent that can handle support tickets across three ecosystems:
HackerRank, Claude, Visa Support

Companies receive thousands of customer support tickets daily. Human support agents waste time answering
simple repetitive questions (like "How do I reset my password?") while urgent issues (like fraud, security
vulnerabilities, or account locks) sit in long queues.

This project builds an automated AI agent that acts as a first responder. It automatically reads incoming
customer messages, checks official internal documentation, answers simple questions safely, and escalates
complex or risky situations to a human team.

@link https://www.hackerrank.com/contests/hackerrank-orchestrate-august26/challenges/message-notification-router
Clone Repo - @link https://github.com/interviewstreet/hackerrank-orchestrate-august26
"""

__author__ = "Jason Monroe (jason@jasonmonroe.com)"
__copyright__ = "Copyright © 2011-2026 Monroe Labs Co."
__date__ = "2026-09-02"
__version__ = "1.0.0"


# Python Libraries
# from dotenv import load_dotenv
# load_dotenv(dotenv_path="../.env", override=True)
import inspect
import os
import sys
import warnings

from models.chroma_model import ChromaModel

# Local Libraries
from pipelines.main import run_process_tickets_pipeline, run_rag_pipeline
from src.constants import (
    APP_NAME,
    ARGS_LIST,
    CHAT_TRANSCRIPT_FILE,
    OUTPUT_FILE,
    SAMPLE_SUPPORT_TICKETS_FILE,
)
from src.data_handler import DataHandler
from src.utils import (
    gen_run_id,
    get_time,
    log_chat_transcript,
    show_banner,
    show_timer,
    start_timer,
)


def run_main_pipeline(args: dict):
    print(
        f"🏃 Runnning {inspect.currentframe().f_code.co_name.title().replace('_', ' ')}"
    )

    if args.get("sample"):
        print(f"\nLoading {SAMPLE_SUPPORT_TICKETS_FILE}")

    data_handle = DataHandler(args)

    # Store helper data into vector database
    chroma_model = ChromaModel()
    if args.get("rag"):
        start_time = start_timer()
        chroma_model = run_rag_pipeline(args, data_handle.__dict__)
        show_timer(start_time)
    # sys.exit(0)

    # Process the tickets 🚩
    output_rows = run_process_tickets_pipeline(
        args, data_handle.__dict__, chroma_model
    )

    log_chat_transcript("OUTPUT_ROWS", output_rows)
    sys.exit(0)

    # Saving output rows to file
    # data_handle.save_data(output_rows)

    return True


def parse_args(command_line_str: str) -> dict:
    return {arg.strip("--"): (arg in command_line_str) for arg in ARGS_LIST}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.remove(CHAT_TRANSCRIPT_FILE)
    log_chat_transcript("APP NAME", f"\n-----  🖥️ {APP_NAME} 🖥️  -----\n")

    prog_start_time = start_timer()
    run_id = gen_run_id()

    print(f"----  🖨️️  START RUN ID: {run_id}  🖨️️  ----")
    log_chat_transcript("Start Program", f"RUN ID: {run_id}")

    global args
    args = parse_args(sys.argv[1:])

    show_banner(APP_NAME)

    # Start Chat Transcript Logging
    log_chat_transcript(
        "PIPELINE_INIT", f"Initialized pipeline with arguments: {args}"
    )
    result = run_main_pipeline(args)

    if result:
        msg = f"✅️ Data saved to {OUTPUT_FILE}."
        print(msg)
        log_chat_transcript("DATA SAVED", msg)

    show_timer(prog_start_time)
    log_chat_transcript(
        "End Program Run Time", get_time(prog_start_time) + f"RUN ID: {run_id}"
    )

    print(f"\n-----  🖨️️ END RUN ID: {run_id} 🖨️️  -----\n")
