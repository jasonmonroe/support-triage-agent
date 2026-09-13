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
import inspect
import os
import sys
import warnings

# Local Libraries
from models.chroma_model import ChromaModel
from pipelines.process_tickets import run_process_tickets_pipeline
from pipelines.rag import run_rag_pipeline
from src.constants import (
    APP_NAME,
    ARGS_LIST,
    CHAT_TRANSCRIPT_FILE,
    CHROMA_COLL_NAME,
    OUTPUT_FILE,
)
from src.data_handler import DataHandler
from src.enums import RagStatus
from src.utils import (
    banner,
    gen_run_id,
    get_time,
    log_chat_transcript,
    show_banner,
    show_timer,
    start_timer,
)


def run_main_pipeline(args: dict) -> bool:
    banner(inspect.currentframe())

    data_handle = DataHandler(args)
    dataset = data_handle.__dict__
    chroma_model = ChromaModel()

    # Store helper data into vector database
    if args.get("rag"):
        start_time = start_timer()
        rag_status = run_rag_pipeline(args, dataset, chroma_model)
        show_timer(start_time)

        # Check if rag was successful
        if rag_status == RagStatus.FAIL:
            log_chat_transcript(
                "MAIN_PIPELINE", "⚠️ No documents were ingested."
            )
            return False
    else:
        collection_count = chroma_model.get_collection_count()
        log_chat_transcript(
            "MAIN_PIPELINE",
            f"🚩 RAG injection flag NOT detected.  `{CHROMA_COLL_NAME}` collection count: {collection_count}.",
        )

        if collection_count == 0:
            log_chat_transcript(
                "MAIN_PIPELINE",
                "🚨 ERROR: There are no collections. Run again with the --rag flag. 🚨",
            )
            return False

    # Process the tickets 🚩
    output_rows = run_process_tickets_pipeline(args, dataset, chroma_model)

    # Saving output rows to file
    data_handle.save_data(output_rows)

    return True


def parse_args(command_line_str: str) -> dict:
    return {arg.strip("--"): (arg in command_line_str) for arg in ARGS_LIST}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if os.path.exists(CHAT_TRANSCRIPT_FILE):
        os.remove(CHAT_TRANSCRIPT_FILE)

    show_banner(APP_NAME)

    log_chat_transcript("MAIN", f"\n-----  🖥️ {APP_NAME} 🖥️  -----")

    prog_start_time = start_timer()
    run_id = gen_run_id()

    log_chat_transcript("MAIN", f"Start Program\nRUN ID: {run_id}")

    global args
    args = parse_args(sys.argv[1:])

    # Start Chat Transcript Logging
    log_chat_transcript("MAIN", f"Initialized pipeline with arguments: {args}")
    result = run_main_pipeline(args)

    if result:
        msg = f"✅️ Data saved to 💾 {OUTPUT_FILE}."
        log_chat_transcript("MAIN", msg)

    show_timer(prog_start_time)
    log_chat_transcript(
        "MAIN",
        get_time(prog_start_time)
        + " End Program Run Time\n"
        + f"RUN ID: {run_id}",
    )
