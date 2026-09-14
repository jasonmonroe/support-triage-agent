# pipelines/process_tickets.py
# +---------------------------------------------------------------------------+
# |                        TICKET PROCESSING PIPELINE                         |
# +---------------------------------------------------------------------------+

# Python Libraries
import inspect
import time

# Local Libraries
from models.support_agent_model import SupportAgentModel
from src.constants import PAUSE_TIMER
from src.ticket_analyzer import TicketAnalyzer
from src.utils import (
    banner,
    get_progress_bar,
    get_time,
    log_chat_transcript,
    show_timer,
    start_timer,
)


def run_process_tickets_pipeline(
    args: dict, dataset: dict, chroma_model
) -> list:

    banner(inspect.currentframe())

    tickets_df = dataset.get("support_tickets")
    row_count = tickets_df.shape[0]
    tickets = tickets_df.itertuples()

    support_agent_model = SupportAgentModel(row_count)
    analyzer = TicketAnalyzer(
        {
            "chroma_model": chroma_model,
            "model": support_agent_model,
        }
    )

    # --- PROCESS TICKETS --- #
    output_rows = []
    for ticket in tickets:
        idx = ticket.Index
        if idx > 0:
            # Logs the exact XML/Text sent to the LLM
            start_time = start_timer()
            prompt = analyzer.build_prompt_by_company(ticket)

            # Break loop (due to no retrieved documents or any other error)
            if not prompt or prompt == "":
                log_chat_transcript(
                    "TICKET_PIPELINE", "😵 Prompt is empty.  Breaking loop."
                )
                break

            response = support_agent_model.get_response(prompt, idx)

            log_chat_transcript(
                f"TICKET_PIPELINE ({idx})",
                f"💬 Prompt\n{prompt}\n\n💬  Response\n{response}\n",
            )

            if not response:
                log_chat_transcript(
                    "TICKET_PIPELINE",
                    (
                        "🚨 No response was given due to an error.  ",
                        f"Breaking loop at row index {idx}. 🚨\n",
                    ),
                )
                break

            output_row = analyzer.format_output(response)

            log_chat_transcript(
                "TICKET_PIPELINE",
                f"Model Response Time: {get_time(start_time)}",
            )

            show_timer(start_time)
            time.sleep(PAUSE_TIMER)

            log_chat_transcript(
                "TICKET_PIPELINE", get_progress_bar(idx, row_count)
            )

            output_rows.append(output_row)

    return output_rows
