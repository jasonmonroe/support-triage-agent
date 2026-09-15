# pipelines/process_tickets.py
# +---------------------------------------------------------------------------+
# |                        TICKET PROCESSING PIPELINE                         |
# +---------------------------------------------------------------------------+

# Python Libraries
import inspect

# Local Libraries
from models.support_agent_model import SupportAgentModel
from src.ticket_analyzer import TicketAnalyzer
from src.utils import (
    banner,
    get_progress_bar,
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

        # Logs the exact XML/Text sent to the LLM
        start_time = start_timer()

        output_row = analyzer.process(ticket)

        log_chat_transcript(
            "🎟️ TICKET_PIPELINE", get_progress_bar(idx, row_count)
        )

        show_timer(start_time)

        output_rows.append(output_row)

    log_chat_transcript("🎟️ TICKET_PIPELINE", output_rows)

    return output_rows
