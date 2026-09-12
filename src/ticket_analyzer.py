# src/ticket_analyzer.py
# +---------------------------------------------------------------------------+
# |                                TICKET ANALYZER                                 |
# +---------------------------------------------------------------------------+

# Python Libraries
from typing import Union

# Vendor Libraries
import pandas as pd

# Local Libraries
from agents.claude_agent import ClaudeAgent
from agents.hackerrank_agent import HackerrankAgent
from agents.support_agent import SupportAgent
from agents.visa_agent import VisaAgent
from src.prompt_builder import PromptBuilder
from src.utils import (
    log_chat_transcript,
    prettify_cols,
    show_timer,
    start_timer,
)


class TicketAnalyzer:
    def __init__(self, dataset: dict) -> None:
        self._agent = None
        self._chroma_model = dataset.get("chroma_model")
        self._support_agent_model = dataset.get("model")

    def build_prompt_by_company(
        self,
        row_index: int,
        ticket_df: pd.DataFrame,
    ) -> str:

        # Get company
        self._agent = self._get_agent(row_index, ticket_df)
        log_chat_transcript(
            "TICKET_ANALYSIS", f"Agent Loaded: {self._agent.title}."
        )

        # Classify the issue then identify req type, classify issue into a
        # product area assess urgency and risk.
        self._agent.classify()

        # Doc retrieval (for dataset)
        start_time = start_timer()
        documents = self._agent.retrieve_relevant_documents()
        show_timer(start_time)

        if len(documents) == 0:
            message = "🚨 ERROR: No retrieved documents found. 🚨"
            log_chat_transcript(message)
            return ""
        else:
            message = f"Retrieved Document Count: {len(documents)}."
            log_chat_transcript(
                "TICKET_ANALYSIS", {"message": message, "documents": documents}
            )

        # Run groundness on the retrieved documents
        start_time = start_timer()
        grounding_results = self._agent.groundness(documents)
        show_timer(start_time)

        self._agent.evaluate_groundness(grounding_results)

        # Assuming all the values are the most accurate from the retrieved
        # documents build a final prompt for final analysis.
        agent_dataset = self._agent.export(prettify_cols(ticket_df))

        # Load Prompt Builder to get the prompt
        dataset = agent_dataset | {
            "document_chunks": documents,
            "row_index": row_index,
        }
        builder = PromptBuilder(dataset)

        return builder.prompt.strip()

    def _get_agent(
        self, row_index: int, ticket_df: pd.DataFrame
    ) -> Union[SupportAgent, ClaudeAgent, HackerrankAgent, VisaAgent]:
        agent_params = {
            "chroma_model": self._chroma_model,
            "support_agent_model": self._support_agent_model,
            "row_index": row_index,
            "ticket_df": ticket_df,
        }

        # Define company
        company = ticket_df.Company.lower()

        if not company:
            return SupportAgent(**agent_params)

        # Registry mapping company names to concrete subclass implementations
        agent_mapping = {
            "claude": ClaudeAgent,
            "hackerrank": HackerrankAgent,
            "visa": VisaAgent,
        }

        if company not in agent_mapping:
            raise ValueError(f"🚨 Unsupported company: '{company}'")

        return agent_mapping[company](**agent_params)
