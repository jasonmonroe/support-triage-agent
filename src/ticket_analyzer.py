# src/ticket_analyzer.py
# +---------------------------------------------------------------------------+
# |                               TICKET ANALYZER                             |
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
from src.enums import Company, Status
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
        self._ticket = {}
        self._row_index = None

    def process(
        self,
        # row_index: int,
        ticket_df: pd.DataFrame,
    ) -> str:

        self._row_index = ticket_df.Index
        self._agent = self._get_agent(ticket_df)
        log_chat_transcript(
            "🎟️ TICKET_ANALYZER", f"Agent Loaded: {self._agent.title}."
        )

        # Classify the issue then identify req type, classify issue into a
        # product area assess urgency and risk.
        self._agent.classify_issue()
        self._agent.make_decision()

        documents = []
        if self._agent.status == Status.ESCALATED:
            """
            Hard gate: make_decision() already escalated on risk signals
            alone. Skip retrieval and the 3-LLM-call groundness pipeline
            entirely — nothing there can un-escalate a ticket already
            flagged as risky, so don't spend the tokens finding out.
            """

            log_chat_transcript(
                "🎟️ TICKET_ANALYZER",
                "🚩 Hard-escalated pre-retrieval: "
                f"{self._agent.justification}",
            )
        else:
            # Doc retrieval (for dataset)
            start_time = start_timer()
            documents = self._agent.retrieve_relevant_documents()
            show_timer(start_time)

            """
            if len(documents) == 0:
                log_chat_transcript(
                    "🎟️ TICKET_ANALYZER",
                    "🚨 ERROR: No retrieved documents found. 🚨",
                )
                return ""
            else:
                log_chat_transcript(
                    "🎟️ TICKET_ANALYZER",
                    {
                        "message": (
                            f"Retrieved Document Count: {len(documents)}."
                        ),
                        "documents": documents,
                    },
                )
            """

            # Run groundness on the retrieved documents
            start_time = start_timer()
            grounding_results = self._agent.groundness(documents)
            # self._agent.evaluate_groundness(grounding_results)
            show_timer(start_time)

        # Assuming all the values are the most accurate from the retrieved
        # documents build a final prompt for final analysis.

        ticket_columns = prettify_cols(ticket_df)
        agent_dataset = self._agent.export(ticket_columns)

        """
        Returns output.  Use three inputs and 5 outputs (
            status,
            product_area,
            response,
            justificiation,
            request_ type
            ) to create the output.csv row
        """

        self._ticket = agent_dataset

        # Load Prompt Builder to get the prompt
        dataset = agent_dataset | {
            "document_chunks": documents,
            "row_index": self._row_index,
        }

        builder = PromptBuilder(dataset)

        return self._get_output(builder.prompt.strip())

    def _get_agent(
        self, ticket_df: pd.DataFrame
    ) -> Union[SupportAgent, ClaudeAgent, HackerrankAgent, VisaAgent]:
        agent_params = {
            "chroma_model": self._chroma_model,
            "row_index": self._row_index,
            "support_agent_model": self._support_agent_model,
            "ticket_df": ticket_df,
        }

        # Define company
        company = ticket_df.Company.lower()

        if not company or company == Company.NONE.lower():
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

    def _get_output(self, prompt: str) -> dict:

        response = self._support_agent_model.get_response(
            prompt, self._row_index
        )

        return self._format_output(response)

    def _format_output(self, response: dict) -> dict:
        """Format response to an output row dict, overriding ticket keys with
        LLM response values."""

        output = {}
        for key, value in self._ticket.items():
            # Override with LLM response if the key exists in the response
            output[key] = response[key] if key in response else value

        return output
