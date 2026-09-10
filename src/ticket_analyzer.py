# src/ticket_analyzer.py
# +---------------------------------------------------------------------------+
# |                                TICKET ANALYZER                                 |
# +---------------------------------------------------------------------------+

# Vendor Libraries
from typing import Union

import pandas as pd

# Local Libraries
from agents.claude_agent import ClaudeAgent
from agents.hackerrank_agent import HackerrankAgent
from agents.support_agent import SupportAgent
from agents.visa_agent import VisaAgent
from src.prompt_builder import PromptBuilder
from src.utils import log_chat_transcript


class TicketAnalyzer:
    def __init__(self, dataset: dict) -> None:
        self.chroma_model = dataset.get("chroma_model")
        self.support_agent_model = dataset.get("model")
        self.agent = None

    def build_prompt_by_company(
        self,
        row_index: int,
        ticket_df: pd.DataFrame,
    ) -> str:

        # Get company
        self.agent = self._get_agent(row_index, ticket_df)
        log_chat_transcript("AGENT_LOADED", self.agent.title)

        # Prioritieze the input data:
        # Company, Issue, Subject
        # self.agent.issue = None

        # Classify the issue
        # identify req type, classify issue into a product area
        # assess urgency and risk
        self.agent.classify()

        # decide whether to replay or escelate

        # Doc retrieval (for dataset)
        # get relevant documents
        documents = self.agent.retrieve_relevant_documents()
        self.agent.ground(documents)

        # model analysis
        # generate a safe, grounded response

        # get prompt
        columns = SupportAgent.normalized_columns(ticket_df)
        exported_dataset = self.agent.export(columns)
        print(f"exported_dataset = {exported_dataset}")

        # Load Prompt Builder to get the prompt
        dataset = exported_dataset | {
            "document_chunks": documents,
            "row_index": row_index,
        }
        builder = PromptBuilder(dataset)

        return builder.prompt.strip()

    def _get_agent(
        self, row_index: int, ticket_df: pd.DataFrame
    ) -> Union[SupportAgent, ClaudeAgent, HackerrankAgent, VisaAgent]:
        agent_params = {
            "row_index": row_index,
            "ticket_df": ticket_df,
            "chroma_model": self.chroma_model,
            "support_agent_model": self.support_agent_model,
        }

        # company = SupportAgent.resolve_company(ticket_df)

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
            raise ValueError(f"Unsupported company: '{company}'")

        self.support_agent_model.title = company.title() + " Agent Model"
        agent_params["support_agent_model"] = self.support_agent_model

        return agent_mapping[company](**agent_params)
