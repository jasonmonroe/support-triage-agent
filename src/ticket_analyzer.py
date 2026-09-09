# src/ticket_analyzer.py
# +---------------------------------------------------------------------------+
# |                                TICKET ANALYZER                                 |
# +---------------------------------------------------------------------------+

# Vendor Libraries
import pandas as pd

# Local Libraries
from agents.claude_agent import ClaudeAgent
from agents.hackerrank_agent import HackerrankAgent
from agents.support_agent import SupportAgent
from agents.visa_agent import VisaAgent
from prompt_builder import PromptBuilder


class TicketAnalyzer:
    def __init__(self, dataset: dict) -> None:
        # self.support_agent = SupportAgent()
        # self.cluade_agent = ClaudeAgent()
        # self.hackerrank_agent = HackerrankAgent()
        # self.visa_agent = VisaAgent()
        # self.chroma_model = dataset.get("chroma_model", None)
        self.chroma_model = dataset.get("chroma_model")
        self.support_agent_model = dataset.get("model")
        self.agent = None

    def build_prompt_by_company(self, ticket_df: pd.DataFrame) -> str:

        # Get company
        self.agent = self._get_agent(ticket_df)

        """
        ┌─────────────────────────────────────────────────────────┐
        │              Input: CSV File of Tickets                 │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 1. READ & PARSE TICKET (Subject, Issue, Company)        │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 2. CLASSIFY & ASSESS (Request Type, Product Area, Risk) │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 3. RETRIEVE KNOWLEDGE (Search Markdown Documentation)   │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 4. MAKE DECISION (Safe to Reply vs. Must Escalate)       │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ 5. GENERATE RESPONSE & JUSTIFICATION                    │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │         Output: CSV File Saved to disk (output.csv)     │
        └─────────────────────────────────────────────────────────┘

        """

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

        exported_dataset = self.agent.export()

        # Load Prompt Builder to get the prompt
        dataset = exported_dataset | {"document_chunks": documents}
        builder = PromptBuilder(dataset)

        return builder.prompt.strip()

    def _get_agent(self, ticket_df: pd.DataFrame):
        # company = ticket_df["company"].lower()

        agent = SupportAgent(ticket_df, self.chroma_model)
        company = agent.company
        if not company:
            # company = SupportAgent.find_company(ticket_df)
            return SupportAgent(ticket_df, self.chroma_model)

        if company == "claude":
            return ClaudeAgent(ticket_df, self.chroma_model)

        elif company == "hackerrank":
            return HackerrankAgent(ticket_df, self.chroma_model)

        elif company == "visa":
            return VisaAgent(ticket_df, self.chroma_model)
        else:
            raise ValueError("Company error!")

    def _merge(self) -> dict:
        return {}
