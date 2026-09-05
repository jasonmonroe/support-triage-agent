# pipelines/main.py
# +---------------------------------------------------------------------------+
# |                               PIPELINES                                   |
# +---------------------------------------------------------------------------+

#  Python Libraries
import inspect
import sys
import time

# Local Libraries
from models.chroma_model import ChromaModel
from models.support_agent_model import SupportAgentModel
from src.constants import PAUSE_TIMER
from src.document_handler import DocumentHandler
from src.utils import (
    get_progress_bar,
    get_time,
    log_chat_transcript,
    show_timer,
    start_timer,
)


def run_rag_pipeline(dataset: dict):
    print(f"Runnning {inspect.currentframe().f_code.co_name}")

    """
    Document Conversion & Metadata Enrichment
    - Convert raw text into LangChain Document objects
    - Add metadata: `company`, `source_file`, `product_area`

    Semantic Chunking
    - `SemanticChunker` evaluates embeddings to find natural topic splits
    - Preserves full document metadata on each chunk

    Vectorization & Chroma Storage
    - Batch-embed chunks & persist to `chroma.db`

    Retrieval Phase (Agentic Querying)
    - Option A: Filtered Similarity Search (by company)
    - Option B: Global Search (when company is None)
    """

    # Process markdown files into enriched chunks
    doc_handle = DocumentHandler(dataset.get("md_files"))
    document_chunks = doc_handle.process()

    # Instantiate LLM Agent Model
    csv_row_cnt = len(dataset.get("support_tickets", []))
    support_agent_model = SupportAgentModel(csv_row_cnt)

    # Store chunks into single target Chroma Collection
    chroma_data = {"model": support_agent_model, "company": None}
    chroma_model = ChromaModel(chroma_data)

    chroma_model.add_vector_documents(document_chunks)

    return chroma_model


def run_process_tickets_pipeline(dataset: dict) -> list:
    sys.exit(0)
    print(f"Runnning {inspect.currentframe().f_code.co_name}")

    tickets_df = dataset.get("support_tickets")
    row_count = tickets_df.shape[0]

    # Load Prompt Builder
    # prompt_builder = PromptBuilder()
    # support_agent_model = SupportAgentModel(row_count)

    output_rows = []
    for row in tickets_df.itertuples():
        print(f"\nrow.Index = {row.Index}")
        if row.Index == 0:
            # if row._______ == "________":

            # print(f"row={row.Issue}")

            log_chat_transcript(
                "PROMPT_ASSEMBLY", f"Assembling prompt for index: {row.Index}"
            )

            start_time = start_timer()

            prompt = None  # assembler.build_prompt_by_user(row)
            log_chat_transcript(
                "PROMPT_BUILT", prompt
            )  # Logs the exact XML/Text sent to the LLM

            response = None

            # response = chat_model.get_response(
            #    prompt, row.Index
            # )  # response is a list
            log_chat_transcript("LLM_RESPONSE", response)

            if not response or hasattr(response, "error"):
                print(
                    f"\n🚨 No response was given due to an error.  Breaking loop at index {row.Index}.\n"
                )
                break

            log_chat_transcript("LLM_RESPONSE_TIME", get_time(start_time))
            show_timer(start_time)

            time.sleep(PAUSE_TIMER)
            output_rows.append(response)

            log_chat_transcript(
                "PROGRESS_BAR", get_progress_bar(row.Index, row_count)
            )
            print(get_progress_bar(row.Index, row_count))

    return output_rows


def _search_content():
    pass
