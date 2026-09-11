# pipelines/main.py
# +---------------------------------------------------------------------------+
# |                               PIPELINES                                   |
# +---------------------------------------------------------------------------+
#  Python Libraries
import inspect
import os
import random
import sys
import time

# Local Libraries
from models.chroma_model import ChromaModel
from models.support_agent_model import SupportAgentModel
from src.constants import CHROMA_DB_DIR, PAUSE_TIMER
from src.document_handler import DocumentHandler
from src.ticket_analyzer import TicketAnalyzer
from src.utils import (
    get_progress_bar,
    get_time,
    log_chat_transcript,
    show_timer,
    start_timer,
    sum_bytes_in_dir,
)


def run_rag_pipeline(args: dict, dataset: dict):
    print(
        f"\n🏃 Runnning {inspect.currentframe().f_code.co_name.title().replace('_', ' ')}..."
    )

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

    data_refresh = args.get("refresh")

    def _ingest(chroma_model: ChromaModel) -> None:
        doc_handle = DocumentHandler(dataset.get("md_files"))
        document_chunks = doc_handle.process()
        log_chat_transcript("🗄️ DOCUMENT_COUNT", doc_handle.count_documents())
        log_chat_transcript("🗄️ CHUNK_COUNT", doc_handle.count_chunks())

        chroma_model.add_vector_documents(document_chunks)

        chroma_db_dir_size = sum_bytes_in_dir(os.path.abspath(CHROMA_DB_DIR))
        log_chat_transcript(
            "CHROMA_DB_DIR_SIZE",
            f"Chroma DB filesize is {chroma_db_dir_size}.",
        )

        doc_handle.show(random.randint(0, doc_handle.count_documents()))

    if data_refresh:
        # Refreshing unconditionally — no count check needed, and no
        # ChromaModel instance needed yet. Wipe first so the client we
        # build next never opens a connection that a later delete could
        # invalidate.
        log_chat_transcript(
            "DATA_REFRESH", "🗑️ Wiping database before ingesting..."
        )
        ChromaModel.delete()

        chroma_model = ChromaModel()
        _ingest(chroma_model)
        return chroma_model

    # Not refreshing — nothing gets deleted on this path, so it's always
    # safe to construct immediately and check the count before deciding
    # whether to do any ingestion work at all.
    chroma_model = ChromaModel()
    collection_count = chroma_model.get_collection_count()
    log_chat_transcript("COLLECTION_COUNT", collection_count)

    if collection_count > 0:
        log_chat_transcript(
            "COLLECTION_COUNT",
            f"✅ Semantic collection already has {collection_count} documents — skipping ingestion.",
        )
    else:
        _ingest(chroma_model)

    return chroma_model


def run_process_tickets_pipeline(
    args: dict, dataset: dict, chroma_model
) -> list:
    print(
        f"\n🏃 Runnning {inspect.currentframe().f_code.co_name.title().replace('_', ' ')}..."
    )

    tickets_df = dataset.get("support_tickets")
    row_cnt = tickets_df.shape[0]
    support_agent_model = SupportAgentModel(row_cnt)

    analyzer = TicketAnalyzer(
        {
            "chroma_model": chroma_model,
            "model": support_agent_model,
        }
    )

    output_rows = []
    for row in tickets_df.itertuples():
        print(f"\nrow.Index = {row.Index}")
        if row.Index == 0:
            # if row._______ == "________":
            # print(f"row={row.Issue}")
            print(f"row={row}")
            log_chat_transcript(
                "PROMPT_ASSEMBLY", f"Assembling prompt for index: {row.Index}"
            )

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

            start_time = start_timer()
            prompt = analyzer.build_prompt_by_company(row.Index, row)
            show_timer(start_time)

            log_chat_transcript(
                "PROMPT_BUILT", prompt
            )  # Logs the exact XML/Text sent to the LLM

            # Break loop (due to no retrieved documents or any other error)
            if prompt == "":
                break

            """
            Returns output.  Use three inputs and 5 outputs (status, product_area, response, justificiation, request_ type)
            to create the output.csv row
            """
            response = analyzer.support_agent_model.get_response(prompt)
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
                "PROGRESS_BAR", get_progress_bar(row.Index, row_cnt)
            )
            print(get_progress_bar(row.Index, row_cnt))

    print("Exiting program...")
    sys.exit(0)

    return output_rows
