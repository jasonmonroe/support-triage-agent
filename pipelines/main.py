# pipelines/main.py
# +---------------------------------------------------------------------------+
# |                               PIPELINES                                   |
# +---------------------------------------------------------------------------+
#  Python Libraries
import inspect
import random
import sys
import time

# Local Libraries
from models.chroma_model import ChromaModel
from models.support_agent_model import SupportAgentModel
from src.constants import PAUSE_TIMER
from src.document_handler import DocumentHandler
from src.ticket_analyzer import TicketAnalyzer
from src.utils import (
    get_progress_bar,
    get_time,
    log_chat_transcript,
    show_timer,
    start_timer,
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
    doc_handle = DocumentHandler(dataset.get("md_files"))

    # Instantiate LLM Agent Model
    chroma_model = ChromaModel()
    collection_count = chroma_model.get_collection_count()
    log_chat_transcript("COLLECTION_COUNT", collection_count)

    if collection_count > 0 and not data_refresh:
        log_chat_transcript(
            "collection_count",
            f"✅ Semantic collection already has {collection_count} documents — skipping ingestion.",
        )
    else:
        # Process markdown files into enriched chunks
        if data_refresh:
            log_chat_transcript(
                "DATA_REFRESH", "🗑️ Resetting database and ingesting..."
            )
            chroma_model.delete_collection()
            chroma_model.delete()

        # Process document chunks and store chunks into single target Chroma Collection.
        document_chunks = doc_handle.process()
        log_chat_transcript(
            "🗄️ DOCUMENT_COUNT", doc_handle.count_documents()
        )  # 774
        log_chat_transcript(
            "🗄️ CHUNK_COUNT", doc_handle.count_chunks()
        )  # 11450
        chroma_model.add_vector_documents(document_chunks)

    # Show documents
    rand_file_order = random.randint(0, collection_count)
    doc_handle.show(rand_file_order)

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
            sys.exit(0)

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

    return output_rows
