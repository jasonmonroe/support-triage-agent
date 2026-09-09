# pipelines/main.py
# +---------------------------------------------------------------------------+
# |                               PIPELINES                                   |
# +---------------------------------------------------------------------------+
#  Python Libraries
import inspect
import random
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
from ticket_analyzer import TicketAnalyzer


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

    data_refresh = dataset.get("refresh")
    if data_refresh:
        log_chat_transcript("DATA_REFRESH_FLAG", True)

    # Instantiate LLM Agent Model
    csv_row_cnt = len(dataset.get("support_tickets", []))

    chroma_model = ChromaModel()

    semantic_count = chroma_model.get_document_count()
    log_chat_transcript("SEMANTIC_COUNT", semantic_count)

    if semantic_count > 0 and not data_refresh:
        log_chat_transcript(
            "SEMANTIC_COUNT",
            f"✅ Semantic collection already has {semantic_count} documents — skipping ingestion.",
        )
    else:
        # Process markdown files into enriched chunks
        doc_handle = DocumentHandler(dataset.get("md_files"))

        if data_refresh:
            log_chat_transcript(
                "DATA_REFRESH", "Wiping database and ingesting..."
            )
            chroma_model.delete()

        # Process document chunks and store chunks into single target Chroma Collection.
        document_chunks = doc_handle.process()
        chroma_model.add_vector_documents(document_chunks)

    # Show documents
    rand_file_order = random.randint(0, semantic_count)
    doc_handle.show(rand_file_order)

    return chroma_model


def run_process_tickets_pipeline(dataset: dict, chroma_model) -> list:
    print(f"Runnning {inspect.currentframe().f_code.co_name}")

    tickets_df = dataset.get("support_tickets")
    row_cnt = tickets_df.shape[0]
    support_agent_model = SupportAgentModel({"row_cnt": row_cnt})

    analyzer = TicketAnalyzer(
        {"model": support_agent_model, "choma_model": chroma_model}
    )

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
            prompt = analyzer.build_prompt_by_company(row)

            log_chat_transcript(
                "PROMPT_BUILT", prompt
            )  # Logs the exact XML/Text sent to the LLM

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


def _search_content():
    pass
