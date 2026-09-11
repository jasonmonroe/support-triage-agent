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
from src.enums import RagStatus
from src.ticket_analyzer import TicketAnalyzer
from src.utils import (
    get_progress_bar,
    get_time,
    log_chat_transcript,
    show_timer,
    start_timer,
    sum_bytes_in_dir,
)


def run_rag_pipeline(
    args: dict, dataset: dict, chroma_model: ChromaModel
) -> RagStatus:
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

    def _ingest(
        chroma_model: ChromaModel, doc_handle: DocumentHandler
    ) -> RagStatus:

        # doc_handle = DocumentHandler(dataset.get("md_files"))
        document_chunks = doc_handle.process()
        document_cnt = doc_handle.count_documents()

        log_chat_transcript(
            "RAG_INJECTION", f"🗄️ DOCUMENT_COUNT: {document_cnt}."
        )
        log_chat_transcript(
            "RAG_INJECTION", f"🗄️ CHUNK_COUNT: {doc_handle.count_chunks()}."
        )

        start_time = start_timer()
        vector_status = chroma_model.add_vector_documents(document_chunks)
        show_timer(start_time)

        log_chat_transcript(
            "RAG_INJECTION", f"Vector Status: {vector_status}."
        )

        chroma_db_dir_size = sum_bytes_in_dir(os.path.abspath(CHROMA_DB_DIR))
        log_chat_transcript(
            "RAG_INJECTION",
            f"Chroma DB filesize is {chroma_db_dir_size}.",
        )

        # Show random document information.
        doc_handle.show(random.randint(0, document_cnt))

        # Get new collection count
        new_collection_count = chroma_model.get_collection_count()
        log_chat_transcript(
            "RAG_INJECTION", f"New collection count: {new_collection_count}."
        )

        return _verify(document_cnt, new_collection_count)

    def _verify(document_cnt: int, collection_count: int) -> RagStatus:
        # Count how many documents were ingested (collection)
        log_chat_transcript(
            "RAG_INJECTION",
            f"{chroma_model.collection_name} Collection Count: {collection_count}.",
        )

        # Verify injection data
        if collection_count == document_cnt:
            log_chat_transcript(
                "RAG_INJECTION",
                f"✅ Success all documents {document_cnt} were collected!",
            )
            return RagStatus.SUCCESS

        elif collection_count > document_cnt:
            raise ValueError(
                f"ERROR: How can there be more collections {collection_count} than documents {document_cnt}?!",
            )

        elif document_cnt > collection_count and collection_count > 0:
            log_chat_transcript(
                "RAG_INJECTION",
                f"⚠️ WARNING: Only {collection_count} were collected.",
            )
            return RagStatus.PARTIAL

        elif collection_count == 0:
            log_chat_transcript(
                "RAG_INJECTION", "🚨 ERROR: No documents were collected!"
            )
            return RagStatus.FAIL

    if data_refresh:
        # Refreshing unconditionally — no count check needed, and no
        # ChromaModel instance needed yet. Wipe first so the client we
        # build next never opens a connection that a later delete could
        # invalidate.
        log_chat_transcript(
            "DATA_REFRESH", "🗑️ Wiping database before ingesting..."
        )
        ChromaModel.delete()
        # Reconnect — the directory was just wiped out from under this
        # instance's existing client/collection.
        chroma_model.reload()

        doc_handle = DocumentHandler(dataset.get("md_files"))
        return _ingest(chroma_model, doc_handle)

    # Not refreshing — nothing gets deleted on this path, so it's always
    # safe to construct immediately and check the count before deciding
    # whether to do any ingestion work at all.
    doc_handle = DocumentHandler(dataset.get("md_files"))
    document_cnt = doc_handle.count_documents()
    collection_count = chroma_model.get_collection_count()

    rag_status = _verify(document_cnt, collection_count)
    if rag_status == RagStatus.FAIL:
        return _ingest(chroma_model, doc_handle)

    elif rag_status == RagStatus.PARTIAL:
        collection_pct = collection_count / document_cnt
        ans = input(
            f"Your {chroma_model.collection_name} collection status is {rag_status} at {collection_pct}%. Do you want to ingest again? Y or N? _"
        )

        if ans[:1].upper() == "Y":
            log_chat_transcript(
                "RAG_PIPELINE", "😊 You chose `Yes`.  Ingesting to begin..."
            )
            return _ingest(chroma_model, doc_handle)
        else:
            log_chat_transcript(
                "RAG_PIPELINE", "😦 You chose `No`.  Exiting RAG."
            )


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
                "TICKET_PIPELINE", f"Assembling prompt for index: {row.Index}"
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

            # Logs the exact XML/Text sent to the LLM
            start_time = start_timer()
            prompt = analyzer.build_prompt_by_company(row.Index, row)
            show_timer(start_time)

            # Break loop (due to no retrieved documents or any other error)
            if prompt == "":
                log_chat_transcript(
                    "TICKET_PIPELINE", "😵 Prompt is empty.  Breaking loop."
                )
                break

            """
            Returns output.  Use three inputs and 5 outputs (
                status,
                product_area,
                response,
                justificiation,
                request_ type
                ) to create the output.csv row
            """
            response = support_agent_model.get_response(prompt, row.Index)
            log_chat_transcript("TICKET_PIPELINE", f"Response: {response}.")

            if not response or hasattr(response, "error"):
                log_chat_transcript(
                    "TICKET_PIPELINE",
                    f"🚨 No response was given due to an error.  Breaking loop at row index {row.Index}. 🚨\n",
                )
                break

            log_chat_transcript(
                "TICKET_PIPELINE",
                f"Model Response Time: {get_time(start_time)}",
            )
            show_timer(start_time)

            time.sleep(PAUSE_TIMER)
            output_rows.append(response)

            log_chat_transcript(
                "TICKET_PIPELINE", get_progress_bar(row.Index, row_cnt)
            )
            # print(get_progress_bar(row.Index, row_cnt))

    print("sys.exit(0) Exiting program...")
    sys.exit(0)

    return output_rows
