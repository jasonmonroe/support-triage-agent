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
from src.constants import (
    CHROMA_DB_DIR,
    HF_BATCH_SIZE,
    INGEST_LIMIT_RETRIES,
    PAUSE_TIMER,
)
from src.document_handler import DocumentHandler
from src.enums import RagStatus
from src.ticket_analyzer import TicketAnalyzer
from src.utils import (
    banner,
    format_bytes,
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
    banner(inspect.currentframe())

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

    def _ingest(
        chroma_model: ChromaModel, doc_handle: DocumentHandler
    ) -> RagStatus:
        stage = "rag_ingestion".upper()

        document_chunks = doc_handle.process()
        document_count = doc_handle.count_documents()

        log_chat_transcript(
            "RAG_INGESTION", f"🗄️ DOCUMENT_COUNT: {document_count}."
        )
        log_chat_transcript(
            "RAG_INGESTION", f"🗄️ CHUNK_COUNT: {doc_handle.count_chunks()}."
        )

        start_time = start_timer()
        log_chat_transcript(
            "RAG_INGESTION",
            f"➕ Adding {len(document_chunks)} 🗃️ vector document chunks using a batch size of {HF_BATCH_SIZE}...",
        )

        # @TODO output first batch to see what it looks like
        log_chat_transcript(
            "DBG: RAG_INGESTION: 1st CHUNK", document_chunks[0]
        )
        log_chat_transcript(
            "DBG: RAG_INGESTION: 2nd CHUNK", document_chunks[0]
        )
        # @TODO

        vector_status = chroma_model.add_vector_documents(document_chunks)
        show_timer(start_time)

        # log_chat_transcript(
        #    "RAG_INGESTION", f"Vector Status: {vector_status}."
        # )

        chroma_db_dir_size = format_bytes(
            sum_bytes_in_dir(os.path.abspath(CHROMA_DB_DIR))
        )
        log_chat_transcript(
            "RAG_INGESTION",
            f"Chroma DB filesize is {chroma_db_dir_size}.",
        )

        # Show random document information.
        doc_handle.show(random.randint(0, document_count))

        # Get new collection count
        new_collection_count = chroma_model.get_collection_count()
        log_chat_transcript(
            "RAG_INGESTION", f"ℹ️ New collection count: {new_collection_count}."
        )

        counts = {
            "collection_count": new_collection_count,
            "chunk_count": len(document_chunks),
            "document_count": document_count,
        }

        return _verify(counts)

    def _verify(counts: dict) -> RagStatus:

        # Count how many documents were ingested (collection)
        collection_count = counts.get("collection_count", 0)
        document_count = counts.get("document_count", 0)
        chunk_count = counts.get("chunk_count", None)

        log_chat_transcript(
            "RAG_INGESTION",
            f"ℹ️ {chroma_model.collection_name} Collection Count: {collection_count}.",
        )

        if collection_count == 0:
            log_chat_transcript(
                "RAG_INGESTION",
                f"🚨 ERROR: No collections found for {chroma_model.collection_name}.",
            )

            return RagStatus.FAIL

        # Verify on initialization
        if chunk_count is None:
            # SUCCESS
            if collection_count > document_count:
                log_chat_transcript(
                    "RAG_INJECTION",
                    "Collections exceed the amount of documents so we can presume it's a success.",
                )

                return RagStatus.SUCCESS

            # SUCCESS OR PARTIAL
            elif collection_count > 0 and collection_count <= document_count:
                log_chat_transcript(
                    "RAG_INGESTION",
                    f"⚠️ Atleast {collection_count} collections but there might be more.",
                )

                return RagStatus.PARTIAL

        else:
            # Verify after ingestion

            # SUCCESS: Collections match the chunks
            if collection_count == chunk_count:
                log_chat_transcript(
                    "RAG_INGESTION",
                    f"✅ Success! All document (chunks) {document_count} were collected.",
                )

                return RagStatus.SUCCESS

            # PARTIALS
            elif collection_count > 0 and collection_count < chunk_count:
                log_chat_transcript(
                    "RAG_INGESTION",
                    f"⚠️ WARNING: Only {collection_count} were collected.",
                )

                return RagStatus.PARTIAL

            print("WARNING: Returning no RAG status!")
            return None

    if data_refresh:
        # Refreshing unconditionally — no count check needed, and no
        # ChromaModel instance needed yet. Wipe first so the client we
        # build next never opens a connection that a later delete could
        # invalidate.
        log_chat_transcript(
            "RAG_PIPELINE", "🗑️ Wiping database before ingesting..."
        )
        ChromaModel.delete()
        # Reconnect — the directory was just wiped out from under this
        # instance's existing client/collection.
        chroma_model.reload()

        rag_status = _ingest(chroma_model, doc_handle)

    # Not refreshing — nothing gets deleted on this path, so it's always
    # safe to construct immediately and check the count before deciding
    # whether to do any ingestion work at all.
    # doc_handle = DocumentHandler(dataset.get("md_files"))
    # document_count = doc_handle.count_documents()
    # collection_count = chroma_model.get_collection_count()

    counts = {
        "collection_count": chroma_model.get_collection_count(),
        "chunk_count": None,
        "document_count": doc_handle.count_documents(),
    }

    rag_status = _verify(counts)
    input_resp = None
    itr = 1

    while rag_status != RagStatus.SUCCESS and itr < INGEST_LIMIT_RETRIES:
        input_message = (
            f"\nIteration: {itr}: ",
            f"Your {chroma_model.collection_name} collection status is ",
            f"{rag_status}.  Do you want to ingest again? Y or N? _ ",
        )

        input_resp = str(input(input_message))[:1].upper()

        if input_resp == "Y":
            log_chat_transcript(
                "RAG_PIPELINE", "😊 You chose `Yes`.  Ingesting to begin..."
            )

            rag_status = _ingest(chroma_model, doc_handle)

        elif input_resp == "N":
            log_chat_transcript(
                "RAG_PIPELINE", "😦 You chose `No`.  Exiting RAG."
            )
            break
        itr += 1


def run_process_tickets_pipeline(
    args: dict, dataset: dict, chroma_model
) -> list:
    banner(inspect.currentframe())

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

    # --- PROCESS TICKETS --- #
    for row in tickets_df.itertuples():
        print(f"\nrow.Index = {row.Index}")
        if row.Index == 0:
            # if row._______ == "________":
            # print(f"row={row.Issue}")
            print(f"row={row}")
            print(f"Assembling prompt for index: {row.Index}")

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
            log_chat_transcript(
                "TICKET_PIPELINE", f"💬{prompt} \n Response: 💬{response}"
            )

            if not response or hasattr(response, "error"):
                log_chat_transcript(
                    "TICKET_PIPELINE",
                    (
                        "🚨 No response was given due to an error.  ",
                        f"Breaking loop at row index {row.Index}. 🚨\n",
                    ),
                )
                break

            log_chat_transcript(
                "TICKET_PIPELINE",
                f"Model Response Time: {get_time(start_time)}",
            )
            show_timer(start_time)

            time.sleep(PAUSE_TIMER)

            log_chat_transcript(
                "TICKET_PIPELINE", get_progress_bar(row.Index, row_cnt)
            )

            output_rows.append(response)
        print("DBG: Breaking after first iter!")
        break

    print("sys.exit(0) Exiting program...")
    sys.exit(0)

    return output_rows
