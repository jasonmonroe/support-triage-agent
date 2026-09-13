# pipelines/rag.py
# +---------------------------------------------------------------------------+
# |                                RAG PIPELINE                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import inspect
import os
import random

# Local Libraries
from models.chroma_model import ChromaModel
from src.constants import (
    CHROMA_COLL_NAME,
    CHROMA_DB_DIR,
    HF_BATCH_SIZE,
    INGEST_LIMIT_RETRIES,
)
from src.document_handler import DocumentHandler
from src.enums import RagStatus
from src.utils import (
    banner,
    format_bytes,
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

    if data_refresh:
        # Reset via the client's own reset() API (see ChromaModel.delete)
        # rather than wiping chroma_db/ on disk — the Rust-backed persistent
        # client doesn't tolerate its files disappearing out from under it.
        log_chat_transcript(
            "RAG_PIPELINE", "🗑️ Wiping database before ingesting..."
        )

        # Rebuild the LangChain Chroma wrapper against the now-empty collection.
        chroma_model.delete()
        chroma_model.reload()

        rag_status = _ingest(chroma_model, doc_handle)

    # Not refreshing — nothing gets deleted on this path, so it's always safe
    # to construct immediately and check the count before deciding whether to
    # do any ingestion work at all.
    counts = {
        "chunk_count": None,
        "collection_count": chroma_model.get_collection_count(),
        "document_count": doc_handle.count_documents(),
    }

    rag_status = _verify(counts)
    input_resp = None
    itr = 1

    while rag_status != RagStatus.SUCCESS and itr < INGEST_LIMIT_RETRIES:
        input_message = (
            f"\nIteration: {itr}: ",
            f"Your {CHROMA_COLL_NAME} collection status is ",
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

    if itr >= INGEST_LIMIT_RETRIES:
        log_chat_transcript(
            "RAG_PIPELINE", "Max iterations exhausted for RAG."
        )


def _ingest(
    chroma_model: ChromaModel, doc_handle: DocumentHandler
) -> RagStatus:

    document_chunks = doc_handle.process()
    document_count = doc_handle.count_documents()

    log_chat_transcript(
        "RAG_INGESTION",
        f"🗄️ DOCUMENT_COUNT: {document_count}\n🗄️ CHUNK_COUNT: {doc_handle.count_chunks()}",
    )

    start_time = start_timer()
    log_chat_transcript(
        "RAG_INGESTION",
        f"➕ Adding {len(document_chunks)} 🗃️ vector document chunks using a batch size of {HF_BATCH_SIZE}...",
    )

    vector_status = chroma_model.add_vector_documents(document_chunks)
    show_timer(start_time)

    log_chat_transcript("RAG_INGESTION", f"Vector Status: {vector_status}.")

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
        "RAG_INGESTION",
        f"ℹ️  `{CHROMA_COLL_NAME}` New collection count: {new_collection_count}",
    )

    counts = {
        "collection_count": new_collection_count,
        "chunk_count": len(document_chunks),
        "document_count": document_count,
    }

    return _verify(counts)


def _verify(counts: dict) -> RagStatus | None:

    # Count how many documents were ingested (collection)
    collection_count = counts.get("collection_count", 0)
    document_count = counts.get("document_count", 0)
    chunk_count = counts.get("chunk_count", None)

    log_chat_transcript(
        "RAG_INGESTION",
        f"ℹ️  `{CHROMA_COLL_NAME}` Collection Count: {collection_count}",
    )

    if collection_count == 0:
        log_chat_transcript(
            "RAG_INGESTION",
            f"🚨 ERROR: No collections found for {CHROMA_COLL_NAME}.",
        )

        return RagStatus.FAIL

    # ℹ️ Note: Verify on initialization.  Assume refresh flag is on or it's empty!
    if chunk_count is None:
        # SUCCESS
        if collection_count > document_count:
            log_chat_transcript(
                "RAG_INJECTION",
                "ℹ️ --- No Refresh ---\nCollections exceed the amount of documents so we can presume it's a success.",
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
        # --- Verify after ingestion --- #

        # SUCCESS: Collections match the chunks
        if collection_count == chunk_count:
            log_chat_transcript(
                "RAG_INGESTION",
                f"✅ Success! All document (chunks) {collection_count} were collected.",
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
