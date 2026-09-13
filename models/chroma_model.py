from __future__ import annotations

# models/chroma_model.py
# +---------------------------------------------------------------------------+
# |                            CHROMA DB MODEL                                |
# +---------------------------------------------------------------------------+
# Python Libraries
import os

import chromadb

# Vector Libraries
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Local Libraries
from models.gemini_model import GeminiModel
from src.constants import (
    CHROMA_COLL_NAME,
    CHROMA_DB_DIR,
    CHROMA_RESULT_CNT,
    CHROMA_SERVER_NO_TELEMETRY,
    HF_BATCH_SIZE,
)
from src.enums import Company
from src.utils import (
    format_bytes,
    get_progress_bar,
    log_chat_transcript,
    sum_bytes_in_dir,
)


class ChromaModel(GeminiModel):
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both
    similarity-based and metadata-structured Self-Query retrieval mechanisms.
    """

    def __init__(self):
        super().__init__()
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY

        self.collection_name = CHROMA_COLL_NAME
        self.reload()
        # self._client = self._load_client()
        # self.vector_storage = self._get_vector_storage()

    def _load_client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(
            path=os.path.abspath(CHROMA_DB_DIR),
            settings=chromadb.Settings(allow_reset=True),
        )

    def reload(self) -> None:
        """
        Reconnects to the persistent store. Required after `delete()`
        wipes the database directory out from under an already-open
        instance — its `_client`/`vector_storage` still reference the
        now-deleted files, so writes through them fail with "attempt to
        write a readonly database" instead of hitting the fresh files.
        """
        self._client = self._load_client()
        self.vector_storage = self._get_vector_storage()

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the primary vector storage collection."""
        if self.embedding_model is None:
            raise ValueError(
                "🚨 An embedding model must be provided in dataset!"
            )

        return Chroma(
            client=self._client,
            embedding_function=self._get_hf_embeddings(),
            collection_name=self.collection_name,
        )

    def _get_hf_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Instantiates local HuggingFace embedding provider.
        Runs locally on CPU/GPU without needing external embedding API keys.
        """
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={
                "device": "mps"
            },  # use `cps` if on Intel based machine
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": HF_BATCH_SIZE,
            },
        )

    def add_vector_documents(
        self, chunks: list, batch_size: int = HF_BATCH_SIZE
    ) -> bool:
        """
        Embeds and adds (chunked) documents to the vector store in batches using
        explicit retry logic that parses vendor rate limit messages and backs
        off gracefully.
        """

        # prev1: Chunks: 11,450, MD Files: 774
        # prev:  Chunks: 10,675, MD Files: 775
        # curr:  Chunks: 10,433, MD Files: 770
        chunk_count = len(chunks)
        vector_chunks = 0
        for i in range(0, chunk_count, batch_size):
            start_time = start_timer()
            log_chat_transcript(
                "CHROMA_MODEL", get_progress_bar(i, chunk_count, batch_size)
            )
            chunk_ids = self.vector_storage.add_documents(
                chunks[i : i + batch_size]
            )
            vector_chunks += len(chunk_ids)

            print(f"{vector_chunks} vector chunks added so far.")
            show_timer(start_time)
        return vector_chunks == chunk_count

    def get_collection_count(self) -> int:
        """Returns the number of documents currently in the vector collection."""
        try:
            return self.vector_storage._collection.count()
        except Exception as e:
            log_chat_transcript(
                "CHROMA_MODEL", f"ERROR: Collection Count: {e}"
            )
            return 0

    def query(
        self, query_str: str, company: str | None = None
    ) -> list[tuple[Document, float]]:
        """
        Queries the vector collection using similarity search with distance scores.
        Applies an exact company metadata filter if specified; otherwise searches
        globally.
        """
        kwargs = {"k": CHROMA_RESULT_CNT}

        if company and company.strip().lower() != Company.NONE.lower():
            kwargs["filter"] = {"company": company.lower()}

        log_chat_transcript("CHROMA_MODEL: QUERY", f"💬 {query_str}")

        return self.vector_storage.similarity_search_with_score(
            query=query_str, search_type="similarity", search_kwargs=kwargs
        )

    def delete(self) -> None:
        """
        Resets the persistent Chroma store via the client's own `reset()`
        API instead of deleting chroma_db/ on disk. The Rust-backed
        persistent client (chromadb>=0.6) manages its own SQLite
        connections/locks internally; wiping the directory's files out
        from under it and pointing a new client at the same path left it
        in an inconsistent state ("attempt to write a readonly database")
        instead of a clean slate. `reset()` requires `allow_reset=True`
        on the client's Settings — see `_load_client`.
        """
        target_db_dir = os.path.abspath(CHROMA_DB_DIR)

        dir_size = format_bytes(sum_bytes_in_dir(target_db_dir))
        log_chat_transcript(
            "CHROMA_MODEL",
            f"🗑️ Resetting collection {self.collection_name} "
            f"({target_db_dir}, {dir_size})...",
        )

        self._client.reset()

        log_chat_transcript(
            "CHROMA_MODEL",
            f"📁 Collection `{self.collection_name}` has been reset and is ready for use.",
        )
