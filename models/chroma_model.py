from __future__ import annotations

# models/chroma_model.py
# +---------------------------------------------------------------------------+
# |                            CHROMA DB MODEL                                |
# +---------------------------------------------------------------------------+
# Python Libraries
import os
import shutil

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
    DOCUMENT_DIR_PERM,
    HF_BATCH_SIZE,
)
from src.enums import Company
from src.utils import log_chat_transcript, sum_bytes_in_dir


class ChromaModel(GeminiModel):
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both
    similarity-based and metadata-structured Self-Query retrieval mechanisms.
    """

    def __init__(self):
        super().__init__()
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY

        self._client = self._load_client()
        self.collection_name = CHROMA_COLL_NAME
        self.vector_storage = self._get_vector_storage()

    def _load_client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(path=os.path.abspath(CHROMA_DB_DIR))

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

    def _search_with_scores(
        self, company: str
    ) -> list[tuple[Document, float]]:
        """
        Executes a similarity search with distance scores against the vector collection,
        applying optional company metadata filtering.
        """
        params = {"k": CHROMA_RESULT_CNT}
        if company:
            params["filter"] = {"company": company.lower()}

        return self.vector_storage.similarity_search_with_score(
            search_type="similarity",
            search_kwargs=params,
        )

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

    """
    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            # e.g., "models/text-embedding-004" or "models/gemini-embedding-001"
            model=f"models/{self.embedding_model}",
            google_api_key=MODEL_API_KEY,
        )
    """

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
        self, documents: list, batch_size: int = HF_BATCH_SIZE
    ) -> bool:
        """
        Embeds and adds documents to the vector store in batches using explicit retry logic
        that parses vendor rate limit messages and backs off gracefully.
        """

        # 11,450 chunks, 774 md files
        # 10,675 chunks, 775 md files?
        document_cnt = len(documents)

        log_chat_transcript(
            "CHROMA_MODEL",
            f"➕ Adding {document_cnt} vector documents with a batch size of {batch_size}.",
        )

        for i in range(0, document_cnt, batch_size):
            # log_chat_transcript(
            #    "CHROMA_MODEL", get_progress_bar(i, document_cnt)
            # )
            self.vector_storage.add_documents(documents[i : i + batch_size])

        return True if i >= document_cnt - 1 else False

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
        Applies an exact company metadata filter if specified; otherwise searches globally.
        """
        kwargs = {"k": CHROMA_RESULT_CNT}

        if company and company.strip().lower() != Company.NONE.lower():
            kwargs["filter"] = {"company": company.lower()}

        return self.vector_storage.similarity_search_with_score(
            query=query_str, **kwargs
        )

    @staticmethod
    def delete() -> None:
        """
        Purges the target database directory (db/) to reset ChromaDB
        states. Static because it's pure filesystem work — it never
        touches a client/collection — which lets callers wipe the
        directory *before* any ChromaModel instance (and its live
        connection) exists, avoiding a stale connection pointed at
        files that no longer exist.
        """
        target_db_dir = os.path.abspath(CHROMA_DB_DIR)

        log_chat_transcript(
            "CHROMA_MODEL",
            f"🗑️ Wiping Database directory: {target_db_dir}...",
        )

        if not os.path.exists(target_db_dir):
            log_chat_transcript(
                "CHROMA_MODEL",
                f"⚠️ Warning: Directory {target_db_dir} does not exist. Creating a fresh one now...",
            )
            os.makedirs(target_db_dir, exist_ok=True)
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            return None

        dir_size = sum_bytes_in_dir(target_db_dir)
        log_chat_transcript(
            "CHROMA_DB_SIZE", f"{target_db_dir} directory is {dir_size}."
        )

        for filename in os.listdir(target_db_dir):
            filepath = os.path.join(target_db_dir, filename)
            try:
                log_chat_transcript(
                    "CHROMA_MODEL",
                    f"Purging database artifact: {filepath}...",
                )

                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)

            except FileNotFoundError as e:
                log_chat_transcript(
                    "CHROMA_MODEL",
                    f"🚨 Failed to wipe element path target {filepath}. Exception: {e}",
                )

        if next(os.scandir(target_db_dir), None) is None:
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            log_chat_transcript(
                "CHROMA_MODEL",
                f"📁 Database directory {CHROMA_DB_DIR} is empty and ready for use.",
            )
            log_chat_transcript(
                "CHROMA_MODEL",
                f"🖊️ {CHROMA_DB_DIR} privileges are set to {DOCUMENT_DIR_PERM}.\n",
            )
