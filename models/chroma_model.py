from __future__ import annotations

# models/chroma_model.py
# +---------------------------------------------------------------------------+
# |                            CHROMA DB MODEL                                |
# +---------------------------------------------------------------------------+
import os
import shutil
from typing import List

import chromadb

# Vector Libraries
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Local Libraries
from constants import (
    CHROMA_COLL_NAME,
    CHROMA_DB_DIR,
    CHROMA_RESULT_CNT,
    CHROMA_SERVER_NO_TELEMETRY,
    DB_BATCH_SIZE,
    DOCUMENT_DIR_PERM,
    MODEL_EMBEDDING,
)


class ChromaModel:
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both
    similarity-based and metadata-structured Self-Query retrieval mechanisms.
    """

    def __init__(self):
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY

        self._client = chromadb.PersistentClient(
            path=os.path.abspath(CHROMA_DB_DIR)
        )
        self.collection_name = CHROMA_COLL_NAME
        self.embedding_model = MODEL_EMBEDDING

        self.retriever = None
        self.vector_storage = self._get_vector_storage()

    def get_retriever(self, company: str | None):
        """
        Initializes vector retriever using the target collection.
        Calculates cosine/Euclidean distance to documents with optional exact metadata filtering.
        """
        params = {"k": CHROMA_RESULT_CNT}
        if company:
            params["filter"] = {"company": company.lower()}

        return self.vector_storage.as_retriever(
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
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )

    def add_vector_documents(
        self, documents: list, batch_size: int = DB_BATCH_SIZE
    ) -> None:
        """Batch-embeds and adds documents to the vector store."""
        document_cnt = len(documents)

        print(
            f"\n# --- ➕ Adding {document_cnt} vector documents with a batch size of {batch_size}. ➕ --- #"
        )

        for i in range(0, document_cnt, batch_size):
            self.vector_storage.add_documents(documents[i : i + batch_size])

    def get_documents(self, query_str: str) -> list:
        """Queries the vector storage collection directly using similarity search."""
        return self.vector_storage.similarity_search(
            query=query_str,
            k=CHROMA_RESULT_CNT,
        )

    def get_document_count(self) -> int:
        """Returns the number of documents currently in the vector collection."""
        try:
            return self.vector_storage._collection.count()
        except Exception:
            return 0

    def query(self, company: str | None, text: str) -> List[Document]:
        """Queries ChromaDB using company-filtered or global search."""
        retriever = self.get_retriever(company)
        return retriever.invoke(text)

    def delete(self) -> None:
        """Purges the target database directory (db/) to reset ChromaDB states."""
        target_db_dir = os.path.abspath(CHROMA_DB_DIR)

        print(f"🧹 Wiping Database directory: {target_db_dir}...")

        if not os.path.exists(target_db_dir):
            print(
                f"⚠️ Warning: Directory {target_db_dir} does not exist. Creating a fresh one now..."
            )
            os.makedirs(target_db_dir, exist_ok=True)
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            return None

        for filename in os.listdir(target_db_dir):
            filepath = os.path.join(target_db_dir, filename)
            try:
                print(f"Purging database artifact: {filepath}...")

                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except FileNotFoundError as e:
                print(
                    f"🚨 Failed to wipe element path target {filepath}. Exception: {e}"
                )

        if next(os.scandir(target_db_dir), None) is None:
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            print(
                f"📁 Database directory {CHROMA_DB_DIR} is empty and ready for use."
            )
            print(
                f"🖊️ {CHROMA_DB_DIR} privileges are set to {DOCUMENT_DIR_PERM}.\n"
            )
