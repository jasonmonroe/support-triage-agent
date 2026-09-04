from __future__ import annotations

# models/chroma_model.py
# +---------------------------------------------------------------------------+
# |                            CHROMA DB MODEL                                |
# +---------------------------------------------------------------------------+
#
# Python Libraries
import os
import shutil
from typing import List

import chromadb

# Vector Libraries
from langchain_chroma import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.documents import Document
from langchain_text_splitters import (
    SemanticChunker,
)

# Local Libraries
from constants import (
    CHROMA_COLL_NAME,
    CHROMA_DB_DIR,
    CHROMA_RESULT_CNT,
    CHROMA_SERVER_NO_TELEMETRY,
    DOCUMENT_CHUNK_SIZE,
    DOCUMENT_DIR_PERM,
    MODEL_EMBEDDING,
    SEMANTIC_THRESH_LIMIT,
)


class ChromaModel:
    """
    Manages the persistent vector store lifecycle using ChromaDB and handles both similarity-based and
    metadata-structured Self-Query retrieval mechanisms using a clean, flat architecture.

    see: https://docs.langchain.com/oss/python/langchain/rag?_gl=1
    """

    def __init__(self, dataset: dict):
        os.environ["CHROMA_SERVER_NO_TELEMETRY"] = CHROMA_SERVER_NO_TELEMETRY

        self._client = chromadb.PersistentClient(
            path=os.path.abspath(CHROMA_DB_DIR)
        )
        self.collection_name = CHROMA_COLL_NAME
        self.embedding_model = MODEL_EMBEDDING
        self.model = None
        self.title = ""

        self.semantic_text_splitter = self._get_semantic_text_splitter()
        self.semantic_storage = self._get_semantic_storage()
        self.vector_storage = self._get_vector_storage()
        self.retriever = self.get_retriever(dataset.get("company"))

        self._set_attrs(dataset)

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_retriever(self, company: str | None):
        """
        Initializes vector retriever using the unified client runtime pool.
        Converts the entire user query string into an embedding and calculates
        cosine/Euclidean distance to documents.

        Note:
        You already know the required metadata filters in advance
        programmatically, OR you are doing broad context searching where strict
        metadata filtering isn't needed.
        """

        params = {"k": CHROMA_RESULT_CNT}
        if company:
            params["filter"] = {"company": company.lower()}

        return self.vector_storage.as_retriever(
            search_type="similarity",
            search_kwargs=params,
        )

    def _get_semantic_storage(self) -> Chroma:
        if self.emedding_model is None:
            raise ValueError(
                "🚨 An embedding model must be provided in dataset!"
            )

        return Chroma(
            client=self._client,
            embedding_function=self.embedding_model,
            collection_name="semantic_chunks",
        )

    def _get_vector_storage(self) -> Chroma:
        """Instantiates the primary flattened synthetic layout collection."""
        return Chroma(
            client=self._client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )

    def _get_semantic_text_splitter(self) -> SemanticChunker:
        return SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=SEMANTIC_THRESH_LIMIT,
        )

    def _get_structured_retriever(self, content: str, metadata_fields: list):
        # Use this for global search if company is unavailable
        """
        Uses an LLM call first to parse the query into a structured filter + clean
        search query, then queries Chroma.

        Note:
        The incoming query is a natural language string containing metadata
        conditions embedded in plain text, OR when the input field company is
        None / ambiguous.
        """
        if self.model is None:
            raise ValueError("🚨 A model is needed for global search.")

        return SelfQueryRetriever.from_llm(
            llm=self.model,
            vectorstore=self.semantic_storage,
            document_contents=content,  # @TODO - come back
            metadata_field_info=metadata_fields,
            structured_query_translator=ChromaTranslator(),
            verbose=True,
            use_original_query=True,
        )

    # @TODO - defunct, now using doc_handler.process()
    def get_semantic_chunks(self) -> list:
        raw_documents = []  # @todo - created documents
        return self.semantic_text_splitter.split_documents(raw_documents)

    def get_semantic_count(self) -> int:
        """Returns the number of documents in the semantic storage collection."""
        try:
            return self.semantic_storage._collection.count()
        except Exception:
            return 0

    def add_semantic_documents(self, semantic_chunks: list) -> None:
        batch_size = DOCUMENT_CHUNK_SIZE
        semantic_chunks_cnt = len(semantic_chunks)

        print(
            f"\n# --- ➕ Adding {semantic_chunks_cnt} semantic documents with a batch size of {batch_size} ➕ --- #"
        )

        for i in range(0, semantic_chunks_cnt, batch_size):
            self.semantic_storage.add_documents(
                semantic_chunks[i : i + batch_size]
            )

    def add_vector_documents(self, documents: list) -> None:
        batch_size = DOCUMENT_CHUNK_SIZE
        document_cnt = len(documents)

        print(
            f"\n# --- ➕ Adding {document_cnt} vector documents with a batch size of {batch_size}. ➕ --- #"
        )

        for i in range(0, document_cnt, batch_size):
            self.vector_storage.add_documents(documents[i : i + batch_size])

    def get_documents(self, query_str: str) -> list:
        """
        @TODO - should this be moved to the document_handler()?
        🎯 SIMPLIFIED FLAT RETRIEVAL:
        Queries the flattened target collection directly. Because data chunks are bundled together
        at ingestion, a single hit returns questions and original context with zero cross-collection lookup logic.
        """
        return self.vector_storage.similarity_search(
            query=query_str,
            k=CHROMA_RESULT_CNT,
        )

    def get_document_count(self) -> int:
        """Returns the number of documents currently in this collection."""
        try:
            return self.vector_storage._collection.count()
        except Exception:
            return 0

    def query(self, text: str, company: str | None) -> List[Document]:
        return self.retriever.invoke(text)

    def delete(self) -> None:
        """
        Aggressively purges only the target database directory (db/)
        to force a true factory reset of ChromaDB states.
        """
        target_db_dir = os.path.abspath(CHROMA_DB_DIR)

        print(f"🧹 Wiping Database directory: {target_db_dir}...")

        if not os.path.exists(target_db_dir):
            print(
                f"⚠️ Warning: Directory {target_db_dir} does not exist.  Creating a fresh one now..."
            )
            os.makedirs(target_db_dir, exist_ok=True)
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            return None

        # Loop through the children of db/ specifically leacing data completely alone
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
                    f"🚨 Failed to wipe element path target {filepath}.  Exception: {e}"
                )

        if next(os.scandir(target_db_dir), None) is None:
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            print(
                f"📁  Database directory {CHROMA_DB_DIR} is empty and ready for use."
            )
            print(
                f"🖊️ {CHROMA_DB_DIR} privileges are set to {DOCUMENT_DIR_PERM}.\n"
            )
