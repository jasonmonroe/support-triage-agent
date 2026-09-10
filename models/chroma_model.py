from __future__ import annotations

import os
import shutil
import time
from typing import List

# models/chroma_model.py
# +---------------------------------------------------------------------------+
# |                            CHROMA DB MODEL                                |
# +---------------------------------------------------------------------------+
# Python Libraries
import chromadb

# Vector Libraries
from google.genai.errors import ClientError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

# Local Libraries
from src.constants import (
    CHROMA_COLL_NAME,
    CHROMA_DB_DIR,
    CHROMA_RESULT_CNT,
    CHROMA_SERVER_NO_TELEMETRY,
    DB_BATCH_SIZE,
    DOCUMENT_DIR_PERM,
    EMBED_PAUSE_TIMER,
    MODEL_API_KEY,
    MODEL_EMBEDDING,
    RATE_LIMIT_PAUSE_TIMER,
    RATE_LIMIT_RETRIES,
)
from src.utils import log_chat_transcript


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
        print(MODEL_EMBEDDING)

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
            embedding_function=self._get_embeddings(),
            collection_name=self.collection_name,
        )

    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:

        # embeddings = OpenAIEmbeddings(
        # model=self.embedding_model,  # e.g. "gemini-embedding-001"
        # openai_api_key=MODEL_API_KEY,
        # openai_api_base=MODEL_API_URL,  # "https://generativelanguage.googleapis.com/v1beta/openai/"
        # check_embedding_ctx_length=False,  # Disables tiktoken token counting
        # )

        return GoogleGenerativeAIEmbeddings(
            model=f"models/{self.embedding_model}",  # e.g., "models/text-embedding-004" or "models/gemini-embedding-001"
            google_api_key=MODEL_API_KEY,
        )

    def add_vector_documents(
        self, documents: list, batch_size: int = DB_BATCH_SIZE
    ) -> None:
        """
        Embeds and adds documents to the vector store in batches using explicit retry logic
        that parses vendor rate limit messages and backs off gracefully.
        """
        document_cnt = len(documents)

        print(
            f"\n# --- ➕ Adding {document_cnt} chunked vector documents with a batch size of {batch_size}. ➕ --- #"
        )

        self.vector_storage.reset_collection()

        total_batches = (document_cnt + batch_size - 1) // batch_size
        for i in range(0, document_cnt, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            print(
                f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)..."
            )
            success = self._add_batch_with_retry(batch, batch_index=batch_num)

            if not success:
                print(
                    f"🚨 Stopping ingestion due to unrecoverable API error at batch {batch_num}. 🚨"
                )
                break

            # Throttling pause to stay under RPM limit
            time.sleep(EMBED_PAUSE_TIMER)

    def _add_batch_with_retry(self, batch: list, batch_index: int = 0) -> bool:
        """
        Attempts to write a document batch to ChromaDB.
        Catches 429 RateLimit/ResourceExhausted errors, extracts delay times, and retries.
        """
        attempt = 0
        while attempt < RATE_LIMIT_RETRIES:
            try:
                self.vector_storage.add_documents(batch)
                return True

            except (ClientError, GoogleGenerativeAIError) as e:
                attempt += 1
                error_msg = str(e)
                log_chat_transcript(
                    "CHROMA_EMBED_RATE_LIMIT",
                    f"Batch {batch_index} Attempt {attempt} Error: {error_msg}",
                )

                if attempt >= RATE_LIMIT_RETRIES:
                    print(
                        f"\n🚨 Batch {batch_index} | Maximum retries ({RATE_LIMIT_RETRIES}) reached! Ingestion failed. 🚨"
                    )
                    return False

                delay_time = self._parse_delay_time(error_msg)
                print(
                    f"\n🚨 Batch {batch_index} | Rate limit / Quota exceeded (429) on attempt {attempt}/{RATE_LIMIT_RETRIES} 🚨"
                )
                print(
                    f"⏸️  Pausing for {delay_time:.2f} seconds before retry..."
                )

                time.sleep(delay_time)

            except Exception as e:
                print(
                    f"\n🚨 Batch {batch_index} | Unexpected error adding batch: {e} 🚨"
                )
                log_chat_transcript("CHROMA_EMBED_UNEXPECTED_ERROR", e)
                return False

        return False

    def _parse_delay_time(self, error_message: str) -> float:
        """
        Parses Google API error response text for vendor retry suggestions (e.g. 'please retry in X.Xs').
        """
        err = error_message.lower()
        anchor_str = "please retry in "
        end_char = "s"

        if anchor_str not in err:
            return float(RATE_LIMIT_PAUSE_TIMER * 2)

        try:
            start_pos = err.find(anchor_str) + len(anchor_str)
            end_pos = err.find(end_char, start_pos)
            delay_str = err[start_pos:end_pos].strip()

            # Remove any trailing non-numeric characters if needed
            delay_time = float(
                "".join(c for c in delay_str if c.isdigit() or c == ".")
            )

            # Cap maximum pause delay to prevent infinitely hanging scripts
            return min(max(delay_time + 1.0, 2.0), 60.0)
        except Exception as e:
            new_timer = float(RATE_LIMIT_PAUSE_TIMER * 2)

            log_chat_transcript(
                "RATE_LIMIT_ERROR", f"{e}\nreturning {new_timer}"
            )
            return new_timer

    def get_documents(self, query_str: str) -> list:
        """Queries the vector storage collection directly using similarity search."""
        return self.vector_storage.similarity_search(
            query=query_str,
            k=CHROMA_RESULT_CNT,
        )

    def get_collection_count(self) -> int:
        """Returns the number of documents currently in the vector collection."""
        try:
            return self.vector_storage._collection.count()
        except Exception as e:
            log_chat_transcript("ERROR COLLECTION_COUNT", e)
            return 0

    def query(self, company: str | None, text: str) -> List[Document]:
        """Queries ChromaDB using company-filtered or global search."""
        retriever = self.get_retriever(company)
        return retriever.invoke(text)

    def delete_collection(self):
        log_chat_transcript(
            "DELETE_COLLECTION", "🗑️ Deleting vector collection."
        )
        self.vector_storage.reset_collection()

    def delete(self) -> None:
        """Purges the target database directory (db/) to reset ChromaDB states."""
        target_db_dir = os.path.abspath(CHROMA_DB_DIR)

        log_chat_transcript(
            "DELETE CHROMA_DB DIR",
            f"🗑️ Wiping Database directory: {target_db_dir}...",
        )

        if not os.path.exists(target_db_dir):
            log_chat_transcript(
                "DELETE CHROMA_DB DIR",
                f"⚠️ Warning: Directory {target_db_dir} does not exist. Creating a fresh one now...",
            )
            os.makedirs(target_db_dir, exist_ok=True)
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            return None

        for filename in os.listdir(target_db_dir):
            filepath = os.path.join(target_db_dir, filename)
            try:
                log_chat_transcript(
                    "DELETE CHROMA_DB DIR",
                    f"Purging database artifact: {filepath}...",
                )

                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)

            except FileNotFoundError as e:
                log_chat_transcript(
                    "DELETE CHROMA_DB DIR",
                    f"🚨 Failed to wipe element path target {filepath}. Exception: {e}",
                )

        if next(os.scandir(target_db_dir), None) is None:
            os.chmod(target_db_dir, DOCUMENT_DIR_PERM)
            log_chat_transcript(
                "DELETE CHROMA_DB DIR",
                f"📁 Database directory {CHROMA_DB_DIR} is empty and ready for use.",
            )
            log_chat_transcript(
                "DELETE CHROMA_DB DIR",
                f"🖊️ {CHROMA_DB_DIR} privileges are set to {DOCUMENT_DIR_PERM}.\n",
            )
