# src/document_handler.py
# +---------------------------------------------------------------------------+
# |                         DOCUMENT HANDLER                                  |
# +---------------------------------------------------------------------------+

# Python Libraries
import html
import re

import frontmatter
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

# Local Libraries
from src.constants import (  # Document data structures
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
)
from src.metadata_extractor import MetadataExtractor
from src.utils import log_chat_transcript


class DocumentHandler:
    def __init__(self, md_files: dict):
        self._md_files = md_files
        self._documents = []
        self._chunks = []

    def _create(self, content: str, metadata: dict) -> Document:
        return Document(
            id=metadata.get("id"),
            type="Document",
            page_content=self._format_html(content),
            metadata=metadata,
        )

    def process(self) -> list:

        if len(self._md_files) == 0:
            log_chat_transcript(
                "DOCUMENT_HANDLER", "⚠️ No markdown files present."
            )
            return []

        meta = MetadataExtractor()

        # Convert raw text into Langchain document objects
        documents = []
        file_order = 1
        for company, company_list in self._md_files.items():
            for company_file_order, company_dict in enumerate(company_list):
                # Get metadata for document creation and  Decode HTML.
                content = company_dict.get("content") or ""
                content = self._format_html(content)
                metadata = meta.extract(
                    company, company_file_order + 1, file_order, company_dict
                )
                """
                log_chat_transcript(
                    (
                        "DOCUMENT_HANDLER: METADATA",
                        f"{file_order}: {company}-{company_file_order + 1}",
                    ),
                    metadata,
                )
                """

                # Strip YAML frontmatter — MetadataExtractor already
                # captured it as structured metadata above, so leaving it
                # in page_content only produces a frontmatter-only chunk
                # with no article body (chunk_idx 0 for every file).
                body = frontmatter.loads(content).content

                # Create Document
                document = self._create(body, metadata)
                documents.append(document)
                file_order += 1

        # Configure text splitters
        self._chunks = self._create_chunks(documents)
        self._documents = documents

        return self._chunks

    def _format_html(self, page_content: str) -> str:
        decoded = html.unescape(page_content.strip())

        # Collapse multiple blank lines / empty <br /> tags to save tokens
        cleaned = re.sub(r"(<br\s*/?>\s*)+", "<br />\n", decoded)
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)

        return cleaned.strip()

    def _create_chunks(self, documents: list) -> list:
        """
        Process all markdown files for each company
        """
        # Define headers (<h1>, <h2>, <h3>) to split on and track in metadata
        headers_md = [
            ("#", "Title"),
            ("##", "Section"),
            ("###", "Subsection"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_md, strip_headers=False
        )

        # Secondary splitter for long sections that exceed chunk limit
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DOCUMENT_CHUNK_SIZE,
            chunk_overlap=DOCUMENT_CHUNK_OVERLAP,
        )

        # Split each document and assign chunk index AFTER splitting.
        chunks = []
        seen_checksum = set()
        for document in documents:
            # Check to see if the file has already been ingested before we
            # add another chunk
            checksum = document.metadata.get("checksum")
            if checksum in seen_checksum:
                log_chat_transcript(
                    "DOCUMENT_HANDLER",
                    f"Checksum: {checksum} found in sub_splits.  Skipping...",
                )
                continue

            # Decode HTML in page content.
            content = self._format_html(document.page_content)

            # Split by markdown structure
            header_splits = markdown_splitter.split_text(content)

            # Sub-split long markdown sections while preserving metadata
            sub_splits = recursive_splitter.split_documents(header_splits)

            # Add parent metadata AND sequential chunk index to every chunk
            for idx, chunk in enumerate(sub_splits):
                chunk.metadata.update(document.metadata)
                chunk.metadata["chunk_idx"] = idx
                chunks.append(chunk)

            if checksum:
                seen_checksum.add(checksum)

        return chunks

    def count_chunks(self) -> int:
        return len(self._chunks)

    def count_documents(self) -> int:
        return len(self._documents)

    def show(self, file_order: int) -> None:
        docs = self._documents

        # Only show 1 document by it's ID
        if file_order:
            doc = next(
                (
                    d
                    for d in docs
                    if d.metadata.get("file_order") == file_order
                ),
                None,
            )

            if doc is None or file_order == 0:
                message = f"⚠️  No document found by file order:{file_order}."
                log_chat_transcript("DOCUMENT_HANDLE: PROFILE", message)
                return None

            docs = [doc]
            # print("\n\t 📄 Document:  ----")

        # Display if no documents to show...
        if not docs:
            message = "⚠️ No documents to show."
            log_chat_transcript("DOCUMENT_HANDLE: PROFILE", message)
            return None

        # This will display all documents or one particular one by file_order
        document_body = ""
        document_body += f"\n# --- 🗃️ Showing {len(docs)} Documents 🗃️ --- #\n"
        for i, doc in enumerate(docs):
            document_body += f"\n\t---- 📄 Document: {i + 1} ----\n"

            for key, value in doc.metadata.items():
                document_body += (
                    f"\n\t{key.title().replace('_', ' ')}: {value}"
                )

            document_body += f"\n\t+--- Document: {i + 1} ---+\n"

            log_chat_transcript("DOCUMENT_HANDLE: PROFILE", document_body)

        print("\n")
