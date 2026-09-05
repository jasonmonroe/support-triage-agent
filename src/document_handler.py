# src/document_handler.py
# +---------------------------------------------------------------------------+
# |                         DOCUMENT HANDLER                                  |
# +---------------------------------------------------------------------------+

# Python Libraries

# Vendor Libraries
from langchain_classic.chains.query_constructor.schema import AttributeInfo
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

    def _create(self, content: str, metadata: dict):

        if "doc_type" not in metadata:
            metadata["doc_type"] = "semantic_chunk"

        return Document(
            id=metadata.get("id"),
            type="Document",
            page_content=content,
            metadata=metadata,
        )

    def process(self) -> list:
        meta = MetadataExtractor()

        # Convert raw text into Langchain document objects
        documents = []
        for company, company_list in self._md_files.items():
            for file_order, company_dict in enumerate(company_list):  # list
                # Get metadata for document creation
                metadata = meta.extract(company, file_order, company_dict)
                log_chat_transcript("DOCUMENT_METADATA", metadata)

                # Create Document
                document = self._create(company_dict.get("content"), metadata)
                documents.append(document)

        # Configure text splitters
        self._chunks = self._create_chunks(documents)
        self._documents = documents

        return self._chunks

    def _compact(self):
        """Compact all markdown content"""
        pass

    def _create_chunks(self, documents: list):
        """
        Process all markdown files for each company
        """
        # Define headers to split on and track in metadata
        headers_md = [
            ("#", "Title"),  # <h1>
            ("##", "Section"),  # <h2>
            ("###", "Subsection"),  # <h3>
            # ("####", "Detail"),  # <h4>
            # ("#####", "Note"),  # <h5>
            # ("######", "Meta"),  # <h6>
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_md, strip_headers=False
        )

        # Secondary splitter for long sections that exceed chunk limit
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DOCUMENT_CHUNK_SIZE,
            chunk_overlap=DOCUMENT_CHUNK_OVERLAP,
        )

        chunks = []

        # Split each document and assign chunk index AFTER splitting.
        for document in documents:
            # Split by markdown structure
            header_splits = markdown_splitter.split_text(document.page_content)

            # Sub-split long markdown sections while preserving metadata
            sub_splits = recursive_splitter.split_documents(header_splits)

            # Add parent metadata AND sequential chunk index to every chunk
            for idx, chunk in enumerate(sub_splits):
                chunk.metadata.update(document.metadata)
                chunk.metadata["chunk_idx"] = idx
                chunks.append(chunk)

        return chunks

    def embed(self):
        pass

    def create_collection(self):
        pass

    def get_semantic_chunks(self, semantic_chunks):
        """
        Maps continuous semantic raw chunks into formal wrapped LangChain
        Documents.
        Tracks a per-page sequence counter to guarantee unique text_ids when a
        single
        page is split into multiple semantic chunks.
        """
        document_chunks = []
        documents = []

        for doc in semantic_chunks:
            metadata = doc.metadata.copy()
            metadata["chunk_seq"] = None

            documents.append(self.create("", metadata))

        return document_chunks

    def count_chunks(self) -> int:
        return len(self._chunks)

    def show(self, id: str) -> None:

        docs = self._documents

        # Only show 1 document by it's ID
        if id:
            # random document
            # rand_doc = documents[?]
            # doc_id = rand_doc.metadata.get("id")
            doc = next(
                (d for d in docs if d.metadata.get("id") == id),
                None,
            )

            if doc is None:
                message = f"⚠️ No document found by ID:{id}."
                log_chat_transcript("DOCUMENT_PROFILE", message)
                return None

            docs = [doc]
            print("\n\t 📄 Document:  ----")

        if not docs:
            message = "⚠️ No documents to show."
            log_chat_transcript("DOCUMENT_PROFILE", message)
            return None

        print(f"\n# --- 🗃️ Showing {len(docs)} Documents 🗃️ --- #")
        for i, doc in enumerate(docs):
            print(f"\n\t 📄 Document: {i + 1} -----")
            print("\t\tSource:", doc.metadata.get("source", "Unknown"))
            print(
                "\t\tFilename:",
                doc.metadata.get("filename", "Unknown filename"),
            )
            print("\t\tPage:", doc.metadata.get("page", "Unknown"))
            print("\t\tPage Content:", doc.page_content)
            print(f"\t+-- Document: {i + 1} ----+")

    @staticmethod
    def metadata_field_info() -> list:
        return [
            AttributeInfo(
                name="company",
                description="Name of company with the ticket issue.",
                type="string",
            ),
            AttributeInfo(
                name="product_area",
                description="Company domain/category based on knowledge base.",
                type="string",
            ),
            AttributeInfo(
                name="source",
                description="Full filepath of the markdown file.",
                type="string",
            ),
            AttributeInfo(
                name="chunck_idx",
                description="Chunk idenfifier for the markdown file.",
                type="integer",
            ),
        ]
