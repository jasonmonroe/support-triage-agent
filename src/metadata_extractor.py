# src/metadata_parser.py
# +---------------------------------------------------------------------------+
# |                         METADATA PARSER                                   |
# +---------------------------------------------------------------------------+
# https://reference.langchain.com/python/langchain-core/documents/base/Document

# Python Libraries
import hashlib
import json
import os
import platform
import uuid
from datetime import datetime, timezone

import frontmatter

# Local Libraries
from src.constants import DOCUMENT_TYPE
from src.utils import format_iso_date, log_chat_transcript


class MetadataExtractor:
    def __init__(self) -> None:

        self.id = None
        self.article_id = None
        self.article_slug = None
        self.breadcrumbs = None
        self.company = None
        self.company_file_order = None
        self.created_at = None  # File timestamp
        self.doc_title = None
        self.doc_type = None
        self.file_order = None
        self.last_updated_exact = None
        self.last_updated_iso = None  # if not found use last_updated_exact
        self.product_area = None
        self.source = None  # full filepath
        self.source_url = None
        self.title_slug = None
        self.utc_datetime = None  # current timestamp

        self._content = ""

    def _get_checksum(self) -> str:
        """Generates a stable data hash, safely selecting the correct text payload

        based on document types to prevent data indexing collisions.
        """
        excluded_keys = {
            "company_file_order",
            "created_at",
            "file_order",
            "id",
            "last_updated_exact",
            "last_updated_iso",
            "utc_datetime",
        }

        # Strips out internal underscores correctly to stabilize JSON
        # serialization keys
        filtered_metadata = {
            key.lstrip("_"): value
            for key, value in self.__dict__.items()
            if key not in excluded_keys
            and key.lstrip("_") not in excluded_keys
            and "date" not in key.lower()
        }

        # 🎯 Route the text hashing target based on document hierarchy
        filtered_metadata["content"] = (
            self._content.strip() if self._content else ""
        )

        metadata_json = json.dumps(
            filtered_metadata, sort_keys=True, separators=(",", ":")
        )

        return hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()

    def _get_created_at(self, filepath: str) -> datetime | None:
        try:
            file_info = os.stat(filepath)

            timestamp = (
                file_info.st_ctime
                if platform.system() == "Windows"
                else getattr(file_info, "st_birthtime", file_info.st_mtime)
            )

            return datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        except Exception as e:
            log_chat_transcript("METADATA_CREATED_AT", e)
            return None

    def _export(self) -> dict:
        """
        Exclude protected attributes (starting with '_') and any None values,
        except for 'product_area' which must always be included.
        """

        excluded_attrs = {"_content"}  # start with explicitly excluded keys
        for key, value in self.__dict__.items():
            # Always keep product_area, even if None
            if key == "product_area":
                continue

            # Exclude private/protected attributes
            if key.startswith("_"):
                excluded_attrs.add(key)
                continue

            # Exclude attributes with None or other falsey values (empty string, 0, etc.)
            if not value:
                excluded_attrs.add(key)

        exported_dicts = {
            key: value
            for key, value in self.__dict__.items()
            if key not in excluded_attrs
        }

        return dict(sorted(exported_dicts.items()))

    def extract(
        self,
        company: str,
        company_file_order: int,
        file_order: int,
        dataset: dict,
    ) -> dict:
        self._content = dataset.get("content").strip()
        content = frontmatter.loads(self._content)

        self.id = uuid.uuid4().hex[:32].lower()
        self.article_id = (content.get("article_id") or "").strip() or None
        self.article_slug = (content.get("article_slug") or "").strip() or None
        self.breadcrumbs = content.get("breadcrumbs")
        self.company = company
        self.company_file_order = (
            company_file_order  # orderable of file by company
        )
        self.doc_title = content.get("title")
        self.doc_type = DOCUMENT_TYPE
        self.file_order = file_order  # orderable of file in data directory
        self.last_updated_exact = (
            str(
                format_iso_date(
                    (content.get("last_updated_exact") or "").strip()
                ).isoformat()
            )
            if content.get("last_updated_exact")
            else None
        )
        self.last_updated_iso = (
            str(
                format_iso_date(
                    (content.get("last_updated_iso") or "").strip()
                ).isoformat()
            )
            if content.get("last_updated_iso")
            else None
        )

        self.product_area = dataset.get("product_area")
        self.source = str(dataset.get("filepath"))
        self.source_url = content.get("source_url")
        self.title_slug = content.get("title_slug")
        self.created_at = str(self._get_created_at(self.source))
        self.utc_datetime = str(datetime.now(timezone.utc).isoformat())

        self.checksum = self._get_checksum()

        return self._export()
