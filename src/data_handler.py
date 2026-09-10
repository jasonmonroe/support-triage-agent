# src/data_handler.py
# +---------------------------------------------------------------------------+
# |                            DATA HANDLER                                   |
# +---------------------------------------------------------------------------+

# Python Libraries
import os
from pathlib import Path

# Vendor Libraries
import pandas as pd

# Local Libraries
from src.constants import (
    CLAUDE_DIR,
    DATA_DIR,
    HACKERRANK_DIR,
    OUTPUT_FILE,
    SAMPLE_SUPPORT_TICKETS_FILE,
    SUPPORT_TICKETS_FILE,
    VISA_DIR,
)
from src.utils import log_chat_transcript


class DataHandler:
    """
    A class to handle data operations.
    """

    def __init__(self, args: dict):
        """
        Initializes the DataHandler instance.
        """
        self.output = None
        self.support_tickets = None
        self.md_files = {"claude": [], "hackerrank": [], "visa": []}

        self._load_data(args.get("sample", False))

        # Load data files
        if args.get("rag"):
            # self._compact_documents(DATA_DIR)
            self.md_files["claude"] = self._load_md_files(CLAUDE_DIR)
            self.md_files["hackerrank"] = self._load_md_files(HACKERRANK_DIR)
            self.md_files["visa"] = self._load_md_files(VISA_DIR)
            log_chat_transcript("MARKDOWN_FILES", self.md_files)

        if args.get("eda", False):
            self._describe_data()

    def _load_data(self, use_sample: bool) -> None:
        """
        Loads data from a specified file path.

        Args:
            file_path (str): The path to the data file.
        """

        self.output = pd.read_csv(OUTPUT_FILE)

        file_path = (
            SAMPLE_SUPPORT_TICKETS_FILE if use_sample else SUPPORT_TICKETS_FILE
        )

        self.support_tickets = pd.read_csv(file_path)
        # itertuples() builds namedtuples, which require valid Python
        # identifiers — a raw "Product Area" column gets silently renamed
        # to a positional "_5" instead. Normalize spaces to underscores
        # up front so every column survives itertuples() with its real name.
        self.support_tickets.columns = self.support_tickets.columns.str.replace(
            " ", "_"
        )

    def _compact_md_files(self, dir) -> None:
        """This version compacts all markdown files together"""
        mds_dir = Path(dir)
        md_files = list(mds_dir.rglob("*.md"))
        self.md_files = [p.read_text(encoding="utf-8") for p in md_files]
        log_chat_transcript("COMPACT_MARKDOWN_FILES", self.md_files)

    def _load_md_files_dbg(self, dir) -> list:
        root_dir = Path(dir)
        print(
            "root_dir:",
            root_dir,
            "exists:",
            root_dir.exists(),
            "is_dir:",
            root_dir.is_dir(),
        )

        target = dir.replace(DATA_DIR, "")
        print("target:", repr(target))

        md_files = []
        for file_path in root_dir.rglob("*"):
            print("found:", file_path, "is_file:", file_path.is_file())
            if file_path.is_file():
                file_dict = self._parse_md_file(target, file_path)
                md_files.append(file_dict)

        print("total files:", len(md_files))
        return md_files

    def _load_md_files(self, dir) -> list:
        root_dir = Path(dir)
        target = dir.replace(DATA_DIR, "")

        md_files = []
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                file_dict = self._parse_md_file(target, file_path)
                md_files.append(file_dict)

        return md_files

    def _parse_md_file(self, target: str, filepath: Path) -> dict:
        path = Path(filepath)
        company_dir = str(os.path.join(DATA_DIR, target))

        # Check if it has a product area
        product_area = None
        parent_dir = path.parent
        file_parent_dir = str(parent_dir).replace(company_dir, "")
        product_area = file_parent_dir.lstrip("/").rstrip("/") or None

        # Use the text value as content
        file_content = (
            path.read_text(encoding="utf-8").strip()
            if path.is_file()
            else None
        )

        return {
            "product_area": product_area,
            "filepath": path,
            "filename": path.name,
            "content": file_content,
        }

    def _describe_data(self) -> None:
        """
        Describes the data.
        """
        print("\n# --- 📚 Data Description 📚 --- #")

        print(f"Number of rows: {len(self.support_tickets)}")
        print(f"Number of columns: {len(self.support_tickets.columns)}")
        print(f"Columns: {self.support_tickets.columns.tolist()}")
        print(f"Data types: {self.support_tickets.dtypes.to_dict()}")
        print(
            f"Missing values: {self.support_tickets.isnull().sum().to_dict()}"
        )
        print(f"Unique values: {self.support_tickets.nunique().to_dict()}")
        print(f"Value counts: {self.support_tickets.value_counts().to_dict()}")
        print(
            f"Descriptive statistics: {self.support_tickets.describe().to_dict()}"
        )

        print("\n# --- Data Head --- #")
        print(self.support_tickets.head())

        print("\n# --- Data Info --- #")
        print(self.support_tickets.info())

        print("\n# --- Data Describe --- #")
        print(self.support_tickets.describe())

        print("\n# --- Data Columns --- #")
        print(self.support_tickets.columns)

        print("\n# --- Data Index --- #")
        print(self.support_tickets.index)

        print("\n# --- Data Values --- #")
        print(self.support_tickets.values)

        print("\n# --- Data Shape --- #")
        print(self.support_tickets.shape)

        print("\n# --- Data Dtypes --- #")
        print(self.support_tickets.dtypes)

    def save_data(self, csv_data) -> None:
        """
        Saves the data.
        """

        print(f"--- 💾 Saving data to {OUTPUT_FILE}.")
        csv_data.to_csv(OUTPUT_FILE, index=False)
