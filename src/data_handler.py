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
from src.utils import log_chat_transcript, pretty_dict


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
        self._filepath = None

        # Load and clean support tickets
        self._load_data(args.get("sample", False))

        if args.get("eda", False):
            self._describe_data()

        self._clean_data()

        # Load data files
        if args.get("rag"):
            # self._compact_documents(DATA_DIR)
            self.md_files["claude"] = self._load_md_files(CLAUDE_DIR)
            self.md_files["hackerrank"] = self._load_md_files(HACKERRANK_DIR)
            self.md_files["visa"] = self._load_md_files(VISA_DIR)
            log_chat_transcript("MARKDOWN_FILES", self.md_files)

    def _load_data(self, use_sample: bool) -> None:
        """
        Loads data from a specified file path.

        Args:
            file_path (str): The path to the data file.
        """

        self.output = pd.read_csv(OUTPUT_FILE)

        filepath = (
            SAMPLE_SUPPORT_TICKETS_FILE if use_sample else SUPPORT_TICKETS_FILE
        )

        print(f"\n 📁 Loading {filepath}")

        self.support_tickets = pd.read_csv(filepath)
        self._filepath = filepath

    # @TODO - defunct
    def _compact_md_files(self, dir) -> None:
        """This version compacts all markdown files together"""
        mds_dir = Path(dir)
        md_files = list(mds_dir.rglob("*.md"))
        self.md_files = [p.read_text(encoding="utf-8") for p in md_files]
        log_chat_transcript("COMPACT_MARKDOWN_FILES", self.md_files)

    # @TODO - defunct
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

    def _clean_data(self):
        log_chat_transcript(
            "CLEANING_SUPPORT_TICKETS", f"🧹 Cleaning {self._filepath}..."
        )

        """
        Note: The csv file is sloppy with unnecessary spaces in the headers.
        Therefore we need to do some slight formatting to make it useable.
        itertuples() builds namedtuples, which require valid Python
        identifiers — a raw "Product Area" column gets silently renamed
        to a positional "_5" instead. Normalize spaces to underscores
        up front so every column survives itertuples() with its real name.
        """

        self.support_tickets.columns = (
            self.support_tickets.columns.str.replace(" ", "_")
        )

        df = self.support_tickets
        df.columns = df.columns.str.strip()

        # 2. Vectorized cleaning for text/object columns
        text_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in text_cols:
            # Fill missing values, ensure string representation, and strip whitespace in one pass
            df[col] = df[col].fillna("").astype(str).str.strip()

        self.support_tickets = df

    def _describe_data(self) -> None:
        """
        Describes the data.
        """
        print("\n# --- 📚 Data Description 📚 --- #")

        # --- Summary Print Statements ---
        print(f"Number of rows: {len(self.support_tickets)}")
        print(f"\n 🗂️ Number of columns: {len(self.support_tickets.columns)}")
        print(
            f"\n 🗂️ Columns:\n{pretty_dict(self.support_tickets.columns.tolist())}"
        )
        print(
            f"\n 🗂️ Data types:\n{pretty_dict(self.support_tickets.dtypes.to_dict())}"
        )
        print(
            f"\n 🗂️ Missing values:\n{pretty_dict(self.support_tickets.isnull().sum().to_dict())}"
        )
        print(
            f"\n 🗂️ Unique values:\n{pretty_dict(self.support_tickets.nunique().to_dict())}"
        )

        # Convert Tuples from value_counts() into string keys for JSON serialization
        val_counts_dict = {
            str(k): v
            for k, v in self.support_tickets.value_counts().to_dict().items()
        }
        print(f"\n 🗂️ Value counts:\n{pretty_dict(val_counts_dict)}")

        print(
            f"\n 🗂️ Descriptive statistics:\n{pretty_dict(self.support_tickets.describe(include='all').to_dict())}"
        )

        print("\n# --- Data Head --- #")
        print(self.support_tickets.head())

        print("\n# --- Data Info --- #")
        print(self.support_tickets.info())

        print("\n# --- Data Describe --- #")
        print(self.support_tickets.describe(include="all"))

        print("\n# --- Data Columns --- #")
        print(self.support_tickets.columns.tolist())

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

        log_chat_transcript(
            "SAVING OUTPUT ROWS", f"--- 💾 Saving data to {OUTPUT_FILE}."
        )
        csv_data.to_csv(OUTPUT_FILE, index=False)
