# src/data_handler.py
# +---------------------------------------------------------------------------+
# |                            DATA HANDLER                                   |
# +---------------------------------------------------------------------------+

# Python Libraries
from pathlib import Path

# Vendor Libraries
import pandas as pd

# Local Libraries
from src.constants import (
    CLAUDE_DIR,
    HACKERRANK_DIR,
    OUTPUT_FILE,
    SAMPLE_SUPPORT_TICKETS_FILE,
    SUPPORT_TICKETS_FILE,
    VISA_DIR,
)


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
        self.data = {"claude": [], "hackerrank": [], "visa": []}

        self._load_data(args.get("sample", False))

        if args.get("eda", False):
            self._describe_data()

        # Load data files
        self.data["claude"] = self._load_md_files(CLAUDE_DIR)
        self.data["hackerrank"] = self._load_md_files(HACKERRANK_DIR)
        self.data["visa"] = self._load_md_files(VISA_DIR)

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

    """
    def _load_claude_md_files(self):
        Create a list of dicts with the filename as the key and the string
        content as the value.
        322 files for claude.
        Example
        data/claude/claude-desktop/desktop-extensions/10949351-getting-started-with-local-mcp-servers-on-claude-desktop.md
        maps to -> "claude" => [
            "10949351-getting-started-with-local-mcp-servers-on-claude-desktop": "_____"
        ]

        claude_md_files = []
        root_dir = Path(CLAUDE_DIR)

        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                filename = Path(file_path)
                key = filename.stem
                file_content = filename.read_text(encoding="utf-8")
                file_dict = {key: file_content}
                claude_md_files.append(file_dict)

        self.data["claude"] = claude_md_files

    def _load_hackerrank_md_files(self):
        hackerrank_md_files = []
        root_dir = Path(HACKERRANK_DIR)

        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                filename = Path(file_path)
                key = filename.stem
                file_content = filename.read_text(encoding="utf-8")
                file_dict = {key: file_content}
                hackerrank_md_files.append(file_dict)

        self.data["hackerrank"] = hackerrank_md_files

    def _load_visa_md_files(self):
        visa_md_files = []
        root_dir = Path(HACKERRANK_DIR)
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                filename = Path(file_path)
                key = filename.stem
                file_content = filename.read_text(encoding="utf-8")
                file_dict = {key: file_content}
                visa_md_files.append(file_dict)

        self.data["visa"] = visa_md_files
    """

    def _load_md_files(self, dir) -> list:
        root_dir = Path(dir)
        md_files = []

        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                filename = Path(file_path)
                file_key = filename.stem
                file_content = filename.read_text(encoding="utf-8")
                file_dict = {file_key: file_content}

                md_files.append(file_dict)

        return md_files

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
        csv_data.to_csv(OUTPUT_FILE, index=False)
