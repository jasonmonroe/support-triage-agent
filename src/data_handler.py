# src/data_handler.py
# +---------------------------------------------------------------------------+
# |                            DATA HANDLER                                     |
# +---------------------------------------------------------------------------+

# Vendor Libraries
import pandas as pd

# Local Libraries
from src.constants import OUTPUT_FILE, SAMPLE_SUPPORT_TICKETS_FILE


class DataHandler:
    """
    A class to handle data operations.
    """

    def __init__(self, dataset: dict) :
        """
        Initializes the DataHandler instance.
        """
        self.output = None
        self.support_tickets = None

        self._load_data(dataset)

    def _load_data(self, dataset: dict) -> None:
        """
        Loads data from a specified file path.

        Args:
            file_path (str): The path to the data file.
        """

        self.output = pd.read_csv(OUTPUT_FILE)
        self.support_tickets = pd.read_csv(SUPPORT_TICKETS_FILE)

        if dataset.get("sample"):
            self.support_tickets = pd.read_csv(SAMPLE_SUPPORT_TICKETS_FILE)