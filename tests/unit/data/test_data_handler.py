# tests/unit/data/test_data_handler.py
# +---------------------------------------------------------------------------+
# |                         DATA HANDLER TEST                                 |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/data/test_data_handler.py -v

# Python Libraries
from pathlib import Path

# Vendor Libraries
import pandas as pd
import pytest

# Local Libraries
from ml_app.data.data_handler import load_csv


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "Issue,Subject,Company,Response,Product Area,Status,Request Type\n"
        "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.,Test Active in the system,HackerRank,'Hi, Tests in HackerRank remain active indefinitely unless a start and end time are set. Without these, tests do not expire automatically. To set expiration times, specify a start and end date/time in the test settings. After expiration: Invited candidates cannot access the test. The \"Invite\" button is disabled; no new invitations can be sent. To check or change expiration settings: Go to the test's Settings and select the General section. Update the Start date & time and End date & time fields as needed. To keep the test active indefinitely, clear these fields by clicking the clear icon (X). If the test has an expiration set, adjust these settings to enable new invitations.',screen,Replied,product_issue\n"
        "site is down & none of the pages are accessible,,None,Escalate to a human,,Escalated,bug\n"
    )
    return csv_file


def test_it_loads_csv_returns_dataframe(sample_csv: Path) -> None:
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)


def test_if_csv_has_expected_columns(sample_csv: Path) -> None:
    df = load_csv(sample_csv)
    expected = [
        "Issue",
        "Subject",
        "Company",
        "Response",
        "Product Area",
        "Status",
        "Request Type",
    ]
    actual = list(df.columns)
    assert actual == expected


def test_csv_row_count(sample_csv: Path) -> None:
    df = load_csv(sample_csv)
    expected = 2  # two data rows in the fixture
    actual = len(df)
    assert actual == expected


def test_if_csv_has_data(sample_csv: Path) -> None:
    df = load_csv(sample_csv)
    assert (
        df.iloc[0]["Issue"]
        == "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system."
    )
    assert df.iloc[0]["Subject"] == "Test Active in the system"
    assert df.iloc[0]["Company"] == "HackerRank"


if __name__ == "__main__":
    pass
