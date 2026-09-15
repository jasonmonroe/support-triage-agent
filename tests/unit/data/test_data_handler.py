# tests/unit/data/test_data_handler.py
# +---------------------------------------------------------------------------+
# |                         DATA HANDLER TEST                                 |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/data/test_data_handler.py -v

# Python Libraries
import os

# Vendor Libraries
import pandas as pd
import pytest

# Local Libraries
from src.data_handler import DataHandler

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]

TICKETS_HEADER = (
    "Issue,Subject,Company,Response,Product Area,Status,Request Type\n"
)


@pytest.fixture
def output_csv(tmp_path, monkeypatch):
    """DataHandler._load_data() unconditionally reads OUTPUT_FILE, so every
    test needs a valid (if empty) one on disk before DataHandler() can even
    be constructed."""
    path = tmp_path / "output.csv"
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)
    monkeypatch.setattr("src.data_handler.OUTPUT_FILE", str(path))
    return path


@pytest.fixture
def tickets_csv(tmp_path, monkeypatch, output_csv):
    """Deliberately messy: leading/trailing whitespace and a blank cell,
    mirroring the real sample CSV's quirks that _clean_data() exists to fix."""
    path = tmp_path / "tickets.csv"
    row1 = (
        "  I forgot my password  ,Password reset,HackerRank,,"
        "account_access,,\n"
    )
    row2 = "Site is down,,None,,,Escalated,bug\n"
    path.write_text(TICKETS_HEADER + row1 + row2)
    monkeypatch.setattr(
        "src.data_handler.SAMPLE_SUPPORT_TICKETS_FILE", str(path)
    )
    monkeypatch.setattr("src.data_handler.SUPPORT_TICKETS_FILE", str(path))
    return path


@pytest.fixture
def rag_dirs(tmp_path, monkeypatch, tickets_csv):
    """Builds a tiny data/{claude,hackerrank,visa}/ tree — including an
    index.md (a per-company table of contents, not a real article, that
    _load_md_files() must skip) and a nested subdirectory (to exercise
    product_area extraction from the folder path)."""
    data_dir = str(tmp_path / "data") + "/"
    claude_dir = os.path.join(data_dir, "claude")
    hackerrank_dir = os.path.join(data_dir, "hackerrank")
    visa_dir = os.path.join(data_dir, "visa")

    billing_dir = os.path.join(claude_dir, "billing")
    os.makedirs(billing_dir)
    os.makedirs(hackerrank_dir)
    os.makedirs(visa_dir)

    with open(os.path.join(claude_dir, "index.md"), "w") as f:
        f.write("table of contents")
    with open(os.path.join(claude_dir, "getting-started.md"), "w") as f:
        f.write("# Getting Started\nSome content.")
    with open(os.path.join(billing_dir, "refunds.md"), "w") as f:
        f.write("# Refunds\nHow refunds work.")
    with open(os.path.join(hackerrank_dir, "faq.md"), "w") as f:
        f.write("# FAQ\nSome content.")
    with open(os.path.join(visa_dir, "support.md"), "w") as f:
        f.write("# Support\nSome content.")

    monkeypatch.setattr("src.data_handler.DATA_DIR", data_dir)
    monkeypatch.setattr("src.data_handler.CLAUDE_DIR", claude_dir)
    monkeypatch.setattr("src.data_handler.HACKERRANK_DIR", hackerrank_dir)
    monkeypatch.setattr("src.data_handler.VISA_DIR", visa_dir)


def test_load_data_reads_support_tickets_as_dataframe(tickets_csv):
    handler = DataHandler({"sample": True})

    assert isinstance(handler.support_tickets, pd.DataFrame)
    assert len(handler.support_tickets) == 2


def test_clean_data_normalizes_spaced_column_names_to_underscores(tickets_csv):
    """itertuples() needs valid identifiers — "Product Area" must become
    "Product_Area", not stay as-is (which itertuples() would otherwise
    silently rename to a positional "_5")."""
    handler = DataHandler({"sample": True})

    assert "Product_Area" in handler.support_tickets.columns
    assert "Product Area" not in handler.support_tickets.columns


def test_clean_data_strips_whitespace_from_text_cells(tickets_csv):
    handler = DataHandler({"sample": True})

    assert handler.support_tickets.iloc[0]["Issue"] == "I forgot my password"


def test_clean_data_fills_missing_values_with_empty_string(tickets_csv):
    handler = DataHandler({"sample": True})

    assert handler.support_tickets.iloc[1]["Subject"] == ""


def test_sample_flag_selects_sample_file_over_full_file(
    tmp_path, monkeypatch, output_csv
):
    sample_path = tmp_path / "sample.csv"
    full_path = tmp_path / "full.csv"
    sample_path.write_text(TICKETS_HEADER + "sample issue,,None,,,,\n")
    full_path.write_text(
        TICKETS_HEADER + "full issue 1,,None,,,,\nfull issue 2,,None,,,,\n"
    )
    monkeypatch.setattr(
        "src.data_handler.SAMPLE_SUPPORT_TICKETS_FILE", str(sample_path)
    )
    monkeypatch.setattr(
        "src.data_handler.SUPPORT_TICKETS_FILE", str(full_path)
    )

    sample_handler = DataHandler({"sample": True})
    full_handler = DataHandler({"sample": False})

    assert len(sample_handler.support_tickets) == 1
    assert len(full_handler.support_tickets) == 2


def test_rag_flag_not_set_leaves_md_files_empty(tickets_csv):
    handler = DataHandler({"sample": True})

    assert handler.md_files == {"claude": [], "hackerrank": [], "visa": []}
    assert handler.md_file_cnt == 0


def test_rag_flag_loads_markdown_files_per_company_and_skips_index(rag_dirs):
    handler = DataHandler({"sample": True, "rag": True})

    claude_files = {f["filename"] for f in handler.md_files["claude"]}
    assert "getting-started.md" in claude_files
    assert "refunds.md" in claude_files
    assert "index.md" not in claude_files  # a TOC, not an article
    assert len(handler.md_files["hackerrank"]) == 1
    assert len(handler.md_files["visa"]) == 1
    assert handler.md_file_cnt == (
        len(handler.md_files["claude"])
        + len(handler.md_files["hackerrank"])
        + len(handler.md_files["visa"])
    )


def test_rag_flag_extracts_product_area_from_subdirectory(rag_dirs):
    handler = DataHandler({"sample": True, "rag": True})
    by_name = {f["filename"]: f for f in handler.md_files["claude"]}

    assert by_name["getting-started.md"]["product_area"] is None
    assert by_name["refunds.md"]["product_area"] == "billing"
    assert "# Refunds" in by_name["refunds.md"]["content"]


def test_save_data_writes_list_of_dicts_to_output_csv(tickets_csv, output_csv):
    handler = DataHandler({"sample": True})

    handler.save_data(
        [
            {"issue": "a", "status": "Replied"},
            {"issue": "b", "status": "Escalated"},
        ]
    )

    written = pd.read_csv(output_csv)
    assert list(written["issue"]) == ["a", "b"]
    assert list(written["status"]) == ["Replied", "Escalated"]


def test_save_data_writes_dataframe_directly_to_output_csv(
    tickets_csv, output_csv
):
    handler = DataHandler({"sample": True})
    df = pd.DataFrame([{"issue": "a", "status": "Replied"}])

    handler.save_data(df)

    written = pd.read_csv(output_csv)
    assert list(written["issue"]) == ["a"]
