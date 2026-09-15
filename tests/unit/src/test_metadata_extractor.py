# tests/unit/src/test_metadata_extractor.py
# +---------------------------------------------------------------------------+
# |                       METADATA EXTRACTOR TESTS                            |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/src/test_metadata_extractor.py -v

# Local Libraries
from src.metadata_extractor import MetadataExtractor


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_extract_populates_fields_from_frontmatter(tmp_path):
    content = (
        '---\n'
        'title: "Getting Started"\n'
        'article_id: "12345"\n'
        'article_slug: "getting-started"\n'
        'source_url: "https://example.com/getting-started"\n'
        'title_slug: "getting-started"\n'
        '---\n\n'
        "# Getting Started\nSome article body text.\n"
    )
    filepath = _write(tmp_path, "getting-started.md", content)

    result = MetadataExtractor().extract(
        company="claude",
        company_file_order=1,
        file_order=5,
        dataset={
            "content": content,
            "product_area": "getting-started",
            "filepath": filepath,
        },
    )

    assert result["company"] == "claude"
    assert result["doc_title"] == "Getting Started"
    assert result["article_id"] == "12345"
    assert result["source_url"] == "https://example.com/getting-started"
    assert result["product_area"] == "getting-started"
    assert result["file_order"] == 5
    assert result["company_file_order"] == 1
    assert "checksum" in result


def test_extract_excludes_missing_optional_fields_but_keeps_product_area(
    tmp_path,
):
    content = '---\ntitle: "No Extras"\n---\n\nBody text.\n'
    filepath = _write(tmp_path, "no-extras.md", content)

    result = MetadataExtractor().extract(
        company="hackerrank",
        company_file_order=1,
        file_order=1,
        dataset={
            "content": content,
            "product_area": None,
            "filepath": filepath,
        },
    )

    assert "article_id" not in result
    assert "last_updated_exact" not in result
    assert "product_area" in result
    assert result["product_area"] is None


def test_extract_normalizes_last_updated_exact_to_iso(tmp_path):
    content = (
        '---\ntitle: "Dated"\nlast_updated_exact: "Apr 15, 2026, 01:46 PM"\n'
        "---\n\nBody.\n"
    )
    filepath = _write(tmp_path, "dated.md", content)

    result = MetadataExtractor().extract(
        "claude",
        1,
        1,
        {"content": content, "product_area": "x", "filepath": filepath},
    )

    assert result["last_updated_exact"].startswith("2026-04-15T13:46:00")


def test_checksum_is_stable_across_independent_extractions(tmp_path):
    """Two extractions of identical content get different random `id`s and
    different created_at/utc_datetime timestamps, but the checksum used for
    ingestion dedup must still match — otherwise re-running ingestion would
    treat every unchanged file as new."""
    content = '---\ntitle: "Stable"\n---\n\nBody.\n'
    filepath = _write(tmp_path, "stable.md", content)
    dataset = {
        "content": content,
        "product_area": "area",
        "filepath": filepath,
    }

    result1 = MetadataExtractor().extract("claude", 1, 1, dataset)
    result2 = MetadataExtractor().extract("claude", 1, 1, dataset)

    assert result1["id"] != result2["id"]
    assert result1["checksum"] == result2["checksum"]


def test_checksum_changes_when_body_content_changes(tmp_path):
    filepath_a = _write(tmp_path, "a.md", "placeholder")
    filepath_b = _write(tmp_path, "b.md", "placeholder")
    dataset_a = {
        "content": '---\ntitle: "A"\n---\n\nBody A.\n',
        "product_area": "x",
        "filepath": filepath_a,
    }
    dataset_b = {
        "content": '---\ntitle: "A"\n---\n\nBody B.\n',
        "product_area": "x",
        "filepath": filepath_b,
    }

    result_a = MetadataExtractor().extract("claude", 1, 1, dataset_a)
    result_b = MetadataExtractor().extract("claude", 1, 1, dataset_b)

    assert result_a["checksum"] != result_b["checksum"]


def test_get_created_at_returns_none_for_missing_file():
    result = MetadataExtractor()._get_created_at(
        "/nonexistent/path/does-not-exist.md"
    )

    assert result is None


def test_export_excludes_protected_and_falsy_values_but_keeps_product_area():
    extractor = MetadataExtractor()
    extractor._content = "irrelevant"
    extractor.company = "claude"
    extractor.doc_title = "Title"
    extractor.article_id = None
    extractor.product_area = None
    extractor.breadcrumbs = []

    result = extractor._export()

    assert result["company"] == "claude"
    assert result["doc_title"] == "Title"
    assert "article_id" not in result
    assert "breadcrumbs" not in result  # empty list is falsy
    assert "_content" not in result
    assert result["product_area"] is None
