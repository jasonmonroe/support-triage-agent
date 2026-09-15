# tests/unit/src/test_document_handler.py
# +---------------------------------------------------------------------------+
# |                       DOCUMENT HANDLER TESTS                              |
# +---------------------------------------------------------------------------+
#
# Run: pytest tests/unit/src/test_document_handler.py -v

# Local Libraries
from src.document_handler import DocumentHandler


def _md_files(company, entries):
    """entries: list of (filename, content) tuples."""
    return {
        company: [
            {
                "product_area": None,
                "filepath": name,
                "filename": name,
                "content": content,
            }
            for name, content in entries
        ]
    }


# --- _format_html() ---------------------------------------------------------- #


def test_format_html_strips_markdown_images():
    handler = DocumentHandler({})
    text = "Before ![alt text](https://example.com/img.png) after."

    result = handler._format_html(text)

    assert "![" not in result
    assert "example.com" not in result
    assert "Before" in result and "after." in result


def test_format_html_strips_html_img_tags():
    handler = DocumentHandler({})
    text = 'Before <img src="https://example.com/img.png" /> after.'

    result = handler._format_html(text)

    assert "<img" not in result
    assert "example.com" not in result


def test_format_html_unescapes_html_entities():
    handler = DocumentHandler({})
    result = handler._format_html("Tests &amp; Questions &lt;pro&gt;")

    assert result == "Tests & Questions <pro>"


def test_format_html_collapses_blank_lines():
    handler = DocumentHandler({})
    result = handler._format_html("Line one\n\n\n\nLine two")

    assert "\n\n" not in result


# --- process() ---------------------------------------------------------------- #


def test_process_returns_empty_list_when_no_files():
    handler = DocumentHandler({})

    assert handler.process() == []


def test_process_strips_frontmatter_and_produces_chunks():
    content = (
        '---\ntitle: "Doc One"\n---\n\n'
        "# Doc One\nThis is the article body content for chunking.\n"
    )
    handler = DocumentHandler(_md_files("claude", [("doc-one.md", content)]))

    chunks = handler.process()

    assert len(chunks) > 0
    assert handler.count_documents() == 1
    assert handler.count_chunks() == len(chunks)
    # Frontmatter fields (only in metadata, not in the chunked page content)
    assert all("title:" not in c.page_content for c in chunks)


def test_process_assigns_sequential_chunk_idx_per_document():
    content = (
        '---\ntitle: "Doc"\n---\n\n'
        "# Doc\nBody content long enough to exist as a chunk.\n"
    )
    handler = DocumentHandler(_md_files("claude", [("doc.md", content)]))

    chunks = handler.process()

    chunk_indices = [c.metadata["chunk_idx"] for c in chunks]
    assert chunk_indices == list(range(len(chunks)))


def test_process_gives_identical_files_the_same_checksum(tmp_path):
    """A fresh MetadataExtractor() per document (src/document_handler.py's
    process() loop) keeps each file's checksum a pure function of its own
    content + filepath — two files with identical content and the same
    filepath must hash identically."""
    content = (
        '---\ntitle: "Same"\n---\n\n# Same\nIdentical body content here.\n'
    )
    filepath = tmp_path / "same.md"
    filepath.write_text(content)
    handler = DocumentHandler(
        _md_files(
            "claude", [(str(filepath), content), (str(filepath), content)]
        )
    )

    handler.process()

    checksums = [d.metadata["checksum"] for d in handler._documents]
    assert checksums[0] == checksums[1]


def test_process_dedups_identical_files_at_the_chunk_stage(tmp_path):
    """Once checksums are correctly identical, _create_chunks()'s dedup
    gate must actually skip the second copy rather than double-ingesting
    it."""
    content = (
        '---\ntitle: "Same"\n---\n\n# Same\nIdentical body content here.\n'
    )
    filepath = tmp_path / "same.md"
    filepath.write_text(content)
    handler = DocumentHandler(
        _md_files(
            "claude", [(str(filepath), content), (str(filepath), content)]
        )
    )

    chunks = handler.process()

    assert handler.count_documents() == 2  # both still parsed as Documents
    company_file_orders = {c.metadata["company_file_order"] for c in chunks}
    assert company_file_orders == {1}  # only the first copy's chunks remain


def test_process_gives_different_files_different_checksums():
    content_a = '---\ntitle: "A"\n---\n\n# A\nBody A content here.\n'
    content_b = '---\ntitle: "B"\n---\n\n# B\nBody B content here.\n'
    handler = DocumentHandler(
        _md_files("claude", [("a.md", content_a), ("b.md", content_b)])
    )

    chunks = handler.process()

    checksums = {c.metadata["checksum"] for c in chunks}
    assert len(checksums) == 2
