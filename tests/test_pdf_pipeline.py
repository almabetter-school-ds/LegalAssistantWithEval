from legal_graphrag.ingestion.pdf_pipeline import (
    clean_text,
    detect_section_heading,
    table_to_markdown,
    PageRecord,
    compute_page_sections,
)


def test_clean_text_dehyphenates_across_line_breaks():
    assert clean_text("contrac-\ntual obligations") == "contractual obligations"


def test_clean_text_collapses_blank_lines():
    assert clean_text("Para one.\n\n\n\nPara two.") == "Para one.\n\nPara two."


def test_detect_section_heading_matches_short_titlecase_line():
    assert detect_section_heading("Governing Law\nThis Agreement shall be governed...") == "Governing Law"


def test_detect_section_heading_ignores_prose_first_line():
    assert detect_section_heading("this agreement is entered into by and between...") is None


def test_table_to_markdown_renders_header_and_rows():
    table = [["Name", "Rate"], ["Widget A", "$5"], ["Widget B", None]]
    md = table_to_markdown(table)
    assert md.splitlines()[0] == "| Name | Rate |"
    assert "| Widget B |  |" in md


def test_compute_page_sections_carries_heading_forward():
    pages = [
        PageRecord(page_number=1, text="Confidentiality\nSome text.", source="pdfplumber", char_count=20),
        PageRecord(page_number=2, text="continued text with no heading", source="pdfplumber", char_count=30),
    ]
    sections = compute_page_sections(pages)
    assert sections[1] == "Confidentiality"
    assert sections[2] == "Confidentiality"  # carried forward, no new heading on page 2
