from mfem_rag.parse import carry, has_body, headings, parse_source, url_for


def test_url_for_fills_path_and_mkdocs_page():
    assert url_for("https://x.org/{path}", "a/b.html") == "https://x.org/a/b.html"
    assert url_for("https://x.org/{page}", "howto/ncmesh.md") == "https://x.org/howto/ncmesh/"
    assert url_for("https://x.org/{page}", "index.md") == "https://x.org/"
    assert url_for("https://x.org/{page}", "tutorial/index.md") == "https://x.org/tutorial/"


def test_carry_keeps_higher_levels_from_previous_page():
    previous = {"h1": "Chapter", "h2": "Old section", "h3": "Old sub"}
    assert carry(previous, {}) == previous
    assert carry(previous, {"h2": "New section"}) == {"h1": "Chapter", "h2": "New section"}


def test_headings_strip_emphasis_but_not_code_operators():
    assert headings({"h1": "**1.2 Installation**", "h2": "*Italic title*", "h3": "operator*()"}) == [
        "1.2 Installation", "Italic title", "operator*()",
    ]


def test_has_body_ignores_heading_only_sections():
    assert not has_body("## Title\n\n")
    assert has_body("## Title\ntext")


def test_markdown_source(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Guide\nIntro.\n## Build\n```bash\n# not a heading\nmake\n```\n## Empty\n"
    )
    records = parse_source({"name": "s", "format": "markdown", "path": str(tmp_path), "url": "https://x.org/{page}"})
    assert [r["metadata"]["headings"] for r in records] == [["Guide"], ["Guide", "Build"]]
    assert "# not a heading" in records[1]["text"]
    assert records[1]["id"] == "s/guide.md#1"
    assert records[1]["metadata"]["title"] == "Guide"
    assert records[1]["metadata"]["url"] == "https://x.org/guide/"


def test_html_source_uses_content_region_and_keeps_identifiers(tmp_path):
    (tmp_path / "page.html").write_text(
        "<html><head><title>Page Title</title></head><body>"
        "<nav>Site menu</nav>"
        "<div class='contents'><h2>boundary_integs</h2><p>See <a href='x.html'>Foo_bar</a>.</p>"
        "<p class='definition'>Definition at line 57.</p></div>"
        "</body></html>"
    )
    records = parse_source({
        "name": "s", "format": "html", "path": str(tmp_path),
        "url": "https://x.org/{path}", "content": "div.contents", "drop": ["p.definition"],
    })
    assert len(records) == 1
    record = records[0]
    assert record["metadata"]["title"] == "Page Title"
    assert record["metadata"]["headings"] == ["boundary_integs"]
    assert "See Foo_bar." in record["text"]
    assert "Site menu" not in record["text"]
    assert "Definition at line" not in record["text"]
