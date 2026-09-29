from mfem_rag.chunk import chunk_recursive, chunk_structured


def section(doc, i, text, headings=(), title="Title", page=None, source="s"):
    return {
        "id": f"{source}/{doc}#{i}",
        "text": text,
        "metadata": {
            "source": source,
            "doc": doc,
            "title": title,
            "headings": list(headings),
            "url": f"https://x.org/{doc}" + (f"#page={page}" if page else ""),
            "page": page,
        },
    }


def paragraphs(n, width=90):
    return "\n\n".join(f"Paragraph {i} " + "x" * width for i in range(n))


# recursive


def test_recursive_chunks_fit_size_and_span_sections():
    records = [section("a.md", 0, "## One\nshort"), section("a.md", 1, "## Two\nalso short")]
    chunks = chunk_recursive(records, size=200, overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["sections"] == ["s/a.md#0", "s/a.md#1"]
    assert "short" in chunks[0]["text"] and "also short" in chunks[0]["text"]

    long = chunk_recursive([section("b.md", 0, paragraphs(30))], size=300, overlap=50)
    assert len(long) > 1
    assert all(len(c["text"]) <= 300 for c in long)


def test_recursive_never_mixes_documents():
    records = [section("a.md", 0, "alpha"), section("b.md", 0, "beta")]
    chunks = chunk_recursive(records, size=200, overlap=0)
    assert [c["metadata"]["doc"] for c in chunks] == ["a.md", "b.md"]
    assert [c["id"] for c in chunks] == ["recursive:s/a.md#0", "recursive:s/b.md#0"]


def test_recursive_cites_the_page_where_a_chunk_starts():
    records = [section("m.pdf", i, paragraphs(3), page=i + 1) for i in range(3)]
    chunks = chunk_recursive(records, size=300, overlap=0)
    pages = [c["metadata"]["page"] for c in chunks]
    assert pages == sorted(pages) and pages[0] == 1 and pages[-1] == 3
    for c in chunks:
        assert c["metadata"]["url"] == f"https://x.org/m.pdf#page={c['metadata']['page']}"


# structured


def test_structured_merges_small_sections_within_a_document():
    records = [section("a.html", i, f"## m{i}()\nMethod {i}.", headings=[f"m{i}()"]) for i in range(3)]
    records.append(section("b.html", 0, "## other\nOther doc."))
    chunks = chunk_structured(records, size=1000, min_size=300)
    assert [c["metadata"]["sections"] for c in chunks] == [
        ["s/a.html#0", "s/a.html#1", "s/a.html#2"],
        ["s/b.html#0"],
    ]


def test_structured_keeps_large_enough_sections_on_their_own():
    records = [section("a.md", 0, "x" * 400), section("a.md", 1, "short")]
    chunks = chunk_structured(records, size=1000, min_size=300)
    assert [c["metadata"]["sections"] for c in chunks] == [["s/a.md#0"], ["s/a.md#1"]]


def test_structured_splits_oversized_sections_on_paragraphs():
    text = paragraphs(20)
    chunks = chunk_structured([section("a.md", 0, text, headings=["Big"])], size=300, min_size=100)
    assert len(chunks) > 1
    assert all(len(c["body"]) <= 300 for c in chunks)
    assert all(c["metadata"]["sections"] == ["s/a.md#0"] for c in chunks)
    # paragraph boundaries are respected: every paragraph lands whole in one chunk
    for p in text.split("\n\n"):
        assert any(p in c["body"] for c in chunks)


def test_structured_keeps_code_blocks_whole():
    code = "```c\nint a;\n\nint b;\n\nint c;\n```"
    text = paragraphs(2) + "\n\n" + code + "\n\n" + paragraphs(2)
    chunks = chunk_structured([section("a.md", 0, text)], size=250, min_size=50)
    assert any(code in c["body"] for c in chunks)


def test_structured_keeps_tables_whole():
    table = "| option | meaning |\n| --- | --- |\n| -ksp_rtol | relative tolerance |\n| -ksp_atol | absolute tolerance |"
    text = paragraphs(2) + "\n\n" + table + "\n\n" + paragraphs(2)
    chunks = chunk_structured([section("a.md", 0, text)], size=250, min_size=50)
    assert any(table in c["body"] for c in chunks)


def test_structured_splits_a_block_larger_than_size():
    code = "```c\n" + "\n".join(f"int v{i} = {i};" for i in range(100)) + "\n```"
    chunks = chunk_structured([section("a.md", 0, code)], size=300, min_size=50)
    assert len(chunks) > 1
    assert all(len(c["body"]) <= 300 for c in chunks)


def test_structured_prefixes_title_and_headings():
    record = section("a.html", 0, "Combination of level sets.", headings=["Detailed Description"],
                     title="MFEM: Combo Class Reference")
    [chunk] = chunk_structured([record])
    assert chunk["text"] == "MFEM: Combo Class Reference > Detailed Description\n\nCombination of level sets."
    assert chunk["body"] == "Combination of level sets."
    assert chunk["metadata"]["headings"] == ["Detailed Description"]


def test_structured_header_skips_title_repeated_as_heading():
    [chunk] = chunk_structured([section("a.md", 0, "text", headings=["Guide", "Build"], title="Guide")])
    assert chunk["text"].startswith("Guide > Build\n\n")


def test_structured_merged_chunk_uses_shared_headings():
    records = [
        section("a.md", 0, "one", headings=["Guide", "Build", "Linux"]),
        section("a.md", 1, "two", headings=["Guide", "Build", "Mac"]),
    ]
    [chunk] = chunk_structured(records)
    assert chunk["metadata"]["headings"] == ["Guide", "Build"]
    assert chunk["metadata"]["url"] == "https://x.org/a.md"


def test_structured_ids_are_sequential_per_document():
    records = [section("a.md", 0, "x" * 400), section("a.md", 1, "y" * 400), section("b.md", 0, "z")]
    ids = [c["id"] for c in chunk_structured(records, size=1000, min_size=300)]
    assert ids == ["structured:s/a.md#0", "structured:s/a.md#1", "structured:s/b.md#0"]
