"""Module 2 - parsing

Turns every source in sources.toml into Markdown, then splits it on headings.
Conversion is left to libraries - pymupdf4llm for PDF, markdownify for HTML -
so every format goes through the same splitter.

Output: data/parsed/<source>.jsonl, one section per line:
    {"id": ..., "text": ..., "metadata": {"source", "doc", "title", "headings", "url", "page"}}

    uv run scripts/parse.py
"""

import json
import re
import tomllib
from pathlib import Path, PurePosixPath

import pymupdf4llm
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter
from markdownify import markdownify

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {"html": "**/*.html", "markdown": "**/*.md", "pdf": "*.pdf"}
SPLITTER = MarkdownHeaderTextSplitter(
    [("#" * n, f"h{n}") for n in range(1, 7)], strip_headers=False
)


def parse_source(source: dict) -> list[dict]:
    path = ROOT / source["path"]
    files = [path] if path.is_file() else sorted(path.glob(PATTERNS[source["format"]]))
    if not files:
        raise FileNotFoundError(f"{source['name']}: no {source['format']} files under {path}")

    records = []
    for file in files:
        doc = file.name if path.is_file() else file.relative_to(path).as_posix()
        url = url_for(source["url"], doc)
        if source["format"] == "pdf":
            title, sections = pdf_sections(file, url, source.get("margins", [0, 0, 0, 0]))
        else:
            if source["format"] == "html":
                title, markdown = html_to_markdown(file, source.get("content"), source.get("drop", []))
            else:
                markdown = file.read_text(encoding="utf-8")
                title = None
            sections = [(headings(d.metadata), d.page_content, url, None) for d in SPLITTER.split_text(markdown)]
            # a Markdown file's title is its first heading
            title = title or next((heads[0] for heads, *_ in sections if heads), None)

        sections = [s for s in sections if has_body(s[1])]
        for i, (heads, text, section_url, page) in enumerate(sections):
            records.append({
                "id": f"{source['name']}/{doc}#{i}",
                "text": text,
                "metadata": {
                    "source": source["name"],
                    "doc": doc,
                    "title": title or file.stem,
                    "headings": heads,
                    "url": section_url,
                    "page": page,
                },
            })
    return records


def html_to_markdown(file: Path, content: str | None, drop: list[str]) -> tuple[str | None, str]:
    """The page's <title>, and its Markdown. The `content` CSS selector picks
    the documentation out of the page chrome; `drop` selectors remove what is
    left of it inside."""
    soup = BeautifulSoup(file.read_text(encoding="utf-8", errors="replace"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    node = soup.select_one(content) if content else soup.body
    if node is None:
        return title, ""
    for selector in drop:
        for element in node.select(selector):
            element.decompose()
    # Links point into the site's own layout: keep their text, drop the targets.
    # No Markdown escaping, so identifiers like boundary_integs stay searchable.
    markdown = markdownify(
        str(node), heading_style="ATX", strip=["a", "img"],
        escape_underscores=False, escape_asterisks=False, escape_misc=False,
    )
    return title, markdown


def pdf_sections(file: Path, url: str, margins: list[float]) -> tuple[str | None, list[tuple[list[str], str, str, int]]]:
    """The PDF's metadata title, and its sections. Split page by page so each
    section knows its page; a page that opens mid-section inherits the
    headings in force at the end of the previous one. `margins` (left, top,
    right, bottom, in points) are ignored on every page: running headers and
    footers live there."""
    # The layout-model mode drops the spaces inside monospace code
    # ("EPS eps" -> "EPSeps"); the classic mode keeps them. The classic mode
    # in turn drops text drawn over shaded boxes - code listings, usually -
    # unless told to ignore vector graphics.
    pymupdf4llm.use_layout(False)
    pages = pymupdf4llm.to_markdown(
        str(file), page_chunks=True, show_progress=False, margins=tuple(margins), ignore_graphics=True
    )
    sections = []
    current: dict[str, str] = {}
    for page in pages:
        number = page["metadata"]["page"]
        for d in SPLITTER.split_text(page["text"]):
            current = carry(current, d.metadata)
            sections.append((headings(current), d.page_content, f"{url}#page={number}", number))
    title = pages[0]["metadata"].get("title") if pages else None
    return title, sections


def carry(previous: dict[str, str], found: dict[str, str]) -> dict[str, str]:
    """Headings for a split: those found on this page, plus the previous
    page's headings at the levels above them."""
    if not found:
        return previous
    top = min(int(key[1:]) for key in found)
    return {**{k: v for k, v in previous.items() if int(k[1:]) < top}, **found}


def headings(metadata: dict[str, str]) -> list[str]:
    """Heading texts, top level first, without Markdown emphasis or HTML tags.
    Only whole-heading *italics* lose their asterisks: operator* keeps its."""
    texts = (re.sub(r"\*\*|`|<[^>]+>", "", metadata[k]).strip() for k in sorted(metadata))
    return [re.sub(r"^\*(.+)\*$", r"\1", text) for text in texts]


def has_body(text: str) -> bool:
    """False for a section that is only its heading line."""
    return any(line.strip() and not line.startswith("#") for line in text.splitlines())


def url_for(template: str, doc: str) -> str:
    """{path} is the file's path relative to the source; {page} is its MkDocs
    directory-style address (howto/ncmesh.md -> howto/ncmesh/, index.md -> its directory)."""
    stem = PurePosixPath(doc).with_suffix("").as_posix()
    page = stem.removesuffix("index") if stem == "index" or stem.endswith("/index") else f"{stem}/"
    return template.format(path=doc, page=page)


def main() -> None:
    with (ROOT / "sources.toml").open("rb") as f:
        sources = tomllib.load(f)["source"]

    out = ROOT / "data" / "parsed"
    out.mkdir(parents=True, exist_ok=True)
    for source in sources:
        print(f"==> {source['name']} ({source['format']})", flush=True)
        records = parse_source(source)
        with (out / f"{source['name']}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        docs = len({r["metadata"]["doc"] for r in records})
        chars = sum(len(r["text"]) for r in records)
        print(f"    {docs} docs, {len(records)} sections, {chars / 1e6:.1f}M chars", flush=True)
