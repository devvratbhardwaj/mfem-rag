"""Module 3 - chunking

Turns the parsed sections into chunks to embed. Two strategies, to compare:

    recursive   each document as one text, cut by RecursiveCharacterTextSplitter
                (paragraphs, then lines, then words) with overlap. Ignores
                section boundaries. The baseline.
    structured  each section is a chunk. Sections over `size` are cut on
                paragraphs, keeping code blocks and tables whole; runs of
                sections under `min_size` are merged within their document.
                Every chunk is prefixed with its title and headings.

Output: data/chunks/<strategy>-<size>.jsonl, one chunk per line:
    {"id": ..., "text": ..., "body": ...,
     "metadata": {"source", "doc", "title", "headings", "url", "page", "sections", "strategy"}}
`text` is what gets embedded and shown to the model; `body` is the document's
own text, for display. `sections` lists the parsed section ids the chunk
draws on - the stable ids a benchmark can label relevance with, since chunk
ids change with the strategy.

    uv run scripts/chunk.py --strategy structured --size 1200
"""

import argparse
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]


def chunk_recursive(records: list[dict], size: int = 1200, overlap: int = 150) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    chunks = []
    for sections in by_document(records):
        # Join the document back together, remembering where each section starts
        starts, offset = [], 0
        for s in sections:
            starts.append(offset)
            offset += len(s["text"]) + 2
        text = "\n\n".join(s["text"] for s in sections)

        pieces = splitter.create_documents([text])
        for i, piece in enumerate(pieces):
            start = piece.metadata["start_index"]
            end = start + len(piece.page_content)
            covered = [s for s, begin in zip(sections, starts) if begin < end and begin + len(s["text"]) > start]
            chunks.append(make_chunk("recursive", i, piece.page_content, piece.page_content,
                                     covered, covered[0]["metadata"]["headings"]))
    return chunks


def chunk_structured(records: list[dict], size: int = 1200, min_size: int = 300) -> list[dict]:
    chunks = []
    for sections in by_document(records):
        # (body, section) units: whole sections, or the pieces of oversized ones
        units = [(piece, s) for s in sections for piece in split(s["text"].strip(), size)]

        groups: list[list[tuple[str, dict]]] = []
        for unit in units:
            if groups and mergeable(groups[-1], unit, size, min_size):
                groups[-1].append(unit)
            else:
                groups.append([unit])

        for i, group in enumerate(groups):
            covered = [s for _, s in group]
            heads = common_prefix([s["metadata"]["headings"] for s in covered])
            body = "\n\n".join(piece for piece, _ in group)
            text = f"{header(covered[0]['metadata']['title'], heads)}\n\n{body}"
            chunks.append(make_chunk("structured", i, text, body, covered, heads))
    return chunks


STRATEGIES = {"recursive": chunk_recursive, "structured": chunk_structured}


def by_document(records: list[dict]) -> list[list[dict]]:
    """Sections grouped by document, in their original order."""
    docs: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        docs.setdefault((r["metadata"]["source"], r["metadata"]["doc"]), []).append(r)
    return list(docs.values())


def split(text: str, size: int) -> list[str]:
    """The text if it fits; otherwise its blocks packed into pieces of at most
    `size`. A block too big on its own is cut by the recursive splitter."""
    if len(text) <= size:
        return [text]
    fallback = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0)
    pieces: list[str] = []
    current = ""
    for block in blocks(text):
        if current and len(current) + 2 + len(block) <= size:
            current += "\n\n" + block
            continue
        if current:
            pieces.append(current)
        if len(block) <= size:
            current = block
        else:
            pieces.extend(fallback.split_text(block))
            current = ""
    if current:
        pieces.append(current)
    return pieces


def blocks(text: str) -> list[str]:
    """Paragraphs: runs of lines between blank lines. A fenced code block is
    one block even with blank lines inside it; a table has none, so it is one
    block already."""
    result: list[str] = []
    lines: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not line.strip() and not fenced:
            if lines:
                result.append("\n".join(lines))
                lines = []
        else:
            lines.append(line)
    if lines:
        result.append("\n".join(lines))
    return result


def mergeable(group: list[tuple[str, dict]], unit: tuple[str, dict], size: int, min_size: int) -> bool:
    """Whether `unit` joins the chunk being built: only while that chunk is
    still under `min_size`, and only if the result stays within `size`."""
    length = sum(len(piece) + 2 for piece, _ in group) - 2
    return length < min_size and length + 2 + len(unit[0]) <= size


def header(title: str, heads: list[str]) -> str:
    """"Title > Heading > Subheading", without a first heading that only
    repeats the title."""
    if heads and heads[0] == title:
        heads = heads[1:]
    return " > ".join([title, *heads])


def common_prefix(lists: list[list[str]]) -> list[str]:
    prefix = lists[0]
    for other in lists[1:]:
        n = 0
        while n < min(len(prefix), len(other)) and prefix[n] == other[n]:
            n += 1
        prefix = prefix[:n]
    return list(prefix)


def make_chunk(strategy: str, i: int, text: str, body: str, covered: list[dict], heads: list[str]) -> dict:
    first = covered[0]["metadata"]
    return {
        "id": f"{strategy}:{first['source']}/{first['doc']}#{i}",
        "text": text,
        "body": body,
        "metadata": {
            "source": first["source"],
            "doc": first["doc"],
            "title": first["title"],
            "headings": heads,
            "url": first["url"],
            "page": first["page"],
            "sections": list(dict.fromkeys(s["id"] for s in covered)),
            "strategy": strategy,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=STRATEGIES, default="structured")
    parser.add_argument("--size", type=int, default=1200, help="maximum chunk length, in characters")
    parser.add_argument("--overlap", type=int, default=150, help="recursive: characters shared by neighbours")
    parser.add_argument("--min-size", type=int, default=300, help="structured: merge sections shorter than this")
    args = parser.parse_args()

    records = []
    for path in sorted((ROOT / "data" / "parsed").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)

    if args.strategy == "recursive":
        chunks = chunk_recursive(records, size=args.size, overlap=args.overlap)
    else:
        chunks = chunk_structured(records, size=args.size, min_size=args.min_size)

    out = ROOT / "data" / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.strategy}-{args.size}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    lengths = sorted(len(c["text"]) for c in chunks)
    print(f"{len(records)} sections -> {len(chunks)} chunks, "
          f"median {lengths[len(lengths) // 2]} chars, max {lengths[-1]} -> {path.relative_to(ROOT)}")
