"""Module 2 - parsing

Turns every source in sources.toml into Markdown, then splits it on headings.
Conversion is left to libraries - pymupdf4llm for PDF, markdownify for HTML -
so every format goes through the same splitter.

Output: data/parsed/<source>.jsonl, one section per line:
    {"id": ..., "text": ..., "metadata": {"source", "doc", "title", "headings", "url", "page"}}

    uv run scripts/parse.py
"""

import re
import tomllib
import warnings
import fnmatch
import platform
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, quote, urlsplit

import pymupdf4llm
from bs4 import BeautifulSoup
from markdownify import markdownify
from mfem_rag.artifacts import SCHEMA_VERSION, fingerprint, file_hash, write_json, write_jsonl
from mfem_rag.markdown import HeadingSplitter, FENCE, HEADING, closes_fence

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {"html": "**/*.html", "markdown": "**/*.md", "pdf": "**/*.pdf", 'code': '**/*'}
SPLITTER = HeadingSplitter()


def source_files(source: dict, report: list[dict] | None = None) -> list[Path]:
    if source['format'] not in PATTERNS:
        raise ValueError(f"{source['name']}: unsupported format {source['format']!r}")
    path = ROOT / source['path']
    if not path.exists():
        if source.get('optional', False):
            warnings.warn(f"{source['name']}: optional source missing: {path}")
            if report is not None:
                report.append({'source': source['name'], 'doc': None, 'status': 'missing_optional',
                               'sections': 0, 'empty': False})
            return []
        raise FileNotFoundError(f"{source['name']}: missing source: {path}")
    files = [path] if path.is_file() else sorted(path.glob(PATTERNS[source['format']]))
    retained = []
    for file in files:
        if not file.is_file() or (source['format'] == 'code' and file.suffix not in {'.cpp', '.hpp', '.opts'}):
            continue
        doc = file.relative_to(path).as_posix() if path.is_dir() else file.name
        matching = [pattern for pattern in source.get('exclude', []) if fnmatch.fnmatch(doc, pattern)]
        if matching:
            if report is not None:
                report.append({'source': source['name'], 'doc': doc, 'status': 'excluded',
                               'patterns': matching, 'sections': 0, 'empty': False})
        else:
            retained.append(file)
    return retained


def parse_source(source: dict, report: list[dict] | None = None) -> list[dict]:
    path = ROOT / source["path"]
    files = source_files(source, report)
    if not files:
        if source.get('optional', False):
            return []
        raise FileNotFoundError(f"{source['name']}: no {source['format']} files under {path}")

    records = []
    for file in files:
        doc = file.name if path.is_file() else file.relative_to(path).as_posix()
        url = url_for(source["url"], doc)
        document_hash = file_hash(file)
        spans = []
        anchors = []
        if source["format"] == "pdf":
            title, sections = pdf_sections(file, url, source.get("margins", [0, 0, 0, 0]))
        else:
            if source["format"] == "html":
                title, markdown = html_to_markdown(
                    file, source.get("content"), source.get("drop", []), source.get("flatten", []), url
                )
            elif source['format'] == 'code':
                raw = file.read_text(encoding='utf-8')
                fence = '`' * max(3, 1 + max((len(m[0]) for m in re.finditer(r'`+', raw)), default=0))
                language = 'cpp' if file.suffix in {'.cpp', '.hpp'} else 'text'
                markdown = f'# {doc}\n\n{fence}{language}\n{raw.rstrip(chr(10))}\n{fence}\n'
                title = doc
            else:
                markdown = normalize_markdown(file.read_text(encoding="utf-8"), url, file.name == 'index.md')
                title = None
            split_sections = SPLITTER.split_text(markdown)
            sections = [(headings(d.metadata), d.page_content,
                         f'{url}#{quote(d.anchor, safe="_:-.")}' if d.anchor else url, None)
                        for d in split_sections]
            spans = [(d.start, d.end) for d in split_sections]
            anchors = [d.anchor for d in split_sections]
            # a Markdown file's title is its first heading
            title = title or next((heads[0] for heads, *_ in sections if heads), None)

        retained = [(i, s) for i, s in enumerate(sections) if has_body(s[1])]
        if report is not None:
            report.append({'source': source['name'], 'doc': doc, 'sha256': document_hash,
                           'status': 'parsed' if retained else 'empty', 'sections': len(retained), 'empty': not retained,
                           'replacement_characters': sum(s[1].count('\ufffd') for _, s in retained)})
        if not retained:
            warnings.warn(f"{source['name']}/{doc}: no retained content")
        for i, (original_index, (heads, text, section_url, page)) in enumerate(retained):
            anchor = anchors[original_index] if anchors else None
            section_key = anchor or fingerprint({'page': page, 'headings': heads, 'text': text})[:16]
            records.append({
                "id": f"{source['name']}/{doc}@{document_hash[:16]}#{section_key}:{i}",
                "text": text,
                "metadata": {
                    "source": source["name"],
                    "doc": doc,
                    "title": title or file.stem,
                    "headings": heads,
                    "url": section_url,
                    "page": page,
                    'document_sha256': document_hash,
                    'anchor': anchor,
                    'source_span': list(spans[original_index]) if spans else None,
                    'span_unit': 'normalized_markdown_characters' if spans else None,
                },
            })
    return records


def html_to_markdown(
    file: Path, content: str | None, drop: list[str], flatten: list[str], url: str = ''
) -> tuple[str | None, str]:
    """The page's <title>, and its Markdown. The `content` CSS selector picks
    the documentation out of the page chrome; `drop` selectors remove what is
    left of it inside; `flatten` selectors become one-line code blocks."""
    soup = BeautifulSoup(file.read_text(encoding="utf-8", errors="replace"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    node = soup.select_one(content) if content else soup.body
    if node is None:
        raise ValueError(f'{file}: content selector {content or "body"!r} not found')
    for selector in drop:
        for element in node.select(selector):
            element.decompose()
    for selector in flatten:
        for element in node.select(selector):
            pre = soup.new_tag("pre")
            pre.string = one_line(element.get_text(" "))
            element.replace_with(pre)
    # Doxygen source listings use div.line, with line numbers in separate spans.
    for fragment in node.select('div.fragment'):
        source_lines = fragment.select('div.line')
        if source_lines:
            for number in fragment.select('span.lineno'):
                number.decompose()
            pre = soup.new_tag('pre')
            pre.string = '\n'.join(line.get_text().replace('\xa0', ' ') for line in source_lines)
            fragment.replace_with(pre)
    for heading in node.select('h1,h2,h3,h4,h5,h6'):
        target = heading.get('id')
        child = heading.select_one('[id], a[name]')
        previous = heading.find_previous_sibling()
        if not target and child:
            target = child.get('id') or child.get('name')
        if not target and previous and previous.name == 'a':
            target = previous.get('id') or previous.get('name')
        if not target and heading.parent.name == 'section':
            target = heading.parent.get('id')
        if target:
            marker = soup.new_tag('p')
            marker.string = f'RAGANCHOR:{target}'
            heading.insert_before(marker)
    for link in node.select('a[href]'):
        link['href'] = urljoin(url, link['href'])
    for image in node.select('img'):
        # Preserve equation/diagram alt text without embedding nontext assets.
        image.replace_with(image.get('alt', ''))
    markdown = markdownify(
        str(node), heading_style="ATX",
        escape_underscores=False, escape_asterisks=False, escape_misc=False,
    )
    return title, markdown


def normalize_markdown(text: str, url: str, index_page: bool = False) -> str:
    """Normalize mixed HTML/Markdown, protecting fenced and inline code."""
    protected = {}
    lines = text.splitlines(keepends=True)
    output, code = [], []
    fence = None
    def stash(value):
        key = f'RAGPROTECTED{len(protected):06d}TOKEN'
        protected[key] = value
        return key
    for line in lines:
        if fence:
            code.append(line)
            if closes_fence(line, fence):
                output.append(stash(''.join(code)) + '\n')
                code = []
                fence = None
        elif match := FENCE.match(line):
            fence = match[1]
            code = [line]
        else:
            output.append(line)
    if code:
        output.append(stash(''.join(code)) + '\n')
    normalized = ''.join(output)
    normalized = re.sub(r'(`+)([^\n]*?)\1', lambda m: stash(m[0]), normalized)
    if re.search(r'</?[A-Za-z][^>]*>', normalized):
        soup = BeautifulSoup(normalized, 'html.parser')
        for link in soup.select('a[href]'):
            link['href'] = markdown_link(url, link['href'], index_page)
        normalized = markdownify(str(soup), heading_style='ATX',
                                 escape_underscores=False, escape_asterisks=False, escape_misc=False)
    # Relative Markdown links are rooted at the source page (not the repo).
    normalized = re.sub(r'(?<!!)\[([^\]\n]+)\]\(([^\s)]+)\)',
                        lambda m: f'[{m[1]}]({markdown_link(url, m[2], index_page)})', normalized)
    for key, value in protected.items():
        normalized = normalized.replace(key, value)
    return normalized


def markdown_link(url: str, target: str, index_page: bool = False) -> str:
    """Resolve MkDocs .md paths relative to the original source file's directory."""
    relative = not urlsplit(target).scheme and not urlsplit(target).netloc
    if relative and target.split('#', 1)[0].endswith('.md') and url.startswith('https://mfem.org/'):
        # A rendered foo/ page corresponds to foo.md one level up; index pages
        # are the exception and are handled by explicit links in their source.
        base = url if index_page else url.rstrip('/').rsplit('/', 1)[0] + '/'
        path, separator, anchor = target.partition('#')
        resolved = urljoin(base, path)
        resolved = resolved[:-len('index.md')] if resolved.endswith('/index.md') else resolved[:-3] + '/'
        return resolved + (separator + anchor if separator else '')
    return urljoin(url, target)


def one_line(text: str) -> str:
    """Text laid out in table cells, as one line of code: whitespace
    collapsed, none inside parentheses or before commas."""
    text = " ".join(text.split())
    text = re.sub(r"\s+([(),])", r"\1", text)
    return re.sub(r"\(\s+", "(", text)


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
    texts = (re.sub(r"\*\*|`|</?(?:span|em|strong|b|i)(?:\s[^>]*)?>", "", metadata[k]).strip() for k in sorted(metadata))
    return [re.sub(r"^\*(.+)\*$", r"\1", text) for text in texts]


def has_body(text: str) -> bool:
    """False for a section that is only its heading line."""
    fence = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if fence:
            if closes_fence(line, fence):
                fence = None
            elif line.strip():
                return True
        elif match := FENCE.match(line):
            fence = match[1]
        elif (line.strip() and not HEADING.match(line)
              and not re.fullmatch(r' {0,3}(=+|-+)\s*', line)
              and not (i + 1 < len(lines) and re.fullmatch(r' {0,3}(=+|-+)\s*', lines[i + 1]))):
            return True
    return False


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
    report, outputs = [], []
    for source in sources:
        print(f"==> {source['name']} ({source['format']})", flush=True)
        records = parse_source(source, report)
        if not records:
            continue
        path = out / f"{source['name']}.jsonl"
        write_jsonl(path, records)
        outputs.append({'file': path.name, 'sha256': file_hash(path), 'sections': len(records)})
        docs = len({r["metadata"]["doc"] for r in records})
        chars = sum(len(r["text"]) for r in records)
        print(f"    {docs} docs, {len(records)} sections, {chars / 1e6:.1f}M chars", flush=True)
    if not outputs:
        raise ValueError('No content parsed; check sources.toml and raw source paths.')
    configuration = {'sources': sources, 'schema_version': SCHEMA_VERSION,
                     'python_version': platform.python_version(),
                     'parser_sha256': file_hash(Path(__file__)),
                     'markdown_sha256': file_hash(Path(__file__).with_name('markdown.py')),
                     'libraries': {name: version(name) for name in ['markdownify', 'pymupdf', 'pymupdf4llm', 'beautifulsoup4']}}
    corpus_id = fingerprint({'configuration': configuration, 'documents': report})
    write_json(out / 'audit.json', report)
    write_json(out / 'manifest.json', {'schema_version': SCHEMA_VERSION, 'corpus_id': corpus_id,
                                     'configuration': configuration, 'outputs': outputs})
