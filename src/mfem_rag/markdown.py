"""Whitespace-preserving Markdown structure shared by parsing and chunking."""

import re
from dataclasses import dataclass

FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
HEADING = re.compile(r'^ {0,3}(#{1,6})[ \t]+(.+?)\s*#*\s*$')
ANCHOR = re.compile(r'^RAGANCHOR:([^\s]+)\s*$')


def closes_fence(line: str, fence: str) -> bool:
    match = FENCE.match(line)
    return bool(match and match[1][0] == fence[0] and len(match[1]) >= len(fence)
                and not match[2].strip())


@dataclass
class Section:
    page_content: str
    metadata: dict
    start: int
    end: int
    anchor: str | None = None


class HeadingSplitter:
    """Split ATX/setext headings outside fences without altering code or lists."""

    def split_text(self, text: str) -> list[Section]:
        lines = text.splitlines(keepends=True)
        sections = []
        heads = {}
        body = []
        fence = None
        offset = start = 0
        anchor = pending_anchor = None

        def emit(end):
            content = ''.join(body).strip('\n')
            if content.strip():
                sections.append(Section(content, dict(heads), start, end, anchor))

        i = 0
        while i < len(lines):
            line = lines[i]
            marker = ANCHOR.match(line.strip()) if fence is None else None
            if marker:
                pending_anchor = marker[1]
                offset += len(line)
                i += 1
                continue
            match = HEADING.match(line) if fence is None else None
            setext = (fence is None and line.strip() and i + 1 < len(lines)
                      and re.fullmatch(r' {0,3}(=+|-+)\s*', lines[i + 1])
                      and not line.startswith(('    ', '\t')))
            if match or setext:
                emit(offset)
                body = []
                start = offset
                level = len(match[1]) if match else (1 if lines[i + 1].lstrip().startswith('=') else 2)
                title = match[2] if match else line.strip()
                heads = {k: v for k, v in heads.items() if int(k[1:]) < level}
                heads[f'h{level}'] = title
                anchor = pending_anchor
                pending_anchor = None
                if setext:
                    body.extend(lines[i:i + 2])
                    offset += len(line) + len(lines[i + 1])
                    i += 2
                    continue
            opening = FENCE.match(line)
            if fence:
                if closes_fence(line, fence):
                    fence = None
            elif opening:
                fence = opening[1]
            body.append(line)
            offset += len(line)
            i += 1
        emit(offset)
        return sections


def blocks(text: str) -> list[str]:
    """Separate prose, fenced code, and table blocks, retaining indentation."""
    result, lines = [], []
    fence = None
    table = False

    def flush():
        if lines:
            result.append('\n'.join(lines))
            lines.clear()

    for line in text.splitlines():
        if fence:
            lines.append(line)
            if closes_fence(line, fence):
                fence = None
                flush()
            continue
        match = FENCE.match(line)
        if match:
            flush()
            table = False
            fence = match[1]
            lines.append(line)
            continue
        is_table = '|' in line
        if is_table != table:
            flush()
            table = is_table
        if not line.strip():
            flush()
        else:
            lines.append(line)
    # PDFs can open/close fences at page boundaries; repair the local block.
    if fence:
        lines.append(fence)
    flush()
    return result
