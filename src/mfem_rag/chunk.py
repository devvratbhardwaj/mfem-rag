"""Chunk parsed sections with complete citations and identifiable configurations.

    uv run scripts/chunk.py --strategy structured --size 1200
    uv run scripts/chunk.py --unit tokens --tokenizer /path/to/tokenizer.json --size 384

Tokenizers are loaded from local files; this command never downloads models.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Callable
from importlib.metadata import version

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mfem_rag.artifacts import SCHEMA_VERSION, fingerprint, file_hash, load_parsed, write_json, write_jsonl
from mfem_rag.markdown import FENCE, blocks, closes_fence

ROOT = Path(__file__).resolve().parents[2]
Measure = Callable[[str], int]


def splitter_measure(measure: Measure) -> Measure:
    """Do not count document-level special tokens once per separator/word."""
    special_tokens = measure('')
    return lambda value: max(0, measure(value) - special_tokens)


def validate(size: int, overlap: int = 0, min_size: int = 0) -> None:
    if size <= 0 or not 0 <= overlap < size or not 0 <= min_size <= size:
        raise ValueError('Require size > 0, 0 <= overlap < size, and 0 <= min_size <= size.')


def configuration(strategy, size, overlap=0, min_size=0, budget_id='characters'):
    return {'schema_version': SCHEMA_VERSION, 'strategy': strategy, 'size': size,
            'overlap': overlap, 'min_size': min_size, 'budget': budget_id,
            'chunker_sha256': file_hash(Path(__file__)),
            'markdown_sha256': file_hash(Path(__file__).with_name('markdown.py')),
            'libraries': {name: version(name) for name in ['langchain-text-splitters', 'tokenizers']}}


def by_document(records: list[dict]) -> list[list[dict]]:
    docs = {}
    for record in records:
        if not record['text'].strip():
            continue
        docs.setdefault((record['metadata']['source'], record['metadata']['doc']), []).append(record)
    return list(docs.values())


def active_fence(text: str) -> str | None:
    active = None
    for line in text.splitlines():
        if active:
            if closes_fence(line, FENCE.match(active)[1]):
                active = None
        elif FENCE.match(line):
            active = line
    return active


def repair_piece(body: str, preceding: str) -> str:
    """Give a baseline substring the fence context it had in the full document."""
    active = active_fence(preceding)
    if active:
        body = active + '\n' + body
    trailing = active_fence(body)
    if trailing:
        body += '\n' + FENCE.match(trailing)[1]
    return body


def chunk_recursive(records: list[dict], size: int = 1200, overlap: int = 150,
                    measure: Measure = len, budget_id: str = 'characters') -> list[dict]:
    validate(size, overlap)
    config = configuration('recursive', size, overlap=overlap, budget_id=budget_id)
    identity = fingerprint({'config': config, 'records': records})[:16]
    chunks = []
    for sections in by_document(records):
        starts, offset = [], 0
        for section in sections:
            starts.append(offset)
            offset += len(section['text']) + 2
        text = '\n\n'.join(section['text'] for section in sections)
        # Reserve room for fence context; retry with a smaller effective budget
        # if the tokenizer's nonadditive lengths require more room.
        effective = size
        while True:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=effective, chunk_overlap=min(overlap, effective - 1),
                length_function=splitter_measure(measure), add_start_index=False, strip_whitespace=False)
            bodies = splitter.split_text(text)
            pieces, cursor = [], 0
            for body in bodies:
                # Find in document order; overlap may repeat part of the previous body.
                start = text.find(body, cursor)
                if start < 0:
                    raise ValueError('Cannot map recursive chunk back to source text.')
                end = start + len(body)
                rendered = repair_piece(body, text[:start])
                pieces.append((start, end, rendered))
                # Search after the previous start (not end), independent of token units.
                cursor = start + 1
            excess = max((measure(body) - size for _, _, body in pieces), default=0)
            if excess <= 0:
                break
            effective -= max(excess, 1)
            if effective <= 0:
                raise ValueError('Budget cannot fit recursive code fence context; increase --size.')
        for i, (start, end, body) in enumerate(pieces):
            covered = [section for section, begin in zip(sections, starts)
                       if begin < end and begin + len(section['text']) > start]
            if not covered:
                continue
            chunk = make_chunk('recursive', i, body, body, covered,
                               covered[0]['metadata']['headings'], identity, config)
            # Exact section-relative spans before fence context is added.
            for location, section in zip(chunk['metadata']['locations'], covered):
                begin = starts[sections.index(section)]
                location['section_span'] = [max(0, start - begin), min(len(section['text']), end - begin)]
            chunks.append(chunk)
    return chunks


def chunk_structured(records: list[dict], size: int = 1200, min_size: int = 300,
                     measure: Measure = len, budget_id: str = 'characters') -> list[dict]:
    validate(size, min_size=min_size)
    config = configuration('structured', size, min_size=min_size, budget_id=budget_id)
    identity = fingerprint({'config': config, 'records': records})[:16]
    chunks = []
    for sections in by_document(records):
        units = []
        for section in sections:
            prefix = header(section['metadata']['title'], section['metadata']['headings'])
            # The budget includes prefixes, fences, repeated headers, and specials.
            fits = lambda body, prefix=prefix: measure(prefix + '\n\n' + body) <= size
            try:
                for piece in split(section['text'].strip('\n'), size, measure, fits):
                    units.append((piece, section))
            except ValueError as error:
                raise ValueError(f"{section['id']}: {error}") from error
        groups = []
        for unit in units:
            if groups and mergeable(groups[-1], unit, size, min_size, measure):
                groups[-1].append(unit)
            else:
                groups.append([unit])
        for i, group in enumerate(groups):
            covered = [section for _, section in group]
            heads = common_prefix([section['metadata']['headings'] for section in covered])
            body = '\n\n'.join(piece for piece, _ in group)
            text = header(covered[0]['metadata']['title'], heads) + '\n\n' + body
            if measure(text) > size:
                raise ValueError('Structured chunk exceeds budget.')
            chunks.append(make_chunk('structured', i, text, body, covered, heads, identity, config))
    return chunks


def split(text: str, size: int, measure: Measure = len,
          fits: Callable[[str], bool] | None = None) -> list[str]:
    """Pack blocks. Code splits on lines; tables repeat headers.

    A code line/table row that cannot fit raises an actionable error instead of
    silently breaking syntax or exceeding the model budget.
    """
    validate(size)
    fits = fits or (lambda value: measure(value) <= size)
    pieces, current = [], ''
    for block in blocks(text):
        for unit in split_block(block, size, measure, fits):
            candidate = current + '\n\n' + unit if current else unit
            if fits(candidate):
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = unit
    if current:
        pieces.append(current)
    return pieces


def split_block(block, size, measure, fits):
    if fits(block):
        return [block]
    lines = block.splitlines()
    opening = FENCE.match(lines[0]) if lines else None
    if opening:
        fence = opening[1]
        content = lines[1:-1] if closes_fence(lines[-1], fence) else lines[1:]
        render = lambda rows: lines[0] + '\n' + '\n'.join(rows) + '\n' + fence
        kind = 'code line'
    elif len(lines) >= 2 and '|' in lines[0] and re.fullmatch(
            r'\s*\|?[\s:|\-]+\|?\s*', lines[1]):
        content = lines[2:]
        render = lambda rows: '\n'.join(lines[:2] + rows)
        kind = 'table row'
    else:
        # Generic prose supports arbitrarily long words without touching code.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=0, length_function=splitter_measure(measure), strip_whitespace=False)
        output = []
        for piece in splitter.split_text(block):
            pending = [piece]
            while pending:
                value = pending.pop(0)
                if fits(value):
                    output.append(value)
                elif len(value) <= 1:
                    raise ValueError('Budget cannot fit the title/headings and one character; increase --size.')
                else:
                    middle = len(value) // 2
                    pending[0:0] = [value[:middle], value[middle:]]
        return output
    result, rows = [], []
    if not content and not fits(render([])):
        raise ValueError(f'Budget cannot fit {kind} header; increase --size.')
    for line in content:
        if not fits(render([line])):
            raise ValueError(f'An indivisible {kind} exceeds the complete text budget; increase --size.')
        if rows and not fits(render(rows + [line])):
            result.append(render(rows))
            rows = []
        rows.append(line)
    if rows or not content:
        result.append(render(rows))
    return result


def mergeable(group, unit, size, min_size, measure=len):
    length = measure('\n\n'.join(piece for piece, _ in group))
    old = group[-1][1]['metadata']['headings']
    new = unit[1]['metadata']['headings']
    # Sibling sections may merge; unrelated branches and separate API methods may not.
    related = old == new or (len(old) >= 2 and len(new) >= 2 and old[:-1] == new[:-1])
    if not related or length >= min_size:
        return False
    combined = group + [unit]
    heads = common_prefix([section['metadata']['headings'] for _, section in combined])
    text = header(group[0][1]['metadata']['title'], heads) + '\n\n' + '\n\n'.join(piece for piece, _ in combined)
    return measure(text) <= size


def header(title, heads):
    if heads and heads[0] == title:
        heads = heads[1:]
    return ' > '.join([title, *heads])


def common_prefix(lists):
    prefix = lists[0]
    for other in lists[1:]:
        n = 0
        while n < min(len(prefix), len(other)) and prefix[n] == other[n]:
            n += 1
        prefix = prefix[:n]
    return list(prefix)


def make_chunk(strategy, i, text, body, covered, heads, identity, config):
    first = covered[0]['metadata']
    unique = {section['id']: section for section in covered}
    locations = [{'section_id': section['id'], 'url': section['metadata']['url'],
                  'page': section['metadata']['page'],
                  'document_sha256': section['metadata'].get('document_sha256'),
                  'source_span': section['metadata'].get('source_span'),
                  'section_span': [0, len(section['text'])]}
                 for section in unique.values()]
    return {'id': f"{strategy}:{identity}:{first['source']}/{first['doc']}#{i}",
            'text': text, 'body': body,
            'metadata': {'source': first['source'], 'doc': first['doc'], 'title': first['title'],
                         'headings': heads, 'url': first['url'], 'page': first['page'],
                         'sections': list(unique), 'locations': locations,
                         'strategy': strategy, 'configuration': config, 'artifact_id': identity}}


STRATEGIES = {'recursive': chunk_recursive, 'structured': chunk_structured}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--strategy', choices=STRATEGIES, default='structured')
    parser.add_argument('--size', type=int, default=1200, help='maximum complete text length in selected units')
    parser.add_argument('--overlap', type=int, default=150, help='recursive overlap in selected units')
    parser.add_argument('--min-size', type=int, default=300, help='structured merge threshold in selected units')
    parser.add_argument('--unit', choices=['characters', 'tokens'], default='characters')
    parser.add_argument('--tokenizer', type=Path, help='local embedding tokenizer.json; required for token units')
    args = parser.parse_args()
    measure, budget_id = len, 'characters'
    if args.unit == 'tokens':
        if args.tokenizer is None or not args.tokenizer.is_file():
            parser.error('--unit tokens requires an existing local --tokenizer file')
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        measure = lambda text: len(tokenizer.encode(text, add_special_tokens=True).ids)
        budget_id = 'tokens:' + file_hash(args.tokenizer)
    elif args.tokenizer:
        parser.error('--tokenizer requires --unit tokens')
    try:
        records, parsed_manifest = load_parsed(ROOT / 'data' / 'parsed')
        if args.strategy == 'recursive':
            chunks = chunk_recursive(records, args.size, args.overlap, measure, budget_id)
        else:
            chunks = chunk_structured(records, args.size, args.min_size, measure, budget_id)
        if not chunks:
            raise ValueError('No chunks produced; check parsed content.')
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    out = ROOT / 'data' / 'chunks'
    run = {'corpus_id': parsed_manifest['corpus_id'], 'configuration': chunks[0]['metadata']['configuration'],
           'chunker_sha256': file_hash(Path(__file__)),
           'markdown_sha256': file_hash(Path(__file__).with_name('markdown.py'))}
    run_id = fingerprint(run)[:16]
    path = out / f'{args.strategy}-{args.size}-{args.unit}-{run_id}.jsonl'
    write_jsonl(path, chunks)
    lengths = sorted(measure(chunk['text']) for chunk in chunks)
    write_json(path.with_suffix('.manifest.json'), {
        'schema_version': SCHEMA_VERSION, 'run_id': run_id, **run, 'file': path.name,
        'sha256': file_hash(path), 'chunks': len(chunks),
        'lengths': {'median': lengths[len(lengths) // 2], 'max': lengths[-1]}})
    print(f'{len(records)} sections -> {len(chunks)} chunks, median {lengths[len(lengths) // 2]} '
          f'{args.unit}, max {lengths[-1]} -> {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
