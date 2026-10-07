"""Validate chunk artifacts and persist externally generated vectors in Qdrant.

Collection metadata binds each collection to one artifact/configuration. Run only
one indexer per collection at a time. Interrupted batches can safely be replayed.
"""

from contextlib import closing

import argparse
import json
import os
from pathlib import Path
import uuid

from qdrant_client import QdrantClient, models

from mfem_rag.artifacts import SCHEMA_VERSION, file_hash, fingerprint
from mfem_rag.embedding import EmbeddingProvider, OllamaEmbedding, validate_vectors


def load_chunks(path: Path):
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('file') != path.name
            or manifest.get('sha256') != file_hash(path)):
        raise ValueError('Chunk manifest/schema/hash mismatch; select a matching JSONL/manifest pair')
    chunks = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    seen = set()
    for chunk in chunks:
        if (not isinstance(chunk, dict) or not isinstance(chunk.get('id'), str)
                or not chunk['id'] or chunk['id'] in seen
                or not isinstance(chunk.get('text'), str) or not chunk['text'].strip()
                or not isinstance(chunk.get('metadata'), dict)):
            raise ValueError('Invalid chunk record or duplicate chunk ID')
        seen.add(chunk['id'])
    if not chunks or manifest.get('chunks') != len(chunks):
        raise ValueError('Chunk manifest count mismatch or empty artifact')
    return chunks, manifest


def point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'mfem-rag:chunk:' + chunk_id))


def collection_state(client, collection):
    info = client.get_collection(collection)
    state = (info.config.metadata or {}).get('mfem_rag')
    if not isinstance(state, dict) or state.get('schema_version') != 1:
        raise ValueError('Collection has no supported MFEM-RAG metadata; choose another collection')
    vectors = info.config.params.vectors
    config = state.get('configuration', {})
    if (not isinstance(vectors, models.VectorParams)
            or vectors.size != config.get('embedding', {}).get('dimension')
            or vectors.distance != models.Distance.COSINE):
        raise ValueError('Collection vector configuration is incompatible')
    if state.get('identity') != fingerprint(config):
        raise ValueError('Collection metadata identity is invalid')
    return state


def save_state(client, collection, state):
    client.update_collection(collection, metadata={'mfem_rag': state})


def index_chunks(path: Path, collection: str, client, provider: EmbeddingProvider, batch_size=16):
    if batch_size <= 0:
        raise ValueError('Batch size must be positive')
    chunks, manifest = load_chunks(path)
    configuration = {'artifact_sha256': manifest['sha256'],
                     'manifest_sha256': file_hash(path.with_suffix('.manifest.json')),
                     'chunk_count': len(chunks), 'embedding': provider.configuration()}
    identity = fingerprint(configuration)
    if client.collection_exists(collection):
        state = collection_state(client, collection)
        if state['configuration'] != configuration:
            raise ValueError('Collection belongs to another artifact/embedding configuration; '
                             f'choose a separate collection (configuration {identity[:16]})')
    else:
        state = {'schema_version': 1, 'identity': identity, 'configuration': configuration,
                 'status': 'incomplete', 'indexed_chunks': 0}
        client.create_collection(collection,
                                 vectors_config=models.VectorParams(
                                     size=configuration['embedding']['dimension'],
                                     distance=models.Distance.COSINE),
                                 metadata={'mfem_rag': state})
        # Refuse to proceed if the server cannot persist collection metadata.
        collection_state(client, collection)
    state['status'] = 'incomplete'
    state.pop('error', None)
    save_state(client, collection, state)
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            existing = client.retrieve(collection, ids=[point_id(c['id']) for c in batch],
                                       with_payload=True, with_vectors=False)
            for point in existing:
                expected = next(c for c in batch if point_id(c['id']) == str(point.id))
                if point.payload != expected:
                    raise ValueError('Existing point payload does not match selected artifact')
            present = {str(point.id) for point in existing}
            missing = [c for c in batch if point_id(c['id']) not in present]
            if missing:
                vectors = provider.embed_documents([c['text'] for c in missing])
                validate_vectors(vectors, len(missing), configuration['embedding']['dimension'])
                client.upsert(collection, points=[models.PointStruct(
                    id=point_id(chunk['id']), vector=vector, payload=chunk)
                    for chunk, vector in zip(missing, vectors, strict=True)], wait=True)
            state['indexed_chunks'] = start + len(batch)
            save_state(client, collection, state)
        if client.count(collection, exact=True).count != len(chunks):
            raise ValueError('Collection point count does not match artifact')
        state['status'] = 'complete'
        save_state(client, collection, state)
    except Exception as error:
        state['error'] = str(error)
        try:
            save_state(client, collection, state)
        except Exception:
            pass  # Retain the original failure; previously saved status remains incomplete.
        raise
    return state


def service_options(parser):
    parser.add_argument('--ollama-url', default=os.getenv('OLLAMA_URL', 'http://localhost:11435'))
    parser.add_argument('--qdrant-url', default=os.getenv('QDRANT_URL', 'http://localhost:6333'))


def main():
    parser = argparse.ArgumentParser(description='Embed and index one explicit chunk artifact')
    parser.add_argument('--input', required=True, type=Path, help='JSONL with matching .manifest.json')
    parser.add_argument('--batch-size', type=int, default=16)
    service_options(parser)
    parser.add_argument('--collection', required=True,
                        help='dedicated name for this artifact and embedding configuration')
    args = parser.parse_args()
    try:
        # Validate before contacting either service.
        load_chunks(args.input)
        with closing(QdrantClient(url=args.qdrant_url, timeout=30)) as client:
            client.get_collections()
            state = index_chunks(args.input, args.collection, client,
                                 OllamaEmbedding(args.ollama_url), args.batch_size)
        print(json.dumps({'collection': args.collection, **state}, indent=2))
    except Exception as error:
        parser.exit(1, f'Indexing failed: {error}\n')
