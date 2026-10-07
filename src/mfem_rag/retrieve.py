"""Semantic retrieval with the collection's exact embedding configuration."""

from contextlib import closing

import argparse
import json

from qdrant_client import QdrantClient

from mfem_rag.embedding import EmbeddingProvider, OllamaEmbedding, validate_vectors
from mfem_rag.index import collection_state, service_options


def retrieve(question: str, collection: str, client, provider: EmbeddingProvider, top_k=5):
    if top_k <= 0:
        raise ValueError('Top-K must be positive')
    state = collection_state(client, collection)
    if state.get('status') != 'complete':
        raise ValueError('Collection indexing is incomplete; rerun its indexing command first')
    if state['configuration']['embedding'] != provider.configuration():
        raise ValueError('Local embedding configuration/digest differs from the collection')
    if client.count(collection, exact=True).count != state['configuration']['chunk_count']:
        raise ValueError('Collection point count changed; rerun indexing to verify it')
    vector = provider.embed_query(question)
    validate_vectors([vector], 1, provider.configuration()['dimension'])
    points = client.query_points(collection, query=vector, limit=top_k,
                                 with_payload=True, with_vectors=False).points
    return [{'score': point.score, 'chunk': point.payload,
             'citations': point.payload['metadata'].get('locations') or [{
                 'url': point.payload['metadata'].get('url'),
                 'page': point.payload['metadata'].get('page')}]} for point in points]


def main():
    parser = argparse.ArgumentParser(description='Retrieve documentation chunks and citations')
    parser.add_argument('question')
    parser.add_argument('--collection', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    service_options(parser)
    args = parser.parse_args()
    try:
        with closing(QdrantClient(url=args.qdrant_url, timeout=30)) as client:
            state = collection_state(client, args.collection)
            provider = OllamaEmbedding(args.ollama_url, model=state['configuration']['embedding']['model'])
            results = retrieve(args.question, args.collection, client, provider, args.top_k)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as error:
        parser.exit(1, f'Retrieval failed: {error}\n')
