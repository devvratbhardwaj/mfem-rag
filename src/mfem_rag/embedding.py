"""Replaceable embedding interface and explicit local Ollama implementation."""

import math
from typing import Protocol

import ollama

MODEL = 'qwen3-embedding:0.6b'
DIMENSION = 1024
QUERY_INSTRUCTION = ('Given a question about MFEM, PETSc, or SLEPc, retrieve relevant '
                     'documentation and code examples that answer the question.')
QUERY_TEMPLATE = 'Instruct: {instruction}\nQuery: {question}'


def validate_vectors(vectors, count, dimension):
    if len(vectors) != count:
        raise ValueError(f'Embedding response count {len(vectors)} != {count}')
    for vector in vectors:
        if len(vector) != dimension or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in vector):
            raise ValueError(f'Expected finite {dimension}-dimensional embeddings')
        if not any(vector):
            raise ValueError('Zero embedding cannot be used for cosine search')
    return vectors


class EmbeddingProvider(Protocol):
    def configuration(self) -> dict: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, question: str) -> list[float]: ...


class OllamaEmbedding:
    def __init__(self, url='http://localhost:11435', model=MODEL, timeout=120):
        if model != MODEL:
            raise ValueError(f'This configuration requires the exact model tag {MODEL}')
        self.client = ollama.Client(host=url, timeout=timeout)
        self.model = model
        self.digest = self._digest()

    def _digest(self):
        try:
            for model in self.client.list().models:
                if model.model == self.model and model.digest:
                    return model.digest
        except Exception as error:
            raise ValueError('Cannot reach Ollama; start ollama serve or check --ollama-url. '
                             f'No model was installed. Details: {error}') from error
        raise ValueError(f'Ollama model {self.model} is missing; install it manually. '
                         'No model was installed or substituted.')

    def configuration(self):
        return {'provider': 'ollama', 'model': self.model, 'digest': self.digest,
                'dimension': DIMENSION, 'document_prefix': '',
                'query_instruction': QUERY_INSTRUCTION, 'query_template': QUERY_TEMPLATE,
                'truncate': False, 'dimensions': None}

    def _embed(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Embedding inputs must be nonempty strings')
        if self._digest() != self.digest:
            raise ValueError('Local model digest changed; use a separate collection')
        try:
            response = self.client.embed(model=self.model, input=texts, truncate=False)
        except Exception as error:
            raise ValueError(f'Ollama embedding failed (truncate=False): {error}') from error
        if response.model != self.model or self._digest() != self.digest:
            raise ValueError('Ollama response model or local digest changed')
        return validate_vectors(response.embeddings, len(texts), DIMENSION)

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, question):
        if not question.strip():
            raise ValueError('Question must not be empty')
        return self._embed([QUERY_TEMPLATE.format(instruction=QUERY_INSTRUCTION,
                                                 question=question)])[0]
