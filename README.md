# MFEM-RAG

A local documentation ingestion and semantic retrieval pipeline for MFEM, PETSc,
and SLEPc. The implemented pipeline crawls, parses, chunks, embeds, indexes, and
retrieves documentation with source citations. Grounded answer generation,
evaluation, and the desktop application are planned in
[REQUIREMENTS.md](REQUIREMENTS.md).

## Prerequisites

Install Docker Engine or Docker Desktop with Docker Compose, `uv`, and `curl`.
Run all commands below from the repository root. The project requires Python
3.14 or later; `uv sync` prepares the Python environment.

The Compose services use host ports **11435** (Ollama), **6333** (Qdrant HTTP),
and **6334** (Qdrant gRPC). These ports and the container names
`container-ollama`, `container-qdrant`, and `container-ollama-models` must be available
for the initial setup. Host Ollama can continue using port 11434.

## Setup and pipeline

Complete each numbered step before proceeding to the next.

1. **Start Docker, services, and models.**

   Start Docker Desktop, or start Docker Engine on Linux:

   ```bash
   sudo systemctl start docker
   ```

   Prepare the Python environment and start all Compose services:

   ```bash
   uv sync --locked
   docker compose up -d --pull missing
   docker compose ps -a
   ```

   This starts Ollama and Qdrant and runs model setup. Missing images are pulled;
   locally cached images are reused. The setup container installs missing
   `qwen3-embedding:0.6b` for embeddings and `gemma3:270m` for future generation.
   The default configuration uses CPU.

   Downloads continue in the background. Repeat the status check until
   `container-ollama-models` shows **Exited (0)**. A nonzero exit indicates a setup
   failure; inspect recent output with `docker compose logs --tail 10 models`.
   Verify both services before continuing:

   ```bash
   curl --fail http://localhost:11435/api/tags
   curl --fail http://localhost:6333/collections
   ```

2. **Crawl the documentation.**

   Review source URLs and revisions in `scripts/crawl.py` and the matching
   parser and citation configuration in `sources.toml`, then run:

   ```bash
   uv run scripts/crawl.py
   ```

   Downloaded sources are stored in `data/raw/`.

3. **Parse the documentation.**

   ```bash
   uv run scripts/parse.py
   ```

   Parsed sections and their manifest are stored in `data/parsed/`.

4. **Chunk the parsed documentation.**

   ```bash
   uv run scripts/chunk.py --strategy structured --size 1200 --min-size 300
   ```

   The command prints the generated JSONL path under `data/chunks/`. Keep that
   file and its matching `.manifest.json` together for the next step.

5. **Embed and index the chunks.**

   Index each JSONL in `data/chunks/` into a separate collection named after
   its artifact:

   ```bash
   for CHUNKS in data/chunks/*.jsonl; do
     COLLECTION="mfem-$(basename "$CHUNKS" .jsonl)-qwen3"
     uv run scripts/embed_and_index.py --input "$CHUNKS" --collection "$COLLECTION" --batch-size 16 || break
   done
   ```

   Indexing embeds the chunks with Ollama and stores the vectors, text, and
   citation metadata in Qdrant. It validates the matching manifest and file hash
   before indexing. Each JSONL contains the corpus chunks for one
   chunking configuration, and each artifact is indexed into its own collection.
   Use a new collection name if the embedding configuration/model digest changes.

6. **Retrieve documentation.**

   After indexing completes, this queries the last collection indexed in the
   same terminal. To query another, set `COLLECTION` to its name first:

   ```bash
   uv run scripts/retrieve.py --collection "$COLLECTION" --top-k 5 \
     "How do I configure PETSc GMRES in MFEM?"
   ```

   Results include chunk text, retrieval scores, and source citations.

## Compare chunk strategies

Keep structured and recursive chunks in separate collections. To generate a
recursive artifact from the same parsed corpus:

```bash
uv run scripts/chunk.py --strategy recursive --size 1200 --overlap 150
```

Repeat step 5 to index the new artifact; previously indexed chunks are skipped.
The filename-based naming creates separate collections such as:

- `mfem-structured-1200-characters-<run-id>-qwen3`
- `mfem-recursive-1200-characters-<run-id>-qwen3`

Query each collection with the same question to compare retrieval results.
The indexer rejects an existing collection if its artifact or embedding
configuration differs; it does not overwrite that collection.

## Persistence and subsequent runs

Qdrant stores embeddings in the host directory `./qdrant_storage`. Ollama stores
models in the named Docker volume `mfem-rag_ollama_models`. Both persist across
container restarts and plain `docker compose down`.

On subsequent runs, start services using the command in step 1 and reuse existing
artifacts to skip completed ingestion steps. Rerun the same indexing command to
resume an interrupted run; matching stored chunks are skipped. Run one indexer
per collection.

Set `QDRANT_STORAGE` to an absolute directory before starting Compose to use a
different Qdrant storage location. When migrating existing Qdrant data, use its
original storage directory and stop the old container before starting Compose;
only one running Qdrant container may use that directory.

## Optional model and service settings

To install `gemma4:12b` as an additional generation option:

```bash
RAG_MODEL=gemma4:12b docker compose up -d --pull missing
```

Check model setup status as in step 1. `RAG_MODEL` selects the generation model
to install; answer generation is not yet implemented in the Python pipeline.

The indexing and retrieval CLIs support `--ollama-url` or `OLLAMA_URL`.
Indexing and retrieval also support `--qdrant-url` or `QDRANT_URL`.
Defaults are `http://localhost:11435` and `http://localhost:6333`.
Qdrant dashboard: http://localhost:6333/dashboard.

## Shutdown and removal

To stop the Compose containers while keeping them:

```bash
docker compose stop
```

To stop and remove the Compose containers and network:

```bash
docker compose down
```

Plain `down` keeps images, downloaded models, and Qdrant data. Do not add
`--volumes` or `--rmi` when retaining models and images. To start again, rerun
`docker compose up -d --pull missing` with the same storage configuration.

## References

- [Ollama Docker](https://docs.ollama.com/docker)
- [Docker Compose startup](https://docs.docker.com/reference/cli/docker/compose/up/)
- [Docker Compose shutdown](https://docs.docker.com/reference/cli/docker/compose/down/)
- [Qdrant quickstart](https://qdrant.tech/documentation/quickstart/)
