"""Module 3 - chunk data/parsed/ into data/chunks/<strategy>-<size>.jsonl.

    uv run scripts/chunk.py --strategy structured --size 1200
    uv run scripts/chunk.py --strategy recursive --size 1200 --overlap 150
"""

from mfem_rag.chunk import main

if __name__ == "__main__":
    main()
