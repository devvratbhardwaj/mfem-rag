# MFEM-RAG

## Problem Statement

Your job is to design and build a **Retrieval-Augmented Generation (RAG) infrastructure for technical software documentation**.

Use the documentation for **MFEM, PETSc, and SLEPc** as the initial corpus. The goal is not to develop new numerical methods or modify these libraries. The focus is on building the software infrastructure required to ingest, search, retrieve, and reason over large technical documentation collections.

The system should allow a user to ask questions such as:

* How do I configure PETSc GMRES in MFEM?
* What PETSc options control solver tolerances?
* How can PETSc GAMG be used with MFEM?
* How do I solve an eigenvalue problem with SLEPc?

Your system should retrieve relevant documentation and use a local language model, initially **Gemma**, to generate a grounded answer with references to the retrieved sources.

## Your Job

Design the architecture and implement the complete pipeline. At minimum, it should address:

* ingestion of documentation from multiple sources
* document parsing and chunking
* embedding generation and indexing
* semantic retrieval
* context construction
* integration with Gemma
* source and citation tracking
* reproducible evaluation
* a desktop frontend built as an **Electron application using shadcn/ui**

The Electron application should provide a simple interface for:

* asking technical questions
* viewing generated answers
* inspecting cited source documents
* optionally viewing retrieved chunks and retrieval scores
* configuring basic RAG settings such as Top-K retrieval and model selection

You are free to choose the specific libraries, embedding models, databases, chunking methods, retrieval strategies, backend framework, and Electron architecture.

The system should be **modular**, so individual components can be replaced and compared independently.

## Evaluation

Create a benchmark of technical questions using the indexed documentation and evaluate the system using appropriate RAG metrics.

The evaluation should consider:

* whether relevant documents are retrieved
* whether generated answers are supported by the retrieved context
* citation correctness
* hallucination rate
* response quality
* retrieval and generation latency

Experiments should compare a few design choices such as chunking strategies, retrieval methods, embedding models, or Top-K settings.

## Deliverable

The final repository should provide a **reproducible RAG harness for technical documentation** together with a usable **Electron desktop application built with shadcn/ui**.

The repository should include scripts or commands to:

* build the documentation index
* run the RAG backend
* launch the Electron application
* run the benchmark
* reproduce the evaluation results

MFEM, PETSc, and SLEPc are the initial use case, but the infrastructure should be sufficiently general that another technical documentation corpus could be substituted with minimal changes.
