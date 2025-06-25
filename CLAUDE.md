# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmbeddixDB is a high-performance vector database written in Go, designed for LLM memory use cases and retrieval-augmented generation (RAG). This is currently a specification-only project with no implementation yet.

## Development Commands

Since this is a Go project (go.mod present), use standard Go commands:

```bash
# Build the project
go build ./...

# Run tests
go test ./...

# Run benchmarks
go test -bench=. ./...

# Format code
go fmt ./...

# Vet code for issues
go vet ./...

# Tidy modules
go mod tidy
```

## Architecture Overview

Based on the specification in `spec/EMBEDDIXDB_SPEC.md`, the project will implement:

- **Core Interface**: `VectorStore` interface with operations for vectors and collections
- **Vector Model**: High-dimensional float32 vectors with unique IDs and metadata
- **Collections**: Named logical groupings of vectors with dimension and distance metric configuration
- **Indexing**: Support for flat (brute-force) and HNSW approximate nearest neighbor search
- **Persistence**: Pluggable storage backends (in-memory, BoltDB, BadgerDB)
- **Distance Metrics**: Cosine similarity, L2 distance, and dot product

## Suggested Implementation Structure

According to the spec, the project should be organized as:
```
vectordb/
├── api/              # REST/gRPC interfaces
├── cmd/              # CLI tools
├── core/             # Core logic (index, store, distance)
├── index/            # Index implementations (flat, hnsw)
├── persistence/      # BoltDB, Badger, etc.
├── utils/            # Helper functions
└── main.go
```

## Key Design Goals

- Pure Go implementation (minimal dependencies)
- Pluggable storage backends
- Real-time vector insertion/deletion
- Metadata-based filtering
- Embeddable library or standalone server
- Optimized for LLM agent contexts