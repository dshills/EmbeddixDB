# EmbeddixDB: Specification for a Vector Database in Go

## Table of Contents
1. Overview  
2. Design Goals  
3. Core Concepts  
4. Data Model  
5. Indexing & Search  
6. API Design  
7. Persistence Layer  
8. Vector Operations  
9. Configuration & Optimization  
10. Deployment Modes  
11. Testing & Benchmarking  
12. Future Enhancements

---

## 1. Overview

EmbeddixDB is optimized for LLM (Large Language Model) memory use cases. It is designed to support multiple concurrent projects or agents, with high-performance vector search and low-latency access to embeddings. This makes it suitable for scenarios like semantic memory in agent frameworks, chat history indexing, and fast retrieval for context expansion.

**EmbeddixDB** is a high-performance, dependency-minimal vector database written in Go. It stores high-dimensional vectors and supports efficient approximate nearest neighbor (ANN) and exact search. The system supports indexing, metadata filtering, and vector similarity search (cosine, dot, Euclidean).

---

## 2. Design Goals

- Written in pure Go (except optional SIMD acceleration).
- Pluggable storage backend: in-memory, on-disk (BoltDB/Badger), or custom.
- Support for exact and approximate indexing (e.g., brute-force, HNSW).
- Real-time vector insertion and deletion.
- Metadata-based filtering (tags, labels, payloads).
- Batch import/export of vectors.
- Embeddable and optionally networked (gRPC or HTTP API).
- Designed for use in LLM agent contexts and retrieval-augmented generation (RAG).

---

## 3. Core Concepts

- **Vector**: High-dimensional float32 slice representing an embedding.
- **VectorID**: Unique string identifier (UUID, ULID, etc.).
- **Payload/Metadata**: Arbitrary key-value pairs stored with each vector.
- **Collection**: A named logical grouping of vectors (like a table).
- **Index**: A search structure (exact or approximate) per collection.

---

## 4. Data Model

```go
type Vector struct {
    ID       string            // Unique ID for the vector
    Values   []float32         // Vector data
    Metadata map[string]string // Optional metadata for filtering
}
```

### Collection

```go
type Collection struct {
    Name       string
    Dimension  int
    IndexType  string // e.g., "flat", "hnsw"
    Distance   string // "cosine", "l2", "dot"
    CreatedAt  time.Time
}
```

---

## 5. Indexing & Search

### Indexing Options

- **Flat (Brute-force)**: Exact search, simple, no preprocessing.
- **HNSW**: Fast ANN search, configurable parameters (M, efConstruction, efSearch).
- **IVF** *(optional)*: Centroid-based inverted index.

### Search API

```go
type SearchRequest struct {
    Query     []float32
    TopK      int
    Filter    map[string]string // Match metadata
    IncludeVectors bool         // Return full vector in results
}

type SearchResult struct {
    ID       string
    Score    float32
    Metadata map[string]string
}
```

### Similarity Metrics

- Cosine Similarity  
- L2 (Euclidean) Distance  
- Dot Product  

---

## 6. API Design

### Interface

```go
type VectorStore interface {
    AddVector(ctx context.Context, collection string, vec Vector) error
    GetVector(ctx context.Context, collection, id string) (Vector, error)
    DeleteVector(ctx context.Context, collection, id string) error
    Search(ctx context.Context, collection string, req SearchRequest) ([]SearchResult, error)
    CreateCollection(ctx context.Context, spec Collection) error
    DeleteCollection(ctx context.Context, name string) error
    ListCollections(ctx context.Context) ([]Collection, error)
}
```

### Optional API Protocols

- REST (JSON over HTTP)
- gRPC (Protobuf definitions)
- CLI (basic control and testing)

---

## 7. Persistence Layer

- In-memory only mode for fast ephemeral workloads
- BoltDB or BadgerDB for embedded persistence
- Pluggable interfaces for external durable stores (e.g., MySQL, S3, etc.)

```go
type Persistence interface {
    SaveVector(ctx context.Context, collection string, vec Vector) error
    LoadVectors(ctx context.Context, collection string) ([]Vector, error)
    DeleteVector(ctx context.Context, collection, id string) error
}
```

---

## 8. Vector Operations

### Distance Functions

```go
func CosineSimilarity(a, b []float32) float32
func DotProduct(a, b []float32) float32
func EuclideanDistance(a, b []float32) float32
```

### Validations

- Vector dimension match
- Unique ID enforcement
- Optional: enforce metadata schema

---

## 9. Configuration & Optimization

- Runtime tuning via config file or flags:
  - Index type
  - Distance metric
  - Cache size
  - Parallelism settings
- Rebuild/retrain index after batch insertions
- Optional WAL (write-ahead log) for crash recovery

---

## 10. Deployment Modes

- **Embedded Library Mode**  
  Use as a Go package in another application.

- **Local Server Mode**  
  Start as a binary with gRPC or REST server.

- **Clustered (Future)**  
  Horizontal scaling with sharding and vector routing.

---

## 11. Testing & Benchmarking

- Unit tests for all core modules
- Integration tests for search, filtering, persistence
- Benchmarks:
  - Insert throughput
  - Query latency (Top-K)
  - Recall@K vs. brute-force baseline

```bash
go test -bench=. ./...
```

---

## 12. Future Enhancements

- FAISS backend integration (via CGO)
- Vector compression (PCA, quantization)
- Streaming ingestion
- Vector versioning
- Hybrid search (vector + keyword)
- Distributed clustering and replica management
- OpenTelemetry instrumentation

---

## Suggested Project Structure

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