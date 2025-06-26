# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmbeddixDB is a high-performance vector database written in Go, designed for LLM memory use cases and retrieval-augmented generation (RAG). The project has evolved from specification to a fully implemented system with advanced features including feedback collection, personalized search, and comprehensive AI integration.

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

## Code Formatting Guidelines

- Run gofmt -w . often to insure proper formatting

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

## Current Implementation Status

EmbeddixDB has reached **v2.2** with comprehensive feature implementation:

### Core Features ✅ Complete
- Vector storage with HNSW and flat indexing
- Multiple persistence backends (Memory, BoltDB, BadgerDB)
- REST API with comprehensive endpoints
- Distance metrics with SIMD optimization

### AI Integration ✅ Complete  
- ONNX Runtime for embedding inference
- Semantic query understanding and intent classification
- Multi-modal content analysis (text, sentiment, entities)
- Auto-embedding pipeline with batching

### Advanced Retrieval ✅ Complete
- Feedback collection and CTR tracking
- Personalized search with user profiling
- Session management and contextual re-ranking
- Machine learning integration for relevance optimization

### Quality Assurance ✅ Complete
- Comprehensive test suite (83.6% coverage for feedback package)
- Race condition detection and fixes
- Performance benchmarking and optimization
- Build system with coverage reporting

## Documentation References

### Technical Planning Documents
- **Performance Optimization Plan**: `docs/PERFORMANCE_OPTIMIZATION_PLAN.md`
  - Comprehensive 67-page performance roadmap
  - 4-phase implementation strategy (16 weeks)
  - Target: 50% latency reduction, 30% throughput increase
  - Covers query optimization, caching, quantization, GPU acceleration

### Project Roadmap
- **Current Status**: `TODO.md` - Detailed feature implementation status
- **Architecture**: `spec/EMBEDDIXDB_SPEC.md` - Original technical specification  
- **AI Integration**: `docs/AI_INTEGRATION.md` - AI system architecture
- **Implementation**: `docs/IMPLEMENTATION_PLAN.md` - Development roadmap

## Performance Optimization Roadmap

The next major development phase focuses on **Performance Optimizations** as outlined in `docs/PERFORMANCE_OPTIMIZATION_PLAN.md`:

### Immediate Priorities (Next 16 weeks)
1. **Query Optimization**: Plan caching, parallel execution, early termination
2. **Multi-Level Caching**: Semantic caching with personalization awareness  
3. **Index Improvements**: Quantization, hierarchical indexing, incremental updates
4. **GPU Acceleration**: CUDA/OpenCL integration for similarity computations

### Success Targets
- **<100ms p95 latency** for LLM applications
- **>200 QPS throughput** for concurrent agents
- **40% memory reduction** through intelligent caching
- **Zero regression** in search quality or reliability