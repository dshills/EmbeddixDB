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

### Advanced Index Optimizations ✅ Phase 4.1-4.2 Complete
- Product Quantization with 256x memory compression
- Quantized HNSW index with reranking pipeline
- Scalar quantization for memory-efficient storage
- Multi-level caching layer for improved performance

## Documentation References

### Technical Planning Documents
- **Performance Optimization Plan**: `docs/PERFORMANCE_OPTIMIZATION_PLAN.md`
  - Comprehensive 67-page performance roadmap
  - 4-phase implementation strategy (16 weeks)
  - Target: 50% latency reduction, 30% throughput increase
  - Covers query optimization, caching, quantization, GPU acceleration
- **Phase 4 Implementation Plan**: `docs/PHASE4_IMPLEMENTATION_PLAN.md`
  - Advanced index optimizations (Product Quantization, Hierarchical HNSW, GPU)
  - Detailed implementation status for completed Phase 4.1-4.2
- **Phase 4 Completion Status**: `docs/PHASE4_COMPLETION_STATUS.md`
  - Complete status report for quantization implementation
  - 256x memory compression achievement documentation
- **Quantization API Documentation**: `docs/QUANTIZATION_API.md`
  - Comprehensive API guide for quantized indexes
  - Configuration examples and performance tuning

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

## EmbeddixDB Memory Usage for This Project

This project uses its own EmbeddixDB instance for persistent memory storage. When working on this codebase, use the MCP tools to store and retrieve relevant information.

### Available MCP Tools

You have access to these database tools via MCP:

#### EmbeddixDB (Vector Database)
- `mcp__embeddixdb__search_vectors` - Search for similar memories using semantic similarity
- `mcp__embeddixdb__add_vectors` - Store new memories with auto-embedding
- `mcp__embeddixdb__get_vector` - Retrieve specific memories by ID
- `mcp__embeddixdb__delete_vector` - Remove outdated memories
- `mcp__embeddixdb__create_collection` - Create new memory collections
- `mcp__embeddixdb__list_collections` - View existing collections
- `mcp__embeddixdb__delete_collection` - Remove collections

#### RelatixDB (Graph Database)
- `mcp__relatixdb__add_node` - Add nodes with ID, type, and properties
- `mcp__relatixdb__add_edge` - Create directed edges between nodes with labels
- `mcp__relatixdb__delete_node` - Remove nodes and their edges
- `mcp__relatixdb__delete_edge` - Remove specific edges
- `mcp__relatixdb__query_neighbors` - Find connected nodes
- `mcp__relatixdb__query_paths` - Find paths between nodes
- `mcp__relatixdb__query_find` - Search nodes by type or properties

### Memory Collections Schema

Create and use these collections for project memory:

1. **`embeddixdb_development`** - Technical decisions and implementation details
   - Dimension: 384 (or match your embedding model)
   - Distance: cosine
   - Metadata fields:
     - `type`: "architecture" | "implementation" | "bug_fix" | "optimization" | "decision"
     - `component`: Specific module/package (e.g., "core", "mcp", "api", "quantization")
     - `version`: Project version when added
     - `importance`: "high" | "medium" | "low"
     - `tags`: Array of relevant keywords

2. **`embeddixdb_conversations`** - User interactions and context
   - Dimension: 384
   - Distance: cosine
   - Metadata fields:
     - `session_id`: Current conversation session
     - `user_intent`: Detected intent of the request
     - `outcome`: Result or resolution
     - `timestamp`: When the interaction occurred

3. **`embeddixdb_issues`** - Known issues and their solutions
   - Dimension: 384
   - Distance: cosine
   - Metadata fields:
     - `status`: "open" | "resolved" | "workaround"
     - `error_type`: Category of issue
     - `solution`: How it was fixed
     - `affected_versions`: Versions with this issue

### Usage Examples

#### Storing Development Decisions
```json
{
  "collection": "embeddixdb_development",
  "vectors": [{
    "content": "Implemented Product Quantization for HNSW index achieving 256x memory compression. Uses 256 centroids with 8-bit scalar quantization for subvectors. Maintains 95% recall@10 with proper reranking.",
    "metadata": {
      "type": "optimization",
      "component": "quantization",
      "version": "v2.2",
      "importance": "high",
      "tags": ["performance", "memory", "indexing", "phase4"]
    }
  }]
}
```

#### Searching for Implementation Details
```json
{
  "collection": "embeddixdb_development",
  "query": "How is quantization implemented in HNSW?",
  "limit": 5,
  "filters": {
    "component": "quantization",
    "type": "implementation"
  }
}
```

#### Storing Issue Resolutions
```json
{
  "collection": "embeddixdb_issues",
  "vectors": [{
    "content": "Race condition in feedback collector when concurrent goroutines update click-through rates. Fixed by adding mutex protection in UpdateClickThroughRate method.",
    "metadata": {
      "status": "resolved",
      "error_type": "concurrency",
      "solution": "Added sync.Mutex to FeedbackCollector struct",
      "affected_versions": ["v2.0", "v2.1"]
    }
  }]
}
```

### Best Practices

1. **Always search before implementing** - Check if similar problems have been solved:
   ```json
   {
     "collection": "embeddixdb_development",
     "query": "concurrency issues in feedback system",
     "limit": 10
   }
   ```

2. **Store significant decisions** - Document why certain approaches were chosen:
   ```json
   {
     "collection": "embeddixdb_development",
     "vectors": [{
       "content": "Chose BoltDB over BadgerDB for default persistence due to simpler API and better stability for embedded use cases. BadgerDB offers better performance but requires more careful tuning.",
       "metadata": {
         "type": "decision",
         "component": "persistence",
         "importance": "high",
         "tags": ["storage", "performance", "tradeoffs"]
       }
     }]
   }
   ```

3. **Track conversation context** - Maintain continuity across interactions:
   ```json
   {
     "collection": "embeddixdb_conversations",
     "vectors": [{
       "content": "User requested implementation of GPU acceleration for similarity search. Discussed CUDA kernel design and memory transfer optimization strategies.",
       "metadata": {
         "session_id": "session_123",
         "user_intent": "performance_optimization",
         "outcome": "planned_for_phase5",
         "timestamp": "2024-01-15T10:00:00Z"
       }
     }]
   }
   ```

4. **Regular memory search** - Before starting any task, search relevant memories:
   - Search for similar features/implementations
   - Check for known issues in the component
   - Review past decisions and their rationales

5. **Update memories** - When implementations change, add new memories linking to old ones:
   ```json
   {
     "collection": "embeddixdb_development",
     "vectors": [{
       "content": "Refactored quantization to use interface-based design, replacing hardcoded PQ implementation. Now supports multiple quantization strategies via Quantizer interface.",
       "metadata": {
         "type": "implementation",
         "component": "quantization",
         "version": "v2.3",
         "importance": "medium",
         "tags": ["refactoring", "interfaces", "extensibility"],
         "replaces": "vector_id_of_old_implementation"
       }
     }]
   }
   ```

### Memory Maintenance

- Periodically search for outdated information and update it
- Use metadata filters to find memories by component when refactoring
- Tag memories with version numbers to track evolution
- Set importance levels to prioritize retrieval of critical information

## RelatixDB Graph Database Usage

RelatixDB complements EmbeddixDB by providing graph-based relationship modeling for code structure and dependencies.

### Graph Schema for Code Analysis

#### Node Types
- **`file`** - Source code files
  - Properties: `path`, `package`, `language`, `size`, `last_modified`
- **`function`** - Function/method definitions
  - Properties: `name`, `signature`, `visibility`, `line_start`, `line_end`
- **`type`** - Struct/interface/type definitions
  - Properties: `name`, `kind` (struct/interface/type), `package`
- **`package`** - Go packages
  - Properties: `name`, `path`, `version`
- **`test`** - Test cases
  - Properties: `name`, `test_type` (unit/integration/benchmark)

#### Edge Labels
- **`contains`** - File contains function/type
- **`calls`** - Function calls another function
- **`implements`** - Type implements interface
- **`imports`** - File/package imports another
- **`tests`** - Test case tests function/type
- **`depends_on`** - Package depends on another
- **`uses`** - Function uses type

### Usage Examples

#### Mapping Code Structure
```json
// Add a file node
{
  "id": "core/store.go",
  "type": "file",
  "props": {
    "path": "core/store.go",
    "package": "core",
    "language": "go",
    "size": "1024"
  }
}

// Add a function node
{
  "id": "core.VectorStore.Add",
  "type": "function",
  "props": {
    "name": "Add",
    "signature": "func (s *VectorStore) Add(ctx context.Context, vector *Vector) error",
    "visibility": "public",
    "line_start": "45",
    "line_end": "62"
  }
}

// Connect file to function
{
  "from": "core/store.go",
  "to": "core.VectorStore.Add",
  "label": "contains"
}
```

#### Tracking Dependencies
```json
// Find all functions that call a specific function
{
  "node": "core.VectorStore.Add",
  "direction": "in",
  "label": "calls"
}

// Find import dependencies of a package
{
  "node": "quantization",
  "direction": "out",
  "label": "imports"
}
```

#### Analyzing Test Coverage
```json
// Add test relationship
{
  "from": "core/store_test.TestAdd",
  "to": "core.VectorStore.Add",
  "label": "tests"
}

// Find all untested functions
// First get all functions, then check which have no incoming "tests" edges
```

#### Finding Impact of Changes
```json
// Find all code that might be affected by changing a function
{
  "node": "core.VectorStore.Search",
  "direction": "in",
  "label": "calls"
}

// Find path between two components
{
  "from": "api/handlers.go",
  "to": "persistence/boltdb.go",
  "max_depth": 5
}
```

### Best Practices for RelatixDB

1. **Consistent Node IDs** - Use hierarchical IDs like `package.Type.Method`
2. **Property Standards** - Keep property names consistent across node types
3. **Bidirectional Relationships** - Consider if edges should be queried in both directions
4. **Regular Updates** - Update graph when code structure changes
5. **Combine with EmbeddixDB** - Use RelatixDB for structure, EmbeddixDB for semantic search

### Integration Example

Combining both databases for comprehensive code intelligence:
```json
// 1. Use RelatixDB to find all functions in a module
{
  "type": "function",
  "props": {
    "package": "quantization"
  }
}

// 2. For each function, use EmbeddixDB to find similar implementations
{
  "collection": "embeddixdb_development",
  "query": "quantization algorithm implementation similar to ProductQuantizer.Encode",
  "limit": 5
}

// 3. Use RelatixDB to trace call paths between similar functions
{
  "from": "quantization.ProductQuantizer.Encode",
  "to": "index.HNSW.addPoint",
  "max_depth": 4
}
```

## Memory Usage Tools

- **Use EmbeddixDB for semantic search and memory storage**
- **Use RelatixDB for structural relationships and dependency analysis**