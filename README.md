# EmbeddixDB

[![Go Version](https://img.shields.io/badge/go-1.21+-blue.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-brightgreen.svg)](https://modelcontextprotocol.io)

A high-performance vector database designed for LLM memory and RAG applications. EmbeddixDB provides an MCP (Model Context Protocol) server for seamless integration with AI assistants like Claude, plus a REST API for traditional applications.

## Features

- **🧠 MCP Server**: Direct integration with Claude and other AI assistants via Model Context Protocol
- **🚀 Vector Search**: HNSW and flat indexes with 256x memory compression via quantization
- **💾 Flexible Storage**: In-memory, BoltDB, or BadgerDB persistence backends
- **🤖 Auto-Embedding**: Automatic text-to-vector conversion with Ollama or ONNX models
- **📊 Advanced Analytics**: Sentiment analysis, entity extraction, and topic modeling
- **🔄 Real-time Operations**: Live vector insertion, updates, and deletion
- **🎯 High Performance**: ~65,000 queries/sec on M1 MacBook Pro

## Quick Start

### 1. MCP Server for AI Assistants

The MCP server is the primary way to use EmbeddixDB with Claude and other AI assistants.

#### Install and Run

```bash
# Clone and build
git clone https://github.com/dshills/EmbeddixDB.git
cd EmbeddixDB
make build-mcp

# Run with persistent storage
./build/embeddix-mcp -persistence bolt -data ./data/embeddix.db

# Or run with in-memory storage for testing
./build/embeddix-mcp -persistence memory
```

#### Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "embeddixdb": {
      "command": "/path/to/embeddix-mcp",
      "args": ["-persistence", "bolt", "-data", "/path/to/data/embeddix.db"]
    }
  }
}
```

#### Available MCP Tools

- `create_collection` - Create a new vector collection
- `add_vectors` - Add vectors with automatic embedding from text
- `search_vectors` - Semantic similarity search
- `get_vector` - Retrieve specific vectors
- `delete_vector` - Remove vectors
- `list_collections` - List all collections
- `delete_collection` - Remove collections

### 2. REST API Server

For traditional applications or when you need a REST API:

```bash
# Using Docker
docker-compose up -d

# Or build from source
make build
./build/embeddix-api
```

## Configuration

Create `~/.embeddixdb.yml` for persistent configuration:

```yaml
# Basic configuration
persistence:
  type: "bolt"              # or "badger", "memory"
  path: "~/embeddix/data"

# Optional: Enable auto-embedding with Ollama
ai:
  embedding:
    engine: "ollama"
    model: "nomic-embed-text"
    ollama:
      endpoint: "http://localhost:11434"
```

Or use command-line flags:

```bash
# MCP server with flags
./build/embeddix-mcp -persistence bolt -data ./data/embeddix.db -verbose

# API server with custom config
./build/embeddix-api -config /path/to/config.yml
```

## MCP Usage Examples

### 1. LLM Memory Storage

Perfect for storing conversation history, user preferences, and session context:

```python
# Store conversation context
mcp.call_tool("add_vectors", {
    "collection": "conversations",
    "vectors": [{
        "content": "User prefers Python examples with type hints",
        "metadata": {
            "user_id": "user123",
            "type": "preference",
            "confidence": 0.9
        }
    }]
})

# Retrieve relevant context
context = mcp.call_tool("search_vectors", {
    "collection": "conversations",
    "query": "What programming language does the user prefer?",
    "limit": 5
})
```

### 2. RAG (Retrieval-Augmented Generation)

Index documents and retrieve relevant chunks:

```python
# Index document chunks
mcp.call_tool("add_vectors", {
    "collection": "docs",
    "vectors": [{
        "content": "EmbeddixDB uses HNSW algorithm for fast similarity search...",
        "metadata": {
            "source": "technical_guide.pdf",
            "page": 42,
            "section": "algorithms"
        }
    }]
})

# Query for information
results = mcp.call_tool("search_vectors", {
    "collection": "docs",
    "query": "How does EmbeddixDB perform similarity search?",
    "limit": 3
})
```

### 3. Tool Learning

Track successful tool usage patterns:

```python
# Record tool usage
mcp.call_tool("add_vectors", {
    "collection": "tool_memory",
    "vectors": [{
        "content": "Successfully used regex pattern '^[A-Z].*\\.$' to match sentences",
        "metadata": {
            "tool": "regex",
            "pattern": "^[A-Z].*\\.$",
            "success": true,
            "use_case": "sentence_matching"
        }
    }]
})
```

## REST API Examples

The REST API is available when running the API server (`embeddix-api`).

### Basic Operations

```bash
# Create a collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384}'

# Add vectors with auto-embedding
curl -X POST http://localhost:8080/collections/docs/documents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "content": "EmbeddixDB is a high-performance vector database",
    "metadata": {"category": "intro"}
  }'

# Search with natural language
curl -X POST http://localhost:8080/collections/docs/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is EmbeddixDB?", "top_k": 5}'
```

### API Documentation

- **Interactive Docs**: http://localhost:8080/docs
- **API Reference**: http://localhost:8080/redoc

## Architecture

- **MCP Server**: Stdio-based server implementing Model Context Protocol for AI assistants
- **Vector Store**: Core interface supporting HNSW and flat indexes with quantization
- **Persistence**: Pluggable backends - Memory, BoltDB, or BadgerDB
- **AI Integration**: ONNX Runtime and Ollama for automatic text embeddings
- **REST API**: Full-featured HTTP API with OpenAPI documentation

## Performance

On Apple M4 Pro (64GB RAM):
- **Search Speed**: ~25,374 queries/sec (concurrent, flat index, 1K vectors)
- **Insert Speed**: ~32,113 vectors/sec (batch mode)
- **Get Vector**: ~13.5M ops/sec
- **Memory Usage**: 5.6 MB for 1K vectors (128 dimensions)
- **Latency**: Search P95 < 180µs, Insert P95 < 6ms

Run benchmarks on your hardware:
```bash
make benchmark
```


## Ollama Integration

EmbeddixDB supports automatic text embedding using locally-hosted Ollama models:

```bash
# 1. Install and start Ollama
ollama serve

# 2. Pull an embedding model
ollama pull nomic-embed-text

# 3. Configure EmbeddixDB
```

Add to `~/.embeddixdb.yml`:
```yaml
ai:
  embedding:
    engine: "ollama"
    model: "nomic-embed-text"
```

## Development

```bash
# Prerequisites: Go 1.21+

# Run tests
make test

# Build everything
make build

# Development mode with hot reload
make run-dev
```

## Documentation

- [MCP API Documentation](docs/MCP_API.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Performance Tuning](docs/PERFORMANCE.md)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">
  Built for AI assistants and LLM applications
</p>
