# EmbeddixDB

[![Go Version](https://img.shields.io/badge/go-1.21+-blue.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

A high-performance vector database with integrated AI capabilities, designed for LLM memory storage and intelligent document processing. Written in Go, EmbeddixDB provides efficient similarity search, persistent storage, auto-embedding, and advanced content analysis.

**⚠️ Pre-Alpha Notice**: This software is in active development and APIs may change. Use in production at your own risk.

## Features

### 🚀 **Core Vector Database**
- **High-Performance Vector Search**: Supports both flat (brute-force) and HNSW (Hierarchical Navigable Small World) indexes
- **Memory-Optimized Indexing**: Quantized HNSW with 256x memory compression through Product Quantization
- **Multiple Distance Metrics**: Cosine similarity, Euclidean (L2), and Dot product with SIMD optimizations
- **Persistent Storage**: Choose from in-memory, BoltDB, or BadgerDB backends
- **Write-Ahead Logging (WAL)**: Ensures data durability and crash recovery
- **RESTful API**: Easy integration with any application
- **Batch Operations**: Efficient bulk vector insertions

### 🤖 **AI Integration**
- **ONNX Runtime Integration**: Production-ready embedding inference with real transformer models
- **Ollama Integration**: Use locally-hosted Ollama models for flexible embedding generation
- **Auto-Embedding API**: Automatic vector generation from text content
- **Content Analysis Pipeline**: Advanced text understanding with sentiment analysis, entity extraction, and topic modeling
- **Model Management**: Support for popular architectures (BERT, RoBERTa, Sentence Transformers, etc.)
- **Intelligent Preprocessing**: Language detection, text segmentation, and key phrase extraction
- **Semantic Query Understanding**: Intent classification and query expansion

### 🔧 **Operations & Monitoring**
- **Configuration Management**: YAML config files with environment variable overrides
- **Data Migration**: Built-in tools for backup, restore, and schema evolution
- **Docker Support**: Production-ready containers with compose configurations
- **Performance Monitoring**: Comprehensive benchmarking suite
- **Graceful Shutdown**: Proper resource cleanup and connection draining

### 🤝 **Model Context Protocol (MCP)**
- **MCP Server**: Enable LLMs to use EmbeddixDB as a memory backend
- **Standardized Tools**: 7 core tools for vector operations via MCP
- **Auto-Embedding Support**: Automatic text-to-vector conversion in MCP handlers
- **Language Agnostic**: Works with any MCP-compatible client
- **Claude Desktop Integration**: Direct integration with Claude Desktop app
- **Flexible Deployment**: Stdio-based communication for universal compatibility

## Quick Start

### Using Docker (Recommended)

```bash
# Start EmbeddixDB with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8080/health

# Stop the service
docker-compose down
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/dshills/EmbeddixDB.git
cd EmbeddixDB

# Install dependencies
go mod download

# Build the server
make build

# Run the server with default configuration
./build/embeddix-api

# Or with custom configuration file
./build/embeddix-api -config /path/to/config.yml

# Or with command-line overrides
./build/embeddix-api -host 0.0.0.0 -port 8080 -db bolt -path data/embeddix.db
```

## Configuration

EmbeddixDB supports flexible configuration through multiple sources with clear precedence:

1. **Command-line flags** (highest priority) - Override any other settings
2. **Environment variables** - Useful for containerized deployments
3. **Configuration file** (`~/.embeddixdb.yml`) - Primary configuration method
4. **Default values** (lowest priority) - Sensible defaults for all settings

### Configuration File

Create a configuration file at `~/.embeddixdb.yml` or specify a custom path with the `-config` flag:

```yaml
# Example configuration for Ollama integration
server:
  host: "0.0.0.0"
  port: 8080

persistence:
  type: "bolt"
  path: "data/embeddix.db"

ai:
  embedding:
    engine: "ollama"
    model: "nomic-embed-text"
    ollama:
      endpoint: "http://localhost:11434"
```

See [`.embeddixdb.yml.example`](.embeddixdb.yml.example) for a complete configuration reference with all available options.

### Environment Variables

Key configuration options can be set via environment variables:

```bash
export EMBEDDIXDB_HOST=0.0.0.0
export EMBEDDIXDB_PORT=8080
export EMBEDDIXDB_OLLAMA_ENDPOINT=http://localhost:11434
export EMBEDDIXDB_PERSISTENCE_BACKEND=bolt
export EMBEDDIXDB_PERSISTENCE_PATH=data/embeddix.db
```

### Ollama Integration

EmbeddixDB seamlessly integrates with Ollama for local embedding generation:

#### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)

2. **Start Ollama service**:
```bash
ollama serve
```

3. **Pull an embedding model**:
```bash
# Recommended models for embeddings
ollama pull nomic-embed-text        # 384 dimensions, fast
ollama pull mxbai-embed-large       # 1024 dimensions, high quality
ollama pull all-minilm              # 384 dimensions, multilingual
```

#### Configuration

Configure EmbeddixDB to use Ollama in your `~/.embeddixdb.yml`:

```yaml
ai:
  embedding:
    engine: "ollama"
    model: "nomic-embed-text"  # Model name from ollama list
    batch_size: 32             # Process texts in batches
    ollama:
      endpoint: "http://localhost:11434"
      timeout: 30s
      # api_key: "optional-key"  # If authentication is enabled
```

#### Verify Setup

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Test embedding generation
curl -X POST http://localhost:8080/ai/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"]}'  
```

## Model Context Protocol (MCP) Server

EmbeddixDB includes an MCP server that enables LLMs to use the vector database as a memory backend through standardized tools.

### MCP Quick Start

```bash
# Build the MCP server
make build-mcp

# Run with default configuration from ~/.embeddixdb.yml
./build/embeddix-mcp

# Run with custom configuration file
./build/embeddix-mcp -config /path/to/config.yml

# Run with in-memory storage (for testing)
./build/embeddix-mcp -persistence memory

# Run with persistent storage and verbose output
./build/embeddix-mcp -persistence bolt -data ./data -verbose
```

The MCP server now uses the same configuration system as the API server, automatically enabling embedding support if configured in `~/.embeddixdb.yml`.

### MCP Tools Available

The MCP server exposes the following tools for LLM integration:

1. **create_collection** - Create a new vector collection
2. **add_vectors** - Add vectors with metadata
3. **search_vectors** - Semantic similarity search
4. **get_vector** - Retrieve specific vectors
5. **delete_vector** - Remove vectors
6. **list_collections** - List all collections
7. **delete_collection** - Remove collections

### Using with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "embeddixdb": {
      "command": "/path/to/embeddix-mcp",
      "args": ["-config", "/path/to/.embeddixdb.yml", "-verbose"]
    }
  }
}
```

Or for a simpler setup without configuration file:

```json
{
  "mcpServers": {
    "embeddixdb": {
      "command": "/path/to/embeddix-mcp",
      "args": ["-persistence", "bolt", "-data", "/path/to/data"]
    }
  }
}
```

### MCP Usage Examples

#### Python Client Example

```python
import json
import subprocess

class EmbeddixMCP:
    def __init__(self, server_path="./build/embeddix-mcp"):
        self.process = subprocess.Popen(
            [server_path, "-persistence", "memory"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        # Initialize connection
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "python-client", "version": "1.0"}
        })
    
    def _call(self, method, params=None):
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1,
            "params": params or {}
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        return response.get("result")
    
    def call_tool(self, tool_name, arguments):
        return self._call("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

# Usage example
client = EmbeddixMCP()

# Create a collection for conversation memory
client.call_tool("create_collection", {
    "name": "conversation_memory",
    "dimension": 384,
    "distance": "cosine"
})

# Store conversation context
client.call_tool("add_vectors", {
    "collection": "conversation_memory",
    "vectors": [{
        "content": "User prefers technical explanations with examples",
        "metadata": {
            "type": "user_preference",
            "timestamp": "2024-01-15T10:30:00Z",
            "confidence": 0.9
        }
    }]
})

# Search for relevant context
results = client.call_tool("search_vectors", {
    "collection": "conversation_memory",
    "query": "What kind of explanations does the user prefer?",
    "limit": 5
})
```

#### Node.js Example

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

class EmbeddixMCP {
  constructor(serverPath = './build/embeddix-mcp') {
    this.process = spawn(serverPath, ['-persistence', 'memory']);
    this.rl = readline.createInterface({
      input: this.process.stdout,
      output: this.process.stdin
    });
    this.requestId = 0;
  }

  async call(method, params = {}) {
    const request = {
      jsonrpc: '2.0',
      method: method,
      params: params,
      id: ++this.requestId
    };
    
    return new Promise((resolve) => {
      this.rl.once('line', (line) => {
        const response = JSON.parse(line);
        resolve(response.result);
      });
      this.process.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  async callTool(name, args) {
    return this.call('tools/call', { name, arguments: args });
  }
}

// Usage
async function main() {
  const mcp = new EmbeddixMCP();
  
  // Initialize
  await mcp.call('initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'nodejs-client', version: '1.0' }
  });
  
  // Create RAG collection
  await mcp.callTool('create_collection', {
    name: 'documents',
    dimension: 768
  });
  
  // Add document chunks
  await mcp.callTool('add_vectors', {
    collection: 'documents',
    vectors: [
      {
        content: 'EmbeddixDB is a vector database designed for LLM applications',
        metadata: { doc_id: '1', chunk: 0 }
      }
    ]
  });
}
```

### MCP Integration Patterns

#### 1. LLM Memory System
Store and retrieve conversation context, user preferences, and session information:

```python
# Store user interaction
client.call_tool("add_vectors", {
    "collection": "user_memory",
    "vectors": [{
        "content": user_message,
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "message_type": "user"
        }
    }]
})

# Retrieve relevant memories
memories = client.call_tool("search_vectors", {
    "collection": "user_memory",
    "query": current_context,
    "limit": 10,
    "filters": {"user_id": user_id}
})
```

#### 2. RAG (Retrieval-Augmented Generation)
Index documents and retrieve relevant chunks for context:

```python
# Index document chunks
for chunk in document_chunks:
    client.call_tool("add_vectors", {
        "collection": "knowledge_base",
        "vectors": [{
            "content": chunk.text,
            "metadata": {
                "source": document.name,
                "page": chunk.page,
                "section": chunk.section
            }
        }]
    })

# Query for relevant information
context = client.call_tool("search_vectors", {
    "collection": "knowledge_base",
    "query": user_question,
    "limit": 5
})
```

#### 3. Agent Tool Memory
Store tool usage patterns and outcomes:

```python
# Record tool usage
client.call_tool("add_vectors", {
    "collection": "tool_memory",
    "vectors": [{
        "content": f"Used {tool_name} with params {params} - Result: {result_summary}",
        "metadata": {
            "tool": tool_name,
            "success": success,
            "execution_time": exec_time,
            "timestamp": timestamp
        }
    }]
})

# Learn from past tool usage
past_usage = client.call_tool("search_vectors", {
    "collection": "tool_memory",
    "query": f"How to use {tool_name} effectively?",
    "limit": 10,
    "filters": {"success": True}
})
```

### MCP Server Options

```bash
./build/embeddix-mcp [options]
  -persistence string   Storage backend: memory, bolt, badger (default "memory")
  -data string         Data directory path (default "./data")
  -verbose            Enable verbose logging
```

For more MCP examples and detailed API documentation, see:
- [MCP API Documentation](docs/MCP_API.md)
- [MCP Examples](docs/MCP_EXAMPLES.md)
- [MCP Implementation Plan](docs/MCP_IMPLEMENTATION_PLAN.md)

## API Usage

### Traditional Vector Operations

#### Create a Collection

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "documents",
    "dimension": 384,
    "index_type": "hnsw",
    "distance": "cosine"
  }'
```

#### Add Vectors

```bash
# Single vector
curl -X POST http://localhost:8080/collections/documents/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "values": [0.1, 0.2, ...],
    "metadata": {
      "title": "Introduction to Vector Databases",
      "category": "technology"
    }
  }'

# Batch insert
curl -X POST http://localhost:8080/collections/documents/vectors/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"id": "doc1", "values": [...], "metadata": {...}},
    {"id": "doc2", "values": [...], "metadata": {...}}
  ]'
```

#### Search for Similar Vectors

```bash
# K-nearest neighbor search
curl -X POST http://localhost:8080/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.15, 0.25, ...],
    "top_k": 5,
    "filter": {"category": "technology"},
    "include_vectors": false
  }'

# Range search (find all vectors within radius)
curl -X POST http://localhost:8080/collections/documents/search/range \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.15, 0.25, ...],
    "radius": 0.5,
    "filter": {"category": "technology"},
    "limit": 100
  }'
```

### 🤖 AI-Enhanced Operations

#### Auto-Embedding from Text

```bash
# Create collection with auto-embedding enabled
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smart_docs",
    "auto_embed": true,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
  }'

# Add text content - vectors generated automatically
curl -X POST http://localhost:8080/collections/smart_docs/documents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "article1",
    "content": "Machine learning is transforming how we build software...",
    "metadata": {"author": "Jane Doe", "category": "tech"}
  }'

# Search using natural language
curl -X POST http://localhost:8080/collections/smart_docs/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does AI impact software development?",
    "top_k": 5,
    "analyze_content": true
  }'
```

#### Content Analysis Pipeline

```bash
# Analyze text content for insights
curl -X POST http://localhost:8080/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I love this new AI technology! It makes development so much faster and more efficient.",
    "include_sentiment": true,
    "include_entities": true,
    "include_topics": true,
    "include_language": true
  }'

# Response includes:
# {
#   "language": {"code": "en", "confidence": 0.99},
#   "sentiment": {"polarity": 0.8, "label": "positive"},
#   "entities": [{"text": "AI", "label": "TECHNOLOGY"}],
#   "topics": [{"label": "Technology", "confidence": 0.95}],
#   "key_phrases": ["AI technology", "development", "efficient"]
# }
```

#### Model Management

```bash
# List available models
curl http://localhost:8080/ai/models

# Load a specific model
curl -X POST http://localhost:8080/ai/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "name": "bert-base-uncased",
    "path": "/models/bert-base-uncased.onnx",
    "config": {
      "batch_size": 16,
      "max_tokens": 512,
      "pooling_strategy": "cls"
    }
  }'

# Get model health status
curl http://localhost:8080/ai/models/bert-base-uncased/health
```

### API Documentation

When the server is running, you can access interactive API documentation at:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Spec**: http://localhost:8080/swagger.yaml

## Architecture

### Core Components

- **Vector Store**: Main interface for all vector operations
- **Index**: Pluggable index implementations (Flat, HNSW)
- **Persistence**: Pluggable storage backends (Memory, BoltDB, BadgerDB)
- **WAL**: Write-ahead logging for durability
- **API Server**: RESTful HTTP server with JSON API and OpenAPI/Swagger documentation

### 🤖 AI Components

- **ONNX Runtime Engine**: Production-ready embedding inference with support for transformer models
- **Model Manager**: Lifecycle management for embedding models with health monitoring
- **Content Analyzer**: Advanced text processing pipeline including:
  - **Language Detector**: Multi-language text identification
  - **Sentiment Analyzer**: Emotion and opinion analysis
  - **Entity Extractor**: Named entity recognition (NER)
  - **Topic Modeler**: Automatic topic classification
  - **Key Phrase Extractor**: Important phrase identification
- **Auto-Embedding API**: Seamless text-to-vector conversion with model selection

### Storage Options

1. **Memory**: Fast, no persistence (good for testing)
2. **BoltDB**: Embedded key-value store, good for single-node deployments
3. **BadgerDB**: High-performance key-value store with advanced features

### Index Types

1. **Flat Index**: Brute-force search, 100% recall, suitable for small datasets
2. **HNSW Index**: Approximate nearest neighbor search, fast for large datasets
3. **Quantized HNSW Index**: Memory-efficient HNSW with Product/Scalar Quantization, 256x memory reduction

## Performance

### Benchmarking

Run comprehensive benchmarks to test performance on your hardware:

```bash
# Basic benchmark
make benchmark

# Detailed benchmark with comparisons
./build/embeddix-benchmark -vectors 10000 -queries 1000 -compare

# Memory-focused benchmark  
./build/embeddix-benchmark -vectors 100000 -memory-profile

# Docker benchmark
docker-compose --profile benchmark run benchmark
```

### Performance Results

#### M1 MacBook Pro (16GB RAM)
| Operation | Performance | Notes |
|-----------|-------------|-------|
| Individual Insert | ~4,000 ops/sec | Single vector operations |
| Batch Insert (100) | ~45,000 vectors/sec | Optimized batch processing |
| Search (Flat, 10K) | ~77,000 queries/sec | 100% recall |
| Search (HNSW, 1M) | ~65,000 queries/sec | >95% recall@10 |
| Search (Quantized HNSW) | ~50,000 queries/sec | 256x memory reduction |
| Concurrent Search (8 threads) | ~87,000 queries/sec | Near-linear scaling |

#### Memory Efficiency
| Index Type | Memory Usage (1M vectors, 384d) | Compression |
|------------|----------------------------------|-------------|
| Flat | 1.5 GB | None |
| HNSW | 2.1 GB | None |
| Quantized HNSW (PQ) | 8.2 MB | 256x |
| Quantized HNSW (SQ) | 384 MB | 4x |


## Development

### Prerequisites

- Go 1.21 or higher
- Docker and Docker Compose (optional)
- Make (optional)

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific package tests
go test ./api -v
```

### Development with Hot Reload

```bash
# Install air for hot reload
make install-tools

# Run with hot reload
make run-dev

# Or use Docker
docker-compose -f docker-compose.dev.yml up
```

## Data Management

### Backup and Restore

```bash
# Export data
curl -X POST http://localhost:8080/admin/export \
  -d '{"output_dir": "/backups/export1"}'

# Import data
curl -X POST http://localhost:8080/admin/import \
  -d '{"input_dir": "/backups/export1"}'
```

### Migration Support

EmbeddixDB includes a migration framework for schema evolution:

```go
migrator := migration.NewMigrator(persistence)
migrator.AddMigration(&migration.Migration{
    Version: 1,
    Name: "add_categories",
    UpFunc: func(ctx context.Context, db core.Persistence) error {
        // Migration logic
    },
})
migrator.MigrateUp(ctx)
```

## Production Deployment

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  embeddixdb:
    image: embeddixdb:latest
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
      - ./config.yml:/etc/embeddixdb/config.yml
    environment:
      - EMBEDDIXDB_CONFIG=/etc/embeddixdb/config.yml
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embeddixdb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: embeddixdb
  template:
    metadata:
      labels:
        app: embeddixdb
    spec:
      containers:
      - name: embeddixdb
        image: embeddixdb:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /etc/embeddixdb
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: embeddixdb-pvc
      - name: config
        configMap:
          name: embeddixdb-config
```

### Monitoring

Coming soon: Prometheus metrics and Grafana dashboards for monitoring:
- Request latency histograms
- Operation counters  
- Resource usage metrics
- Index performance statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Roadmap

### ✅ **Recently Completed** 
- [x] **Configuration System**: YAML-based configuration with environment overrides
- [x] **Ollama Integration**: Local LLM embedding support
- [x] **ONNX Runtime Integration**: Production-ready embedding inference
- [x] **Auto-Embedding API**: Automatic vector generation from text
- [x] **Content Analysis Pipeline**: Sentiment, entities, topics, language detection
- [x] **Model Management**: BERT, RoBERTa, Sentence Transformers support
- [x] **Advanced Text Processing**: Key phrase extraction and text segmentation
- [x] **BM25 Text Search Engine**: Traditional full-text search capabilities
- [x] **Hybrid Search Fusion**: Combine vector and text search with intelligent ranking
- [x] **Feedback & Learning System**: User feedback tracking and personalized search
- [x] **Memory-Optimized Indexing**: Product Quantization with 256x compression
- [x] **Multi-Level Caching**: Intelligent caching layer for improved performance
- [x] **Model Context Protocol (MCP)**: LLM integration via standardized protocol

### 🚧 **In Progress**
- [ ] **Hierarchical Indexing**: Two-level HNSW for massive scale
- [ ] **GPU Acceleration**: CUDA/OpenCL integration for similarity computations

### 🔮 **Future Plans**

#### Q1 2025
- [ ] **Client SDKs**: Official Python and JavaScript libraries
- [ ] **Prometheus Metrics**: Comprehensive monitoring integration
- [ ] **Multi-Modal Embeddings**: Image and audio support via CLIP models

#### Q2 2025  
- [ ] **Distributed Mode**: Multi-node deployment with automatic sharding
- [ ] **Advanced Query Processing**: Query expansion and re-ranking
- [ ] **Real-time Analytics**: Search analytics dashboard

#### Long Term
- [ ] **Additional Index Types**: IVF, LSH, and custom algorithms
- [ ] **GraphQL API**: Alternative API interface
- [ ] **Streaming Updates**: Real-time vector updates via WebSockets
- [ ] **Vector Compression**: Advanced quantization techniques

## Troubleshooting

### Common Issues

1. **Ollama connection refused**
   ```
   Error: failed to create embedding engine: connection refused
   ```
   Solution: Ensure Ollama is running (`ollama serve`) and the endpoint is correct.

2. **Out of memory with large datasets**
   ```
   Error: cannot allocate memory
   ```
   Solution: Use quantized indexes or increase system memory. Configure:
   ```yaml
   vector_store:
     default_index_type: "quantized_hnsw"
   ```

3. **Slow embedding generation**
   
   Solution: Increase batch size or use a smaller model:
   ```yaml
   ai:
     embedding:
       batch_size: 64
       model: "all-minilm"  # Faster model
   ```

### Getting Help

- 📖 [Documentation](docs/)
- 💬 [GitHub Discussions](https://github.com/dshills/EmbeddixDB/discussions)
- 🐛 [Issue Tracker](https://github.com/dshills/EmbeddixDB/issues)
- 📧 Email: support@embeddixdb.com (coming soon)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/EmbeddixDB.git
cd EmbeddixDB

# Install development dependencies
make install-tools

# Run tests
make test

# Run linters
make lint
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Go community for excellent libraries like BoltDB and BadgerDB
- ONNX Runtime team for making ML inference accessible
- Ollama team for democratizing LLM access
- Contributors and early adopters who provided valuable feedback

---

<p align="center">
  Made with ❤️ by the EmbeddixDB team
</p>
