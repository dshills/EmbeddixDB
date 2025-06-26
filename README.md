# EmbeddixDB

This is pre-alpha software and should be treated as such.

A high-performance vector database with integrated AI capabilities, designed for LLM memory storage and intelligent document processing. Written in Go, EmbeddixDB provides efficient similarity search, persistent storage, auto-embedding, and advanced content analysis.

## Features

### 🚀 **Core Vector Database**
- **High-Performance Vector Search**: Supports both flat (brute-force) and HNSW (Hierarchical Navigable Small World) indexes
- **Multiple Distance Metrics**: Cosine similarity, Euclidean (L2), and Dot product
- **Persistent Storage**: Choose from in-memory, BoltDB, or BadgerDB backends
- **Write-Ahead Logging (WAL)**: Ensures data durability and crash recovery
- **RESTful API**: Easy integration with any application
- **Batch Operations**: Efficient bulk vector insertions

### 🤖 **AI Integration (NEW)**
- **ONNX Runtime Integration**: Production-ready embedding inference with real transformer models
- **Auto-Embedding API**: Automatic vector generation from text content
- **Content Analysis Pipeline**: Advanced text understanding with sentiment analysis, entity extraction, and topic modeling
- **Model Management**: Support for popular architectures (BERT, RoBERTa, Sentence Transformers, etc.)
- **Intelligent Preprocessing**: Language detection, text segmentation, and key phrase extraction

### 🔧 **Operations & Monitoring**
- **Data Migration**: Built-in tools for backup, restore, and schema evolution
- **Docker Support**: Production-ready containers with compose configurations
- **Performance Monitoring**: Comprehensive benchmarking suite

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

# Run the server
./embeddixdb -host 0.0.0.0 -port 8080 -db bolt -path data/embeddix.db
```

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

### 🤖 AI-Enhanced Operations (NEW)

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

### 🤖 AI Components (NEW)

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

## Performance

Run benchmarks to test performance on your hardware:

```bash
# Basic benchmark
make benchmark

# Detailed benchmark with comparisons
./embeddixdb-benchmark -vectors 10000 -queries 1000 -compare

# Docker benchmark
docker-compose --profile benchmark run benchmark
```

Example results on M1 MacBook Pro:
- Individual Insert: ~4,000 ops/sec
- Batch Insert (100): ~45,000 vectors/sec
- Search (Flat, 10K vectors): ~77,000 queries/sec
- Concurrent Search: ~87,000 queries/sec

## Configuration

### Command Line Flags

```bash
./embeddixdb \
  -host 0.0.0.0 \           # Host to bind to
  -port 8080 \              # Port to listen on
  -db bolt \                # Database type: memory, bolt, badger
  -path data/embeddix.db \  # Database file path
  -wal \                    # Enable write-ahead logging
  -wal-path data/wal        # WAL directory path
```

### Environment Variables

- `EMBEDDIX_LOG_LEVEL`: Set logging level (debug, info, warn, error)

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

## Monitoring

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
- [x] **ONNX Runtime Integration**: Production-ready embedding inference
- [x] **Auto-Embedding API**: Automatic vector generation from text
- [x] **Content Analysis Pipeline**: Sentiment, entities, topics, language detection
- [x] **Model Management**: BERT, RoBERTa, Sentence Transformers support
- [x] **Advanced Text Processing**: Key phrase extraction and text segmentation

### 🚧 **In Progress**
- [ ] **BM25 Text Search Engine**: Traditional full-text search capabilities
- [ ] **Hybrid Search Fusion**: Combine vector and text search with intelligent ranking

### 🔮 **Future Plans**
- [ ] **Multi-Modal Support**: Image and audio embedding support
- [ ] **Distributed Clustering**: Multi-node deployment with sharding
- [ ] **GPU Acceleration**: CUDA support for faster similarity search
- [ ] **Advanced Retrieval**: Re-ranking and query expansion
- [ ] **Real-time Analytics**: Search analytics and user behavior insights
- [ ] **Client SDKs**: Python, JavaScript, and Rust client libraries
- [ ] **Additional Index Types**: IVF, LSH, and custom algorithms
- [ ] **Prometheus Integration**: Comprehensive metrics and monitoring
- [ ] **GraphQL API**: Alternative API interface

## Acknowledgments

This project was inspired by the need for a simple, fast, and reliable vector database for LLM applications. Special thanks to the Go community for excellent libraries like BoltDB and BadgerDB.
