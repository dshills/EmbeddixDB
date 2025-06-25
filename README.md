# EmbeddixDB

A high-performance vector database designed for LLM memory storage, written in Go. EmbeddixDB provides efficient similarity search, persistent storage, and a RESTful API for easy integration.

## Features

- **High-Performance Vector Search**: Supports both flat (brute-force) and HNSW (Hierarchical Navigable Small World) indexes
- **Multiple Distance Metrics**: Cosine similarity, Euclidean (L2), and Dot product
- **Persistent Storage**: Choose from in-memory, BoltDB, or BadgerDB backends
- **Write-Ahead Logging (WAL)**: Ensures data durability and crash recovery
- **RESTful API**: Easy integration with any application
- **Batch Operations**: Efficient bulk vector insertions
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

### Create a Collection

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

### Add Vectors

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

### Search for Similar Vectors

```bash
curl -X POST http://localhost:8080/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.15, 0.25, ...],
    "top_k": 5,
    "filter": {"category": "technology"},
    "include_vectors": false
  }'
```

## Architecture

### Core Components

- **Vector Store**: Main interface for all vector operations
- **Index**: Pluggable index implementations (Flat, HNSW)
- **Persistence**: Pluggable storage backends (Memory, BoltDB, BadgerDB)
- **WAL**: Write-ahead logging for durability
- **API Server**: RESTful HTTP server with JSON API

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

- [ ] Prometheus metrics integration
- [ ] Range queries and hybrid search
- [ ] Distributed clustering support
- [ ] GPU acceleration for similarity search
- [ ] Additional index types (IVF, LSH)
- [ ] GraphQL API
- [ ] Python and JavaScript client SDKs

## Acknowledgments

This project was inspired by the need for a simple, fast, and reliable vector database for LLM applications. Special thanks to the Go community for excellent libraries like BoltDB and BadgerDB.