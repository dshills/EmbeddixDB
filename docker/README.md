# EmbeddixDB Docker Guide

This guide explains how to run EmbeddixDB using Docker and Docker Compose.

## Quick Start

### Running with Docker

Build and run the EmbeddixDB server:

```bash
# Build the Docker image
docker build -t embeddixdb:latest .

# Run with BoltDB persistence
docker run -d \
  --name embeddixdb \
  -p 8080:8080 \
  -v embeddix_data:/app/data \
  embeddixdb:latest

# Run with in-memory storage (no persistence)
docker run -d \
  --name embeddixdb \
  -p 8080:8080 \
  embeddixdb:latest \
  -db memory

# Run with BadgerDB
docker run -d \
  --name embeddixdb \
  -p 8080:8080 \
  -v embeddix_data:/app/data \
  embeddixdb:latest \
  -db badger -path /app/data/badger
```

### Running with Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f embeddixdb

# Stop the service
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Development Mode

For development with hot reload:

```bash
# Start development server with hot reload
docker-compose -f docker-compose.dev.yml up

# Run with test client
docker-compose -f docker-compose.dev.yml --profile test up
```

## Configuration Options

### Environment Variables

- `EMBEDDIX_LOG_LEVEL`: Set logging level (debug, info, warn, error)

### Command Line Arguments

You can override default arguments in docker-compose.yml:

```yaml
command: [
  "-host", "0.0.0.0",
  "-port", "8080",
  "-db", "bolt",
  "-path", "/app/data/embeddix.db",
  "-wal"  # Enable Write-Ahead Logging
]
```

## Monitoring

To enable Prometheus and Grafana monitoring:

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d

# Access services:
# - EmbeddixDB API: http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## Running Benchmarks

```bash
# Run benchmarks in Docker
docker-compose --profile benchmark run benchmark

# Or directly with Docker
docker run --rm embeddixdb:latest \
  go run ./cmd/benchmark \
  -vectors 10000 \
  -queries 1000 \
  -dim 384
```

## Data Persistence

Data is stored in Docker volumes:

```bash
# List volumes
docker volume ls | grep embeddix

# Backup data
docker run --rm \
  -v embeddix_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/embeddix_backup.tar.gz -C /data .

# Restore data
docker run --rm \
  -v embeddix_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/embeddix_backup.tar.gz -C /data
```

## Health Checks

The container includes health checks:

```bash
# Check health status
docker inspect embeddixdb --format='{{.State.Health.Status}}'

# Manual health check
curl http://localhost:8080/health
```

## Building Multi-Architecture Images

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t embeddixdb:latest \
  --push .
```

## Troubleshooting

### View logs
```bash
docker logs embeddixdb
docker-compose logs -f --tail=100 embeddixdb
```

### Enter container shell
```bash
docker exec -it embeddixdb /bin/sh
```

### Reset everything
```bash
docker-compose down -v
docker system prune -a
```

## Security Notes

- The container runs as non-root user (uid: 1000)
- Only port 8080 is exposed
- Data directory permissions are restricted
- Use volume mounts for persistent data
- Consider using Docker secrets for sensitive configuration