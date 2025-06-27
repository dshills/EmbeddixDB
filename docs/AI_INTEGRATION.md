# AI Integration Guide

EmbeddixDB now includes comprehensive AI capabilities for intelligent document processing, auto-embedding, and advanced content analysis. This guide covers all AI features and their usage.

## Table of Contents

1. [Quick Start](#quick-start)
2. [ONNX Runtime Integration](#onnx-runtime-integration)
3. [Auto-Embedding API](#auto-embedding-api)
4. [Content Analysis Pipeline](#content-analysis-pipeline)
5. [Model Management](#model-management)
6. [Configuration](#configuration)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Enable AI Features

```bash
# Start EmbeddixDB with AI features enabled
./build/embeddix-api \
  -host 0.0.0.0 \
  -port 8080 \
  -db bolt \
  -path data/embeddix.db \
  -enable-ai \
  -models-path /path/to/models
```

### Create AI-Enhanced Collection

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "intelligent_docs",
    "auto_embed": true,
    "model_name": "all-MiniLM-L6-v2",
    "analyze_content": true
  }'
```

### Add Content with Auto-Analysis

```bash
curl -X POST http://localhost:8080/collections/intelligent_docs/documents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "content": "Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis and personalized treatment plans.",
    "metadata": {"source": "research_paper", "author": "Dr. Smith"}
  }'
```

## ONNX Runtime Integration

### Supported Model Architectures

EmbeddixDB automatically detects and optimizes for popular model architectures:

| Architecture | Default Config | Pooling Strategy | Max Tokens |
|-------------|----------------|------------------|------------|
| **BERT** | CLS pooling | `cls` | 512 |
| **DistilBERT** | CLS pooling | `cls` | 512 |
| **RoBERTa** | CLS pooling | `cls` | 512 |
| **Sentence-T5** | Mean pooling | `mean` | 512 |
| **All-MiniLM** | Mean pooling | `mean` | 256 |
| **E5** | Mean pooling | `mean` | 512 |
| **BGE** | CLS pooling | `cls` | 512 |

### Model Loading and Validation

```bash
# Load a model with custom configuration
curl -X POST http://localhost:8080/ai/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom-bert",
    "path": "/models/bert-base-uncased.onnx",
    "config": {
      "batch_size": 32,
      "max_tokens": 512,
      "pooling_strategy": "cls",
      "normalize_embeddings": true,
      "enable_gpu": false
    }
  }'
```

### Pooling Strategies

- **`cls`**: Use the CLS token embedding (first token) - ideal for BERT-style models
- **`mean`**: Average all token embeddings - good for sentence transformers
- **`max`**: Take maximum values across all tokens - useful for specialized tasks

### Memory Optimization

The system automatically recommends optimal batch sizes based on available memory:

```bash
# Get memory recommendations
curl http://localhost:8080/ai/models/bert-base/recommendations
```

## Auto-Embedding API

### Collection Configuration

```bash
# Create collection with auto-embedding
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smart_docs",
    "auto_embed": true,
    "model_name": "all-MiniLM-L6-v2",
    "chunk_size": 512,
    "overlap": 50,
    "analyze_content": true
  }'
```

### Document Processing

When you add text content, the system automatically:

1. **Segments text** into optimal chunks
2. **Generates embeddings** using the specified model
3. **Analyzes content** for sentiment, entities, and topics
4. **Stores vectors** with enriched metadata

```bash
# Add document - vectors generated automatically
curl -X POST http://localhost:8080/collections/smart_docs/documents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "article1",
    "content": "Long article content here...",
    "metadata": {
      "title": "AI in Healthcare",
      "category": "research"
    }
  }'
```

### Batch Processing

```bash
# Process multiple documents efficiently
curl -X POST http://localhost:8080/collections/smart_docs/documents/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "doc1",
      "content": "First document content...",
      "metadata": {"category": "tech"}
    },
    {
      "id": "doc2", 
      "content": "Second document content...",
      "metadata": {"category": "science"}
    }
  ]'
```

## Content Analysis Pipeline

### Available Analyzers

#### 1. Language Detection

Identifies the language of text content with confidence scores:

```bash
curl -X POST http://localhost:8080/ai/analyze/language \
  -H "Content-Type: application/json" \
  -d '{"content": "Bonjour, comment allez-vous?"}'

# Response:
# {
#   "language": {
#     "code": "fr",
#     "name": "French", 
#     "confidence": 0.98
#   }
# }
```

#### 2. Sentiment Analysis

Analyzes emotional tone and opinion polarity:

```bash
curl -X POST http://localhost:8080/ai/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"content": "I absolutely love this new technology!"}'

# Response:
# {
#   "sentiment": {
#     "polarity": 0.85,
#     "confidence": 0.92,
#     "label": "positive"
#   }
# }
```

#### 3. Entity Extraction

Identifies and classifies named entities:

```bash
curl -X POST http://localhost:8080/ai/analyze/entities \
  -H "Content-Type: application/json" \
  -d '{"content": "Apple Inc. was founded by Steve Jobs in California."}'

# Response:
# {
#   "entities": [
#     {
#       "text": "Apple Inc.",
#       "label": "ORGANIZATION",
#       "confidence": 0.95,
#       "start_pos": 0,
#       "end_pos": 10
#     },
#     {
#       "text": "Steve Jobs",
#       "label": "PERSON", 
#       "confidence": 0.98,
#       "start_pos": 25,
#       "end_pos": 35
#     }
#   ]
# }
```

#### 4. Topic Modeling

Automatically classifies content into topics:

```bash
curl -X POST http://localhost:8080/ai/analyze/topics \
  -H "Content-Type: application/json" \
  -d '{"content": "Machine learning algorithms are transforming healthcare diagnostics."}'

# Response:
# {
#   "topics": [
#     {
#       "id": "tech-001",
#       "label": "Technology",
#       "keywords": ["machine learning", "algorithms"],
#       "confidence": 0.89,
#       "weight": 0.75
#     },
#     {
#       "id": "health-001", 
#       "label": "Health",
#       "keywords": ["healthcare", "diagnostics"],
#       "confidence": 0.82,
#       "weight": 0.65
#     }
#   ]
# }
```

#### 5. Key Phrase Extraction

Identifies the most important phrases using TF-IDF scoring:

```bash
curl -X POST http://localhost:8080/ai/analyze/keyphrases \
  -H "Content-Type: application/json" \
  -d '{"content": "Artificial intelligence and machine learning are revolutionizing data science."}'

# Response:
# {
#   "key_phrases": [
#     "artificial intelligence",
#     "machine learning", 
#     "data science",
#     "revolutionizing"
#   ]
# }
```

### Comprehensive Analysis

Analyze all aspects of content in a single request:

```bash
curl -X POST http://localhost:8080/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I love how AI is transforming healthcare! Companies like Google and Microsoft are leading the innovation.",
    "include_sentiment": true,
    "include_entities": true,
    "include_topics": true,
    "include_language": true,
    "include_keyphrases": true
  }'
```

## Model Management

### Lifecycle Management

```bash
# List all available models
curl http://localhost:8080/ai/models

# Get detailed model information
curl http://localhost:8080/ai/models/bert-base-uncased/info

# Check model health
curl http://localhost:8080/ai/models/bert-base-uncased/health

# Unload a model to free memory
curl -X POST http://localhost:8080/ai/models/bert-base-uncased/unload

# Reload a model
curl -X POST http://localhost:8080/ai/models/bert-base-uncased/reload
```

### Model Health Monitoring

The system tracks comprehensive health metrics:

```json
{
  "model_name": "bert-base-uncased",
  "status": "healthy",
  "loaded_at": "2023-12-01T10:00:00Z",
  "last_check": "2023-12-01T10:05:00Z",
  "latency": "45ms",
  "error_rate": 0.001,
  "memory_usage_mb": 512,
  "cpu_usage": 15.5,
  "gpu_usage": 0.0,
  "total_inferences": 15420,
  "average_throughput": 450.2
}
```

### Performance Optimization

```bash
# Get performance recommendations
curl http://localhost:8080/ai/models/bert-base/optimize

# Update model configuration
curl -X PUT http://localhost:8080/ai/models/bert-base/config \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 64,
    "num_threads": 4,
    "optimization_level": "all"
  }'
```

## Configuration

### Environment Variables

```bash
# Enable AI features
export EMBEDDIX_AI_ENABLED=true

# Set models directory
export EMBEDDIX_MODELS_PATH=/path/to/models

# Configure default model
export EMBEDDIX_DEFAULT_MODEL=all-MiniLM-L6-v2

# Set memory limits
export EMBEDDIX_AI_MEMORY_LIMIT=2048  # MB

# Enable GPU acceleration (if available)
export EMBEDDIX_GPU_ENABLED=true
```

### Configuration File

Create `config/ai.yaml`:

```yaml
ai:
  enabled: true
  models_path: "/app/models"
  default_model: "all-MiniLM-L6-v2"
  memory_limit_mb: 2048
  
  # Model configurations
  models:
    all-MiniLM-L6-v2:
      path: "/app/models/all-MiniLM-L6-v2.onnx"
      batch_size: 32
      max_tokens: 256
      pooling_strategy: "mean"
      normalize_embeddings: true
      
    bert-base-uncased:
      path: "/app/models/bert-base-uncased.onnx"
      batch_size: 16
      max_tokens: 512
      pooling_strategy: "cls"
      
  # Content analysis settings
  analysis:
    enable_sentiment: true
    enable_entities: true
    enable_topics: true
    enable_language: true
    enable_keyphrases: true
    
    # Language detection
    min_confidence: 0.8
    supported_languages: ["en", "es", "fr", "de", "zh"]
    
    # Sentiment analysis
    sentiment_threshold: 0.1
    
    # Entity extraction
    entity_types: ["PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY"]
    
    # Topic modeling
    max_topics: 5
    topic_threshold: 0.3
```

## Production Deployment

### Docker Configuration

```dockerfile
FROM golang:1.21-alpine AS builder

# Install ONNX Runtime dependencies
RUN apk add --no-cache \
    build-base \
    cmake \
    wget

# Download ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.0.tgz \
    && cp -r onnxruntime-linux-x64-1.16.0/* /usr/local/

# Build EmbeddixDB
WORKDIR /app
COPY . .
RUN go build -o embeddix-api ./cmd/embeddix-api

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

# Copy ONNX Runtime libraries
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /app/embeddix-api .

# Create directories
RUN mkdir -p /app/data /app/models /app/config

# Copy configuration
COPY config/ /app/config/

EXPOSE 8080
CMD ["./embeddix-api", "-config", "/app/config/ai.yaml"]
```

### Docker Compose with AI

```yaml
version: '3.8'

services:
  embeddixdb:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models  # Mount your ONNX models
      - ./config:/app/config
    environment:
      - EMBEDDIX_AI_ENABLED=true
      - EMBEDDIX_MODELS_PATH=/app/models
      - EMBEDDIX_DEFAULT_MODEL=all-MiniLM-L6-v2
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Model downloading service
  model-downloader:
    image: alpine/curl
    volumes:
      - ./models:/models
    command: |
      sh -c "
        # Download popular models (examples)
        curl -L -o /models/all-MiniLM-L6-v2.onnx 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx'
        curl -L -o /models/bert-base-uncased.onnx 'https://huggingface.co/bert-base-uncased/resolve/main/model.onnx'
      "
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embeddixdb-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embeddixdb-ai
  template:
    metadata:
      labels:
        app: embeddixdb-ai
    spec:
      containers:
      - name: embeddixdb
        image: embeddixdb:latest-ai
        ports:
        - containerPort: 8080
        env:
        - name: EMBEDDIX_AI_ENABLED
          value: "true"
        - name: EMBEDDIX_MODELS_PATH
          value: "/app/models"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
```

## Troubleshooting

### Common Issues

#### 1. ONNX Runtime Not Found

```bash
# Error: "onnxruntime.so not found"
# Solution: Install ONNX Runtime libraries

# Ubuntu/Debian:
sudo apt-get install libonnxruntime1.16.0

# CentOS/RHEL:
sudo yum install onnxruntime

# macOS:
brew install onnxruntime

# Or set library path:
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

#### 2. Model Loading Failures

```bash
# Check model file integrity
curl http://localhost:8080/ai/models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/bert-base.onnx"}'

# Verify model compatibility
curl http://localhost:8080/ai/models/inspect \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/bert-base.onnx"}'
```

#### 3. Memory Issues

```bash
# Monitor memory usage
curl http://localhost:8080/ai/system/memory

# Get optimization recommendations
curl http://localhost:8080/ai/system/optimize

# Reduce batch size in model config
curl -X PUT http://localhost:8080/ai/models/bert-base/config \
  -d '{"batch_size": 8, "max_tokens": 256}'
```

#### 4. Performance Issues

```bash
# Check model performance
curl http://localhost:8080/ai/models/bert-base/stats

# Enable performance profiling
curl -X POST http://localhost:8080/ai/system/profiling/enable

# Get performance recommendations
curl http://localhost:8080/ai/system/performance/analyze
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Start with debug logging
./build/embeddix-api -log-level debug -enable-ai-debug

# Check logs
curl http://localhost:8080/ai/system/logs
```

### Support

For additional support:

1. **Check logs**: `/app/logs/embeddixdb.log`
2. **Monitor health**: `GET /ai/health`
3. **System diagnostics**: `GET /ai/system/diagnostics`
4. **Report issues**: [GitHub Issues](https://github.com/dshills/EmbeddixDB/issues)

## Performance Benchmarks

Typical performance on various hardware configurations:

| Hardware | Model | Batch Size | Throughput | Latency |
|----------|-------|------------|------------|---------|
| CPU (8 cores) | all-MiniLM-L6-v2 | 32 | 450 docs/sec | 45ms |
| CPU (8 cores) | bert-base-uncased | 16 | 280 docs/sec | 85ms |
| GPU (RTX 3080) | all-MiniLM-L6-v2 | 64 | 1200 docs/sec | 25ms |
| GPU (RTX 3080) | bert-base-uncased | 32 | 850 docs/sec | 45ms |

These benchmarks include full content analysis pipeline (embedding + sentiment + entities + topics).