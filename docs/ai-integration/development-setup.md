# Development Setup Guide: AI Integration

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB+ (16GB recommended for model development)
- **Storage**: 50GB+ free space (for models and datasets)
- **Go**: 1.21+ 
- **Python**: 3.8+ (for HuggingFace integration)

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ 
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for model training/optimization)
- **Storage**: 200GB+ SSD
- **Docker**: Latest version

### Software Dependencies

#### Go Dependencies
```bash
# Core AI/ML libraries
go get github.com/yalue/onnxruntime_go
go get github.com/nlpodyssey/spago
go get github.com/pdevine/tensor
go get github.com/chewxy/math32

# Data processing
go get github.com/blevesearch/bleve/v2
go get github.com/kljensen/snowball
go get github.com/bbalet/stopwords

# Utilities
go get github.com/spf13/viper
go get github.com/prometheus/client_golang
go get go.uber.org/zap
```

#### Python Dependencies
```bash
# Create virtual environment
python -m venv embeddix-ai
source embeddix-ai/bin/activate  # On Windows: embeddix-ai\Scripts\activate

# Install core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install onnx onnxruntime
pip install numpy scipy scikit-learn
pip install datasets huggingface-hub

# Text processing
pip install nltk spacy
pip install langdetect textstat

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

## Environment Setup

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/dshills/EmbeddixDB.git
cd EmbeddixDB

# Install Go dependencies
go mod download

# Setup Python environment
python -m venv ai-dev
source ai-dev/bin/activate
pip install -r requirements-ai.txt
```

### 2. ONNX Runtime Setup

#### Linux/macOS
```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz

# Set environment variables
export ORT_LIB_PATH=/path/to/onnxruntime-linux-x64-1.16.0/lib
export LD_LIBRARY_PATH=$ORT_LIB_PATH:$LD_LIBRARY_PATH
```

#### Windows
```powershell
# Download and extract ONNX Runtime
# Set environment variable
$env:ORT_LIB_PATH = "C:\path\to\onnxruntime-win-x64-1.16.0\lib"
$env:PATH = "$env:ORT_LIB_PATH;$env:PATH"
```

### 3. GPU Support (Optional)

#### NVIDIA CUDA Setup
```bash
# Install CUDA Toolkit (version 11.8 recommended)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN
# Download from NVIDIA (requires registration)
# Extract and copy files to CUDA directory

# Verify installation
nvidia-smi
nvcc --version
```

#### Docker with GPU Support
```dockerfile
# Dockerfile.ai-dev
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Go
RUN wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install ONNX Runtime GPU
RUN pip install onnxruntime-gpu

WORKDIR /app
COPY . .
RUN go mod download
```

## Development Configuration

### 1. Configuration Files

#### AI Configuration (`config/ai-config.yaml`)
```yaml
ai:
  models:
    default_text: "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: "./models"
    download_timeout: "10m"
    
  inference:
    batch_size: 32
    max_concurrency: 4
    enable_gpu: true
    optimization_level: 2
    
  text_processing:
    max_tokens: 512
    chunk_size: 512
    chunk_overlap: 50
    normalize_embeddings: true
    
  hybrid_search:
    default_fusion: "rrf"
    default_weights:
      vector: 0.7
      text: 0.3
      freshness: 0.1
      
  performance:
    enable_caching: true
    cache_ttl: "1h"
    max_cache_size: "1GB"
    metrics_interval: "30s"

development:
  log_level: "debug"
  enable_profiling: true
  mock_models: false
  test_data_path: "./testdata"
```

#### Model Registry (`config/models.yaml`)
```yaml
models:
  - name: "sentence-transformers/all-MiniLM-L6-v2"
    type: "onnx"
    url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx"
    checksum: "sha256:abcd1234..."
    dimension: 384
    max_tokens: 512
    size_mb: 22
    
  - name: "sentence-transformers/all-mpnet-base-v2"
    type: "huggingface"
    dimension: 768
    max_tokens: 514
    size_mb: 420
    
  - name: "openai/clip-vit-base-patch32"
    type: "onnx"
    url: "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/model.onnx"
    dimension: 512
    modalities: ["text", "image"]
    size_mb: 151
```

### 2. Environment Variables

```bash
# Development environment
export EMBEDDIX_ENV=development
export EMBEDDIX_LOG_LEVEL=debug

# AI Configuration
export EMBEDDIX_MODELS_DIR=./models
export EMBEDDIX_ENABLE_GPU=true
export EMBEDDIX_BATCH_SIZE=32

# Python environment
export PYTHONPATH=/path/to/embeddix/python
export TRANSFORMERS_CACHE=./models/transformers

# ONNX Runtime
export ORT_LOG_LEVEL=3
export ORT_TENSORRT_FP16_ENABLE=1

# Optional: API keys for external services
export HUGGINGFACE_API_TOKEN=hf_xxxx
export OPENAI_API_KEY=sk-xxxx
```

## Development Workflow

### 1. Running Development Server

```bash
# Start development server with AI features
go run cmd/embeddix-api/main.go \
  --config=config/ai-config.yaml \
  --log-level=debug \
  --enable-ai=true \
  --models-dir=./models

# Or run the built binary
./build/embeddix-api \
  --config=config/ai-config.yaml \
  --log-level=debug \
  --enable-ai=true \
  --models-dir=./models

# Or using Docker
docker-compose -f docker-compose.ai-dev.yml up
```

### 2. Testing Setup

#### Unit Tests
```bash
# Run AI integration tests
go test ./core/ai/... -v

# Run with coverage
go test ./core/ai/... -cover -coverprofile=coverage.out
go tool cover -html=coverage.out

# Benchmark tests
go test ./core/ai/... -bench=. -benchmem
```

#### Integration Tests
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
go test ./tests/integration/ai/... -tags=integration

# Load test with sample data
go run tests/load/ai_load_test.go
```

### 3. Model Development Tools

#### Model Conversion Script
```python
# scripts/convert_models.py
import torch
import onnx
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

def convert_to_onnx(model_name, output_path):
    """Convert HuggingFace model to ONNX format"""
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 512))
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'embeddings': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"Model converted to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_models.py <model_name> <output_path>")
        sys.exit(1)
    
    convert_to_onnx(sys.argv[1], sys.argv[2])
```

#### Model Benchmarking
```go
// tools/model_benchmark.go
package main

import (
    "context"
    "fmt"
    "time"
    "github.com/dshills/EmbeddixDB/core/ai"
)

func benchmarkModel(modelName string, texts []string) {
    engine, err := ai.LoadEmbeddingEngine(modelName, ai.DefaultConfig())
    if err != nil {
        panic(err)
    }
    defer engine.Close()
    
    // Warmup
    engine.Embed(context.Background(), []string{"warmup"})
    
    // Benchmark
    start := time.Now()
    embeddings, err := engine.Embed(context.Background(), texts)
    duration := time.Since(start)
    
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Model: %s\n", modelName)
    fmt.Printf("Texts: %d\n", len(texts))
    fmt.Printf("Embeddings: %d x %d\n", len(embeddings), len(embeddings[0]))
    fmt.Printf("Duration: %v\n", duration)
    fmt.Printf("Texts/sec: %.2f\n", float64(len(texts))/duration.Seconds())
}
```

## Development Tools

### 1. VS Code Configuration

#### Settings (`.vscode/settings.json`)
```json
{
    "go.testFlags": ["-v", "-count=1"],
    "go.buildFlags": ["-tags=ai"],
    "go.lintTool": "golangci-lint",
    "python.defaultInterpreterPath": "./ai-dev/bin/python",
    "python.testing.pytestEnabled": true,
    "files.associations": {
        "*.onnx": "binary"
    }
}
```

#### Launch Configuration (`.vscode/launch.json`)
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug AI Server",
            "type": "go",
            "request": "launch",
            "mode": "auto",
            "program": "./cmd/server",
            "args": [
                "--config=config/ai-config.yaml",
                "--log-level=debug",
                "--enable-ai=true"
            ],
            "env": {
                "EMBEDDIX_ENV": "development",
                "EMBEDDIX_ENABLE_GPU": "false"
            }
        },
        {
            "name": "Test AI Integration",
            "type": "go",
            "request": "launch",
            "mode": "test",
            "program": "./core/ai",
            "args": ["-test.v"]
        }
    ]
}
```

### 2. Debugging Tools

#### Model Inspector
```go
// tools/model_inspector.go
package main

import (
    "fmt"
    "github.com/yalue/onnxruntime_go"
)

func inspectONNXModel(modelPath string) {
    session, err := onnxruntime.NewSession(modelPath, nil)
    if err != nil {
        panic(err)
    }
    defer session.Destroy()
    
    // Get input info
    inputCount := session.GetInputCount()
    fmt.Printf("Inputs: %d\n", inputCount)
    
    for i := 0; i < inputCount; i++ {
        name := session.GetInputName(i)
        info := session.GetInputTypeInfo(i)
        fmt.Printf("  Input %d: %s, Type: %v\n", i, name, info)
    }
    
    // Get output info
    outputCount := session.GetOutputCount()
    fmt.Printf("Outputs: %d\n", outputCount)
    
    for i := 0; i < outputCount; i++ {
        name := session.GetOutputName(i)
        info := session.GetOutputTypeInfo(i)
        fmt.Printf("  Output %d: %s, Type: %v\n", i, name, info)
    }
}
```

#### Performance Profiler
```go
// tools/profiler.go
package main

import (
    "context"
    "fmt"
    "runtime"
    "time"
    _ "net/http/pprof"
    "net/http"
)

func profileEmbedding() {
    // Start pprof server
    go func() {
        fmt.Println("Profiling server: http://localhost:6060/debug/pprof/")
        http.ListenAndServe("localhost:6060", nil)
    }()
    
    // Profile memory
    var m1, m2 runtime.MemStats
    runtime.GC()
    runtime.ReadMemStats(&m1)
    
    // Run embedding
    engine := loadTestEngine()
    texts := generateTestTexts(1000)
    
    start := time.Now()
    embeddings, _ := engine.Embed(context.Background(), texts)
    duration := time.Since(start)
    
    runtime.GC()
    runtime.ReadMemStats(&m2)
    
    fmt.Printf("Duration: %v\n", duration)
    fmt.Printf("Embeddings: %d\n", len(embeddings))
    fmt.Printf("Memory used: %d KB\n", (m2.TotalAlloc-m1.TotalAlloc)/1024)
    fmt.Printf("Memory peak: %d KB\n", (m2.Sys-m1.Sys)/1024)
}
```

### 3. Monitoring and Metrics

#### Development Metrics Dashboard
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana:/var/lib/grafana/dashboards
```

#### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'embeddixdb-ai'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

## Testing and Validation

### 1. Test Data Setup

```bash
# Download test datasets
mkdir -p testdata
cd testdata

# Small dataset for unit tests
wget https://example.com/test_embeddings_small.json

# Medium dataset for integration tests  
wget https://example.com/test_embeddings_medium.json

# Large dataset for performance tests
wget https://example.com/test_embeddings_large.json
```

### 2. Automated Testing

```bash
# Run full test suite
make test-ai

# Run specific test categories
make test-embedding
make test-hybrid-search
make test-model-management

# Performance regression tests
make benchmark-ai
```

### 3. Manual Testing

#### Test Embedding Generation
```bash
curl -X POST http://localhost:8080/v1/collections/test/embed \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "test1", "content": "This is a test document"}
    ],
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

#### Test Hybrid Search
```bash
curl -X POST http://localhost:8080/v1/collections/test/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "fusion": {"algorithm": "rrf", "weights": {"vector": 0.7, "text": 0.3}}
  }'
```

This development setup provides a comprehensive environment for building and testing AI integration features in EmbeddixDB. The configuration supports both CPU and GPU development, includes proper debugging tools, and establishes testing workflows for reliable development.