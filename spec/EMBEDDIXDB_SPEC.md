# EmbeddixDB: Specification for a Vector Database in Go

## Table of Contents
1. Overview  
2. Design Goals  
3. Core Concepts  
4. Data Model  
5. Indexing & Search  
6. **AI Integration** *(NEW)*
7. API Design  
8. Persistence Layer  
9. Vector Operations  
10. Configuration & Optimization  
11. Deployment Modes  
12. Testing & Benchmarking  
13. Future Enhancements

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

## 6. AI Integration

### Overview

EmbeddixDB now includes comprehensive AI capabilities that transform it from a traditional vector database into an intelligent document processing and retrieval system. The AI integration provides automatic embedding generation, advanced content analysis, and seamless text-to-vector workflows.

### Core AI Components

#### 6.1 ONNX Runtime Engine

**Purpose**: Production-ready embedding inference using transformer models

**Architecture**:
```go
type EmbeddingEngine interface {
    Embed(ctx context.Context, content []string) ([][]float32, error)
    EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error)
    GetModelInfo() ModelInfo
    Warm(ctx context.Context) error
    Close() error
}

type ONNXEmbeddingEngine struct {
    modelName  string
    modelPath  string
    config     ModelConfig
    session    ONNXSession
    tokenizer  *Tokenizer
    stats      *InferenceStats
}
```

**Features**:
- Support for popular transformer architectures (BERT, RoBERTa, Sentence Transformers)
- Automatic model architecture detection and optimization
- Multiple pooling strategies (CLS, mean, max)
- Attention mask support for improved embedding quality
- Graceful fallback to mock implementations for development

**Supported Models**:
- BERT family (bert-base-uncased, bert-large-uncased)
- RoBERTa (roberta-base, roberta-large)
- DistilBERT (distilbert-base-uncased)
- Sentence Transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
- E5 models (e5-base-v2, e5-large-v2)
- BGE models (bge-base-en-v1.5, bge-large-en-v1.5)

#### 6.2 Content Analysis Pipeline

**Purpose**: Advanced text understanding and metadata extraction

**Components**:

1. **Language Detector**
   ```go
   type LanguageDetector interface {
       DetectLanguage(ctx context.Context, content string) (LanguageInfo, error)
   }
   ```
   - Supports 12+ languages with confidence scoring
   - Unicode-aware text processing
   - Handles mixed-language content

2. **Sentiment Analyzer**
   ```go
   type SentimentAnalyzer interface {
       AnalyzeSentiment(ctx context.Context, content string) (SentimentScore, error)
   }
   ```
   - Lexicon-based approach with 50+ sentiment words
   - Negation detection and intensity modifiers
   - Returns polarity (-1.0 to 1.0) and confidence scores

3. **Entity Extractor**
   ```go
   type EntityExtractor interface {
       ExtractEntities(ctx context.Context, content string) ([]Entity, error)
   }
   ```
   - Named Entity Recognition (NER)
   - Supports PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, etc.
   - Pattern-based extraction with confidence scoring

4. **Topic Modeler**
   ```go
   type TopicModeler interface {
       ExtractTopics(ctx context.Context, content string) ([]Topic, error)
   }
   ```
   - 12 predefined topic categories (Technology, Business, Science, etc.)
   - Keyword-based classification with TF-IDF scoring
   - Confidence and weight metrics for each topic

5. **Key Phrase Extractor**
   ```go
   type KeyPhraseExtractor interface {
       ExtractKeyPhrases(ctx context.Context, content string) ([]string, error)
   }
   ```
   - N-gram based phrase extraction (1-4 words)
   - TF-IDF scoring with position and capitalization bonuses
   - Stop word filtering and overlap removal

#### 6.3 Model Management System

**Purpose**: Lifecycle management for embedding models

```go
type ModelManager interface {
    LoadModel(ctx context.Context, modelName string, config ModelConfig) error
    UnloadModel(modelName string) error
    GetEngine(modelName string) (EmbeddingEngine, error)
    ListModels() []ModelInfo
    GetModelHealth(modelName string) (ModelHealth, error)
}
```

**Features**:
- Dynamic model loading and unloading
- Health monitoring with performance metrics
- Memory usage optimization
- Architecture-specific configuration
- Batch size recommendations based on available memory

**Model Health Metrics**:
```go
type ModelHealth struct {
    ModelName    string        `json:"model_name"`
    Status       string        `json:"status"`
    LoadedAt     time.Time     `json:"loaded_at"`
    Latency      time.Duration `json:"latency"`
    ErrorRate    float64       `json:"error_rate"`
    MemoryUsage  int64         `json:"memory_usage_mb"`
    CPUUsage     float64       `json:"cpu_usage"`
}
```

#### 6.4 Auto-Embedding API

**Purpose**: Seamless text-to-vector conversion with content analysis

**Collection Configuration**:
```go
type Collection struct {
    Name         string `json:"name"`
    AutoEmbed    bool   `json:"auto_embed"`
    ModelName    string `json:"model_name"`
    ChunkSize    int    `json:"chunk_size"`
    ChunkOverlap int    `json:"chunk_overlap"`
    AnalyzeContent bool `json:"analyze_content"`
}
```

**Document Processing Flow**:
1. **Text Segmentation**: Split long documents into optimal chunks
2. **Content Analysis**: Extract language, sentiment, entities, topics, key phrases
3. **Embedding Generation**: Convert text to vectors using specified model
4. **Metadata Enrichment**: Augment with analysis results
5. **Vector Storage**: Store embeddings with enriched metadata

**API Interface**:
```go
type AutoEmbedAPI interface {
    AddDocument(ctx context.Context, collection string, doc Document) error
    AddDocuments(ctx context.Context, collection string, docs []Document) error
    SearchSemantic(ctx context.Context, req SemanticSearchRequest) (SemanticSearchResult, error)
}
```

### AI Configuration

#### Model Configuration
```go
type ModelConfig struct {
    Name                string        `json:"name"`
    Type                string        `json:"type"`
    Path                string        `json:"path"`
    BatchSize           int           `json:"batch_size"`
    MaxTokens           int           `json:"max_tokens"`
    PoolingStrategy     string        `json:"pooling_strategy"`
    NormalizeEmbeddings bool          `json:"normalize_embeddings"`
    EnableGPU           bool          `json:"enable_gpu"`
}
```

#### Content Analysis Configuration
```go
type AnalysisConfig struct {
    EnableSentiment   bool     `json:"enable_sentiment"`
    EnableEntities    bool     `json:"enable_entities"`
    EnableTopics      bool     `json:"enable_topics"`
    EnableLanguage    bool     `json:"enable_language"`
    EnableKeyPhrases  bool     `json:"enable_keyphrases"`
    MinConfidence     float64  `json:"min_confidence"`
    SupportedLanguages []string `json:"supported_languages"`
}
```

### Performance Characteristics

#### Embedding Inference
- **Throughput**: 450-1200 docs/sec (depending on model and hardware)
- **Latency**: 25-85ms per batch (batch size dependent)
- **Memory**: 512MB-2GB per loaded model
- **Scalability**: Horizontal scaling via model sharding

#### Content Analysis
- **Language Detection**: ~10,000 texts/sec
- **Sentiment Analysis**: ~8,000 texts/sec  
- **Entity Extraction**: ~5,000 texts/sec
- **Topic Modeling**: ~6,000 texts/sec
- **Key Phrase Extraction**: ~4,000 texts/sec

### Integration Points

#### REST API Endpoints
- `POST /ai/models/load` - Load embedding model
- `GET /ai/models` - List available models
- `GET /ai/models/{name}/health` - Model health status
- `POST /ai/analyze` - Comprehensive content analysis
- `POST /collections/{name}/documents` - Auto-embed documents
- `POST /collections/{name}/search/semantic` - Semantic search

#### Event System
```go
type AIEventListener interface {
    OnModelLoaded(modelName string, info ModelInfo)
    OnModelUnloaded(modelName string)
    OnContentAnalyzed(docID string, insights ContentInsights)
    OnEmbeddingGenerated(docID string, vectors [][]float32)
}
```

### Security Considerations

- **Model Validation**: Verify ONNX model integrity and compatibility
- **Input Sanitization**: Validate and sanitize text inputs
- **Resource Limits**: Memory and CPU usage controls
- **Access Control**: Model access permissions and API rate limiting
- **Data Privacy**: Configurable data retention and anonymization

---

## 7. API Design

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