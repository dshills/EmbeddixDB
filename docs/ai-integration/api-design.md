# API Design: AI Integration

## REST API Endpoints

### Auto-Embedding API

#### Embed Documents
```http
POST /v1/collections/{collection_name}/embed
Content-Type: application/json
Authorization: Bearer {token}

{
  "documents": [
    {
      "id": "doc1",
      "content": "Machine learning is transforming healthcare by enabling more accurate diagnoses...",
      "metadata": {
        "category": "healthcare",
        "author": "Dr. Smith",
        "date": "2024-01-15",
        "source": "medical_journal"
      },
      "type": "text"
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "options": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "normalize_vectors": true,
    "extract_metadata": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "processed": 1,
  "vectors_created": 3,
  "processing_time_ms": 150,
  "model_info": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "version": "1.0"
  },
  "results": [
    {
      "document_id": "doc1",
      "chunks": [
        {
          "chunk_id": "doc1_chunk_0",
          "vector_id": "vec_001",
          "content": "Machine learning is transforming healthcare...",
          "embedding": [0.1, 0.2, ...],
          "metadata": {
            "chunk_index": 0,
            "start_pos": 0,
            "end_pos": 127
          }
        }
      ],
      "insights": {
        "language": "en",
        "topics": ["healthcare", "machine_learning"],
        "entities": [
          {"text": "Dr. Smith", "label": "PERSON", "confidence": 0.95}
        ],
        "sentiment": {"polarity": 0.1, "confidence": 0.8}
      }
    }
  ]
}
```

#### Hybrid Search
```http
POST /v1/collections/{collection_name}/search/hybrid
Content-Type: application/json

{
  "query": "latest advances in cancer treatment using AI",
  "limit": 10,
  "fusion": {
    "algorithm": "rrf",
    "weights": {
      "vector": 0.7,
      "text": 0.3,
      "freshness": 0.1
    }
  },
  "options": {
    "include_explanation": true,
    "expand_query": true,
    "rerank": true,
    "min_score": 0.1
  },
  "filters": {
    "category": "healthcare",
    "date": {
      "gte": "2023-01-01"
    }
  }
}
```

**Response:**
```json
{
  "query_info": {
    "original": "latest advances in cancer treatment using AI",
    "processed": "cancer treatment artificial intelligence advances recent",
    "intent": {
      "type": "factual",
      "confidence": 0.89
    },
    "expansion": {
      "synonyms": ["oncology", "tumor", "machine learning", "ML"],
      "concepts": ["medical AI", "treatment protocols", "diagnosis"],
      "entities": [
        {"text": "cancer", "type": "MEDICAL_CONDITION"},
        {"text": "AI", "type": "TECHNOLOGY"}
      ]
    }
  },
  "results": [
    {
      "id": "vec_001",
      "score": 0.92,
      "document_id": "doc1",
      "content": "Recent breakthrough in AI-powered cancer diagnosis...",
      "metadata": {
        "title": "AI in Oncology: 2024 Review",
        "category": "healthcare",
        "date": "2024-01-15"
      },
      "explanation": {
        "vector_score": 0.85,
        "text_score": 0.78,
        "fusion_score": 0.92,
        "matched_terms": ["cancer", "AI", "treatment"],
        "semantic_similarity": 0.85
      }
    }
  ],
  "performance": {
    "total_time_ms": 45,
    "vector_search_ms": 15,
    "text_search_ms": 20,
    "fusion_ms": 8,
    "rerank_ms": 2
  },
  "debug_info": {
    "vector_candidates": 100,
    "text_candidates": 85,
    "fusion_candidates": 150,
    "final_results": 10
  }
}
```

### Model Management API

#### List Available Models
```http
GET /v1/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "sentence-transformers/all-MiniLM-L6-v2",
      "version": "1.0",
      "status": "ready",
      "info": {
        "dimension": 384,
        "max_tokens": 512,
        "languages": ["en"],
        "use_case": "general-purpose",
        "size_mb": 22
      },
      "performance": {
        "tokens_per_second": 2500,
        "memory_usage_mb": 150,
        "accuracy_score": 0.85
      },
      "usage": {
        "requests_today": 1524,
        "total_requests": 45832
      }
    }
  ]
}
```

#### Load Model
```http
POST /v1/models/{model_name}/load
Content-Type: application/json

{
  "version": "latest",
  "config": {
    "batch_size": 32,
    "enable_gpu": true,
    "optimization_level": 2
  }
}
```

#### Model Health Check
```http
GET /v1/models/{model_name}/health
```

**Response:**
```json
{
  "status": "healthy",
  "loaded_at": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "performance": {
    "avg_latency_ms": 25,
    "p95_latency_ms": 45,
    "throughput_rps": 100,
    "error_rate": 0.001
  },
  "resources": {
    "cpu_usage": 0.65,
    "memory_usage_mb": 150,
    "gpu_usage": 0.80,
    "gpu_memory_mb": 1200
  }
}
```

### Content Analysis API

#### Analyze Content
```http
POST /v1/analyze/content
Content-Type: application/json

{
  "content": "Artificial intelligence is revolutionizing the healthcare industry...",
  "options": {
    "extract_topics": true,
    "extract_entities": true,
    "analyze_sentiment": true,
    "generate_summary": true,
    "detect_language": true
  }
}
```

**Response:**
```json
{
  "insights": {
    "language": {
      "detected": "en",
      "confidence": 0.99
    },
    "topics": [
      {
        "label": "artificial_intelligence",
        "confidence": 0.95,
        "keywords": ["AI", "machine learning", "automation"]
      },
      {
        "label": "healthcare",
        "confidence": 0.89,
        "keywords": ["medical", "diagnosis", "treatment"]
      }
    ],
    "entities": [
      {
        "text": "healthcare industry",
        "label": "INDUSTRY",
        "confidence": 0.92,
        "start_pos": 45,
        "end_pos": 62
      }
    ],
    "sentiment": {
      "polarity": 0.2,
      "confidence": 0.85,
      "label": "positive"
    },
    "summary": "Content discusses AI applications in healthcare...",
    "readability": {
      "score": 12.5,
      "level": "college"
    },
    "complexity": 0.7
  },
  "statistics": {
    "word_count": 156,
    "sentence_count": 8,
    "paragraph_count": 3,
    "unique_words": 98
  }
}
```

### Query Intelligence API

#### Analyze Query Intent
```http
POST /v1/analyze/query
Content-Type: application/json

{
  "query": "how to treat diabetes with machine learning",
  "context": {
    "user_id": "user123",
    "session_id": "session456", 
    "previous_queries": [
      "diabetes complications",
      "AI in medicine"
    ]
  }
}
```

**Response:**
```json
{
  "intent": {
    "type": "procedural",
    "confidence": 0.87,
    "subtype": "how_to"
  },
  "entities": [
    {
      "text": "diabetes",
      "type": "MEDICAL_CONDITION",
      "confidence": 0.95
    },
    {
      "text": "machine learning",
      "type": "TECHNOLOGY",
      "confidence": 0.91
    }
  ],
  "expansion": {
    "synonyms": ["ML", "AI", "artificial intelligence"],
    "related_terms": ["treatment", "therapy", "medical AI"],
    "concepts": ["predictive modeling", "clinical decision support"]
  },
  "suggested_filters": {
    "category": ["healthcare", "technology"],
    "content_type": ["research", "clinical_trial"],
    "date_range": "recent"
  },
  "complexity": 0.6,
  "specificity": 0.8
}
```

## GraphQL API

### Schema Definition

```graphql
type Query {
  # Search operations
  hybridSearch(input: HybridSearchInput!): HybridSearchResult!
  semanticSearch(input: SemanticSearchInput!): SemanticSearchResult!
  
  # Model operations
  models: [Model!]!
  model(name: String!): Model
  
  # Content analysis
  analyzeContent(content: String!, options: AnalysisOptions): ContentInsights!
  analyzeQuery(query: String!, context: QueryContext): QueryAnalysis!
  
  # Performance metrics
  searchMetrics(timeRange: TimeRange!): SearchMetrics!
}

type Mutation {
  # Document operations
  embedDocuments(input: EmbedDocumentsInput!): EmbedResult!
  updateDocuments(input: UpdateDocumentsInput!): UpdateResult!
  
  # Model operations
  loadModel(name: String!, config: ModelConfig): LoadModelResult!
  unloadModel(name: String!): Boolean!
  
  # Feedback
  submitFeedback(input: FeedbackInput!): Boolean!
  
  # Configuration
  updateSearchWeights(weights: SearchWeights!): Boolean!
}

type Subscription {
  # Real-time updates
  embeddingProgress(jobId: String!): EmbeddingProgress!
  modelHealth(modelName: String!): ModelHealth!
  searchPerformance: SearchPerformanceUpdate!
}

# Input types
input HybridSearchInput {
  collection: String!
  query: String!
  limit: Int = 10
  fusion: FusionConfig
  filters: JSON
  options: SearchOptions
}

input EmbedDocumentsInput {
  collection: String!
  documents: [DocumentInput!]!
  model: String
  options: EmbedOptions
}

# Response types
type HybridSearchResult {
  results: [SearchResult!]!
  queryInfo: QueryInfo!
  performance: SearchPerformance!
  debugInfo: SearchDebugInfo
}

type SearchResult {
  id: String!
  score: Float!
  content: String!
  metadata: JSON!
  explanation: SearchExplanation
}

type Model {
  name: String!
  version: String!
  status: ModelStatus!
  info: ModelInfo!
  performance: ModelPerformance!
  usage: ModelUsage!
}

enum ModelStatus {
  LOADING
  READY
  ERROR
  UNLOADED
}
```

### Example GraphQL Queries

```graphql
# Hybrid search with detailed results
query SearchWithExplanation($collection: String!, $query: String!) {
  hybridSearch(input: {
    collection: $collection
    query: $query
    limit: 10
    fusion: {
      algorithm: RRF
      weights: { vector: 0.7, text: 0.3 }
    }
    options: {
      includeExplanation: true
      expandQuery: true
      rerank: true
    }
  }) {
    results {
      id
      score
      content
      metadata
      explanation {
        vectorScore
        textScore
        fusionScore
        matchedTerms
      }
    }
    queryInfo {
      original
      processed
      intent {
        type
        confidence
      }
      expansion {
        synonyms
        concepts
      }
    }
    performance {
      totalTimeMs
      vectorSearchMs
      textSearchMs
      fusionMs
    }
  }
}

# Model management
query GetModelHealth($modelName: String!) {
  model(name: $modelName) {
    name
    status
    performance {
      avgLatencyMs
      throughputRps
      errorRate
    }
    usage {
      requestsToday
      totalRequests
    }
  }
}

# Content analysis
query AnalyzeDocument($content: String!) {
  analyzeContent(
    content: $content
    options: {
      extractTopics: true
      extractEntities: true
      analyzeSentiment: true
      generateSummary: true
    }
  ) {
    language {
      detected
      confidence
    }
    topics {
      label
      confidence
      keywords
    }
    entities {
      text
      label
      confidence
    }
    sentiment {
      polarity
      label
      confidence
    }
    summary
  }
}
```

## WebSocket API

### Real-time Embedding Progress

```javascript
// Connect to embedding progress stream
const ws = new WebSocket('ws://localhost:8080/v1/ws/embed-progress');

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Progress: ${progress.processed}/${progress.total} documents`);
  
  // Progress event structure:
  // {
  //   "job_id": "embed_job_123",
  //   "processed": 150,
  //   "total": 1000,
  //   "percentage": 15.0,
  //   "current_document": "doc_150",
  //   "estimated_completion": "2024-01-15T11:45:00Z",
  //   "performance": {
  //     "documents_per_second": 25.5,
  //     "avg_processing_time_ms": 39
  //   }
  // }
};
```

### Model Health Monitoring

```javascript
// Monitor model health in real-time
const healthWS = new WebSocket('ws://localhost:8080/v1/ws/model-health/sentence-transformers');

healthWS.onmessage = (event) => {
  const health = JSON.parse(event.data);
  
  // Health event structure:
  // {
  //   "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  //   "timestamp": "2024-01-15T10:30:00Z",
  //   "status": "healthy",
  //   "metrics": {
  //     "cpu_usage": 0.65,
  //     "memory_mb": 150,
  //     "requests_per_second": 100,
  //     "avg_latency_ms": 25,
  //     "error_rate": 0.001
  //   },
  //   "alerts": []
  // }
};
```

## SDK Examples

### Python SDK

```python
from embeddixdb import EmbeddixClient

# Initialize client
client = EmbeddixClient(
    url="http://localhost:8080",
    api_key="your-api-key"
)

# Auto-embed documents
result = client.embed_documents(
    collection="healthcare_docs",
    documents=[
        {
            "id": "doc1",
            "content": "Machine learning in healthcare...",
            "metadata": {"category": "healthcare"}
        }
    ],
    model="sentence-transformers/all-MiniLM-L6-v2",
    options={
        "chunk_size": 512,
        "extract_metadata": True
    }
)

# Hybrid search
search_results = client.hybrid_search(
    collection="healthcare_docs",
    query="AI cancer treatment",
    fusion={
        "algorithm": "rrf",
        "weights": {"vector": 0.7, "text": 0.3}
    },
    options={
        "expand_query": True,
        "rerank": True,
        "include_explanation": True
    }
)

# Content analysis
insights = client.analyze_content(
    content="Artificial intelligence is transforming healthcare...",
    options={
        "extract_topics": True,
        "extract_entities": True,
        "analyze_sentiment": True
    }
)

print(f"Detected topics: {insights['topics']}")
print(f"Entities: {insights['entities']}")
print(f"Sentiment: {insights['sentiment']['label']}")
```

### JavaScript SDK

```javascript
import { EmbeddixClient } from '@embeddixdb/client';

const client = new EmbeddixClient({
  url: 'http://localhost:8080',
  apiKey: 'your-api-key'
});

// Embed documents with progress tracking
const embedJob = await client.embedDocuments({
  collection: 'research_papers',
  documents: documents,
  model: 'sentence-transformers/all-MiniLM-L6-v2'
});

// Track progress
embedJob.onProgress((progress) => {
  console.log(`Progress: ${progress.percentage}%`);
});

await embedJob.complete();

// Perform hybrid search
const results = await client.hybridSearch({
  collection: 'research_papers',
  query: 'quantum computing applications',
  fusion: {
    algorithm: 'rrf',
    weights: { vector: 0.8, text: 0.2 }
  },
  options: {
    expandQuery: true,
    rerank: true,
    minScore: 0.1
  }
});

console.log(`Found ${results.length} results`);
results.forEach(result => {
  console.log(`${result.score}: ${result.content.substring(0, 100)}...`);
});
```

This API design provides a comprehensive interface for all AI integration features while maintaining consistency and ease of use across different programming languages and integration patterns.