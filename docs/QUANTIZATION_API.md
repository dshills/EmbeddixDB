# Quantization API Documentation

## Overview

EmbeddixDB supports advanced vector quantization techniques for memory-efficient storage and search. This document describes the quantization APIs and configuration options available.

## Quantization Types

### Product Quantization (PQ)
Divides vectors into subvectors and quantizes each independently using k-means clustering.

**Benefits:**
- 8x to 256x memory compression
- Maintains search accuracy through reranking
- Suitable for high-dimensional vectors

**Configuration:**
```go
quantization.QuantizerConfig{
    Type:              quantization.ProductQuantization,
    Dimension:         384,
    NumSubvectors:     8,        // Number of subvectors
    BitsPerSubvector:  8,        // 4-8 bits (determines compression)
    MemoryBudgetMB:    50,       // Memory limit for codebooks
    DistanceMetric:    "cosine",
    EnableAsymmetric:  true,     // Better accuracy for search
    TrainingTimeout:   300,      // Seconds for training
}
```

### Scalar Quantization (SQ)
Quantizes each vector component independently with uniform binning.

**Benefits:**
- 2x to 8x memory compression
- Fast encode/decode operations
- Good for lower-dimensional vectors

**Configuration:**
```go
quantization.QuantizerConfig{
    Type:             quantization.ScalarQuantization,
    Dimension:        128,
    BitsPerComponent: 8,         // 4, 6, or 8 bits
    DistanceMetric:   "l2",
}
```

## Creating Quantized Indexes

### Via Go API

```go
import (
    "github.com/dshills/EmbeddixDB/index"
    "github.com/dshills/EmbeddixDB/core/quantization"
)

// Create quantized HNSW configuration
config := index.QuantizedHNSWConfig{
    HNSWConfig: index.DefaultHNSWConfig(),
    QuantizerConfig: quantization.QuantizerConfig{
        Type:              quantization.ProductQuantization,
        Dimension:         384,
        MemoryBudgetMB:    20,
        DistanceMetric:    "cosine",
        EnableAsymmetric:  true,
    },
    RerankerConfig: index.RerankerConfig{
        Enable:        true,
        RerankerRatio: 0.2,  // Rerank top 20%
        MinCandidates: 10,
        MaxCandidates: 100,
    },
    KeepOriginalVectors: true,
}

// Create quantized index
qindex, err := index.NewQuantizedHNSW(config)
if err != nil {
    return err
}

// Train the quantizer
trainingVectors := [][]float32{...}
ctx := context.Background()
err = qindex.Train(ctx, trainingVectors)
if err != nil {
    return err
}

// Insert vectors
vector := core.Vector{
    ID:     "doc1",
    Values: embedding,
    Metadata: map[string]string{"category": "tech"},
}
err = qindex.Insert(vector)

// Search
results, err := qindex.Search(queryVector, 10, nil)
```

### Via REST API

#### Create Collection with Quantization

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "quantized_docs",
    "dimension": 384,
    "index_type": "quantized_hnsw",
    "distance": "cosine",
    "quantization": {
      "type": "product_quantization",
      "memory_budget_mb": 20,
      "enable_asymmetric": true
    },
    "reranker": {
      "enable": true,
      "reranker_ratio": 0.2,
      "min_candidates": 10
    }
  }'
```

#### Train Quantizer

```bash
curl -X POST http://localhost:8080/collections/quantized_docs/quantizer/train \
  -H "Content-Type: application/json" \
  -d '{
    "training_vectors": [
      [0.1, 0.2, 0.3, ...],
      [0.2, 0.3, 0.4, ...],
      ...
    ]
  }'
```

#### Check Quantizer Status

```bash
curl http://localhost:8080/collections/quantized_docs/quantizer/status

# Response:
# {
#   "trained": true,
#   "memory_reduction": 256.0,
#   "code_size": 8,
#   "quantizer_type": "product_quantization"
# }
```

## Configuration Options

### Quantizer Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `Type` | string | Quantization type: "product_quantization" or "scalar_quantization" | "product_quantization" |
| `Dimension` | int | Vector dimension | Required |
| `NumSubvectors` | int | Number of subvectors (PQ only) | Auto-calculated |
| `BitsPerSubvector` | int | Bits per subvector (PQ): 4-8 | 8 |
| `BitsPerComponent` | int | Bits per component (SQ): 4, 6, 8 | 8 |
| `MemoryBudgetMB` | int | Memory limit for codebooks | 10 |
| `DistanceMetric` | string | "l2", "cosine", "dot" | "l2" |
| `EnableAsymmetric` | bool | Use asymmetric distance for better accuracy | true |
| `TrainingTimeout` | int | Training timeout in seconds | 300 |

### Reranker Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `Enable` | bool | Enable reranking with original vectors | true |
| `RerankerRatio` | float64 | Fraction of candidates to rerank (0.1-1.0) | 0.2 |
| `MinCandidates` | int | Minimum candidates for reranking | 10 |
| `MaxCandidates` | int | Maximum candidates for reranking | 100 |
| `UseAsymmetric` | bool | Use asymmetric distance in reranking | true |
| `CacheDistTable` | bool | Cache distance tables (future optimization) | true |

## Performance Tuning

### Memory vs. Accuracy Trade-offs

**High Compression (256x):**
```go
QuantizerConfig{
    Type:             quantization.ProductQuantization,
    BitsPerSubvector: 4,  // 16 clusters per subvector
    RerankerRatio:    0.3, // Rerank more candidates
}
```

**Balanced (64x):**
```go
QuantizerConfig{
    Type:             quantization.ProductQuantization,
    BitsPerSubvector: 6,  // 64 clusters per subvector
    RerankerRatio:    0.2,
}
```

**Lower Compression, Higher Speed (16x):**
```go
QuantizerConfig{
    Type:             quantization.ProductQuantization,
    BitsPerSubvector: 8,  // 256 clusters per subvector
    RerankerRatio:    0.1,
}
```

### Memory Budget Guidelines

| Vector Count | Dimension | Recommended Budget |
|--------------|-----------|-------------------|
| 100K | 128 | 5MB |
| 500K | 256 | 20MB |
| 1M | 384 | 50MB |
| 5M | 512 | 200MB |

### Training Set Size Guidelines

- **Minimum**: 1000 vectors
- **Recommended**: 10K-100K vectors
- **Rule of thumb**: 100x the number of clusters
- **For PQ**: At least 256 * NumSubvectors training vectors

## Monitoring and Statistics

### Get Index Statistics

```bash
curl http://localhost:8080/collections/quantized_docs/stats

# Response:
# {
#   "quantized_vectors": 50000,
#   "original_vectors": 50000,
#   "memory_reduction": 256.0,
#   "code_size": 8,
#   "quantizer_trained": true,
#   "reranker_enabled": true
# }
```

### Memory Usage Estimation

```bash
curl http://localhost:8080/collections/quantized_docs/memory

# Response:
# {
#   "base_memory_usage": 3200000,
#   "quantized_vectors": 400000,
#   "original_vectors": 102400000,
#   "total_memory": 105600000,
#   "memory_reduction": 256.0
# }
```

## Best Practices

### Training
1. Use representative data for training
2. Ensure sufficient training data (>1000 vectors)
3. Use a subset of your dataset for training
4. Train once and reuse the quantizer

### Configuration
1. Start with default settings and tune based on requirements
2. Use Product Quantization for high-dimensional vectors (>100D)
3. Use Scalar Quantization for lower-dimensional vectors (<100D)
4. Enable reranking for accuracy-critical applications

### Memory Management
1. Set appropriate memory budgets to avoid OOM
2. Monitor memory usage during training
3. Consider the training data size vs. memory budget ratio
4. Use incremental training for very large datasets

### Performance
1. Enable asymmetric distance for better search accuracy
2. Tune reranker ratio based on accuracy requirements
3. Use higher bits per subvector for better accuracy
4. Profile search performance vs. memory usage

## Error Handling

### Common Errors

**Training Errors:**
```
- "insufficient training data": Need more training vectors
- "training timeout": Increase training timeout
- "memory budget exceeded": Reduce memory budget or increase limit
```

**Search Errors:**
```
- "quantizer not trained": Call Train() before Insert()
- "dimension mismatch": Ensure query matches index dimension
- "no candidates found": Check training data quality
```

### Troubleshooting

1. **Poor Search Quality**: Increase reranker ratio or bits per subvector
2. **High Memory Usage**: Reduce memory budget or bits per subvector
3. **Slow Training**: Reduce training data size or increase timeout
4. **Training Failures**: Check training data distribution and size

## Migration Guide

### From Standard HNSW

```go
// Old HNSW index
hnswConfig := index.DefaultHNSWConfig()
hnswIndex := index.NewHNSWIndex(dimension, distanceMetric, hnswConfig)

// New quantized HNSW index
quantizedConfig := index.DefaultQuantizedHNSWConfig(dimension, distanceMetric)
quantizedIndex, err := index.NewQuantizedHNSW(quantizedConfig)

// Train before use
err = quantizedIndex.Train(ctx, trainingVectors)

// Same API for insert and search
quantizedIndex.Insert(vector)
results, err := quantizedIndex.Search(query, k, filter)
```

### Configuration Migration

Standard HNSW configurations remain the same:
- `M`, `EfConstruction`, `EfSearch` parameters are preserved
- Add quantization and reranker configuration
- Training step is required for quantized indexes

## Examples

See the test files for comprehensive examples:
- `index/quantized_hnsw_test.go` - Basic usage examples
- `core/quantization/*_test.go` - Quantizer-specific examples
- `benchmark/quantization_benchmark.go` - Performance benchmarks