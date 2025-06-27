# Ollama Embedding Integration Guide

This guide explains how to use Ollama for generating embeddings in EmbeddixDB.

## Overview

EmbeddixDB now supports Ollama as an embedding provider, allowing you to use locally-hosted embedding models through the Ollama API. This provides an alternative to ONNX models with the following benefits:

- Easy model management through Ollama
- Support for a wide variety of embedding models
- No need to manage model files directly
- Simple HTTP API integration

## Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull an embedding model:
   ```bash
   ollama pull nomic-embed-text
   # or
   ollama pull all-minilm
   ```
3. Ensure Ollama is running (default port: 11434)

## Configuration

### Basic Configuration

```go
import (
    "github.com/dshills/EmbeddixDB/core/ai"
    "github.com/dshills/EmbeddixDB/core/ai/embedding"
)

config := ai.ModelConfig{
    Type:                "ollama",
    Path:                "nomic-embed-text",  // Model name in Ollama
    OllamaEndpoint:      "http://localhost:11434",  // Optional, this is the default
    NormalizeEmbeddings: true,
    BatchSize:           32,
}

engine, err := embedding.CreateEngine(config)
if err != nil {
    log.Fatal(err)
}
```

### Using with ModelManager

```go
manager := ai.NewModelManager(10)  // Max 10 models

// Set the engine factory to support Ollama
manager.SetEngineFactory(embedding.CreateEngine)

// Load an Ollama model
config := ai.ModelConfig{
    Type: "ollama",
    Path: "nomic-embed-text",
}

err := manager.LoadModel(ctx, "text-embedder", config)
if err != nil {
    log.Fatal(err)
}

// Use the model
engine, err := manager.GetEngine("text-embedder")
if err != nil {
    log.Fatal(err)
}

embeddings, err := engine.Embed(ctx, []string{"Hello, world!"})
```

## Supported Models

Any embedding model available in Ollama can be used. Popular options include:

| Model | Dimensions | Description |
|-------|------------|-------------|
| nomic-embed-text | 768 | High-quality text embeddings |
| all-minilm | 384 | Lightweight, fast embeddings |
| mxbai-embed-large | 1024 | Large, high-quality embeddings |

## API Usage

### Single Embedding

```go
ctx := context.Background()
content := []string{"This is a test document"}

embeddings, err := engine.Embed(ctx, content)
if err != nil {
    log.Fatal(err)
}

// embeddings[0] contains the embedding vector for the input text
```

### Batch Embedding

```go
content := []string{
    "First document",
    "Second document",
    "Third document",
}

embeddings, err := engine.EmbedBatch(ctx, content, 2)  // Process in batches of 2
if err != nil {
    log.Fatal(err)
}
```

## Performance Considerations

1. **Batch Processing**: While Ollama doesn't natively support batch embedding, the engine processes requests sequentially. Use `EmbedBatch` for better error handling and progress tracking.

2. **Connection Pooling**: The engine uses a single HTTP client with a 30-second timeout. Adjust if needed for longer processing times.

3. **Model Loading**: The first request after creating an engine may be slower as Ollama loads the model. Use `Warm()` to pre-load:
   ```go
   err := engine.Warm(ctx)
   ```

4. **Dimension Auto-Detection**: If you don't specify dimensions in the config, the engine will auto-detect them on the first embedding request.

## Error Handling

The Ollama engine returns structured errors that can be inspected:

```go
embeddings, err := engine.Embed(ctx, content)
if err != nil {
    if embErr, ok := err.(*embedding.EmbeddingError); ok {
        log.Printf("Operation: %s, Model: %s, Retryable: %v",
            embErr.Op, embErr.Model, embErr.IsRetryable())
    }
}
```

## Custom Ollama Endpoints

If Ollama is running on a different host or port:

```go
config := ai.ModelConfig{
    Type:           "ollama",
    Path:           "nomic-embed-text",
    OllamaEndpoint: "http://192.168.1.100:11434",
}
```

## Comparing with ONNX

| Feature | Ollama | ONNX |
|---------|--------|------|
| Model Management | Via Ollama CLI | Manual file management |
| Model Variety | Wide selection | Limited to converted models |
| Performance | HTTP overhead | Direct inference |
| GPU Support | Automatic via Ollama | Requires ONNX Runtime GPU |
| Offline Usage | Yes (after model pull) | Yes |
| Memory Usage | Managed by Ollama | Direct control |

## Troubleshooting

### Connection Errors
- Ensure Ollama is running: `ollama serve`
- Check the endpoint URL is correct
- Verify the model is installed: `ollama list`

### Model Not Found
- Pull the model first: `ollama pull <model-name>`
- Check exact model name with: `ollama list`

### Timeout Errors
- Increase the HTTP client timeout for large models
- Ensure sufficient system resources
- Check Ollama logs for issues

## Example: Complete Integration

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/dshills/EmbeddixDB/core"
    "github.com/dshills/EmbeddixDB/core/ai"
    "github.com/dshills/EmbeddixDB/core/ai/embedding"
)

func main() {
    // Create vector store
    store, err := core.NewVectorStore(core.Config{
        PersistenceType: "memory",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer store.Close()

    // Configure Ollama embedding
    embeddingConfig := ai.ModelConfig{
        Type: "ollama",
        Path: "nomic-embed-text",
    }

    // Create embedding engine
    engine, err := embedding.CreateEngine(embeddingConfig)
    if err != nil {
        log.Fatal(err)
    }

    // Warm up the engine
    ctx := context.Background()
    if err := engine.Warm(ctx); err != nil {
        log.Fatal(err)
    }

    // Create a collection
    if err := store.CreateCollection("documents", 768, "cosine"); err != nil {
        log.Fatal(err)
    }

    // Generate embeddings and add to store
    texts := []string{
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "EmbeddixDB provides high-performance vector storage",
    }

    embeddings, err := engine.Embed(ctx, texts)
    if err != nil {
        log.Fatal(err)
    }

    // Add vectors to the store
    for i, embedding := range embeddings {
        vector := core.Vector{
            ID:     fmt.Sprintf("doc_%d", i),
            Values: embedding,
            Metadata: map[string]interface{}{
                "text": texts[i],
            },
        }
        
        if err := store.AddVector("documents", vector); err != nil {
            log.Fatal(err)
        }
    }

    // Search for similar documents
    query := "database technology"
    queryEmbedding, err := engine.Embed(ctx, []string{query})
    if err != nil {
        log.Fatal(err)
    }

    results, err := store.SearchVectors("documents", queryEmbedding[0], 3, nil)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Search results for '%s':\n", query)
    for _, result := range results {
        fmt.Printf("- Score: %.3f, Text: %s\n", 
            result.Score, result.Metadata["text"])
    }
}
```

## Contributing

When contributing Ollama-related features:

1. Add tests using the mock Ollama server
2. Update this documentation
3. Consider adding new Ollama-specific features
4. Ensure backward compatibility