# MCP Embedding Support Guide

EmbeddixDB's MCP (Model Context Protocol) server includes built-in support for automatic text embedding, allowing you to work with natural language content directly without managing embeddings manually.

## Overview

The MCP server can automatically convert text content into vector embeddings using ONNX models. This enables:

- Natural language search queries
- Text document storage without pre-computed embeddings
- Seamless integration with LLM applications

## Starting the MCP Server with Embedding Support

### Basic Usage

```bash
embeddix-mcp --enable-embedding --model /path/to/model.onnx
```

### Command Line Options

- `--enable-embedding`: Enable text embedding support
- `--model`: Path to ONNX embedding model file
- `--model-name`: Name of the embedding model (default: "all-MiniLM-L6-v2")
- `--persistence`: Storage type: memory, bolt, or badger (default: "memory")
- `--data`: Data directory path (default: "./data")
- `--verbose`: Enable verbose logging

### Example with All Options

```bash
embeddix-mcp \
  --enable-embedding \
  --model ./models/all-MiniLM-L6-v2.onnx \
  --model-name all-MiniLM-L6-v2 \
  --persistence bolt \
  --data ./vectordb \
  --verbose
```

## Using Embedding Features

### Adding Vectors with Text Content

When embedding is enabled, you can provide text content instead of raw vectors:

```json
{
  "tool": "add_vectors",
  "arguments": {
    "collection": "documents",
    "vectors": [
      {
        "id": "doc1",
        "content": "This is a sample document about machine learning.",
        "metadata": {
          "category": "ML",
          "author": "John Doe"
        }
      },
      {
        "id": "doc2",
        "content": "Natural language processing is a subset of AI.",
        "metadata": {
          "category": "NLP",
          "author": "Jane Smith"
        }
      }
    ]
  }
}
```

The server will:
1. Automatically generate embeddings for the content
2. Store the vectors with their metadata
3. Preserve the original content in metadata

### Searching with Natural Language

Search using natural language queries instead of vectors:

```json
{
  "tool": "search_vectors",
  "arguments": {
    "collection": "documents",
    "query": "What are the applications of artificial intelligence?",
    "limit": 5,
    "include_metadata": true
  }
}
```

### Mixed Mode

You can still provide raw vectors when needed:

```json
{
  "tool": "add_vectors",
  "arguments": {
    "collection": "documents",
    "vectors": [
      {
        "id": "vec1",
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"type": "pre-computed"}
      },
      {
        "id": "vec2",
        "content": "This will be auto-embedded",
        "metadata": {"type": "auto-embedded"}
      }
    ]
  }
}
```

## Supported Models

EmbeddixDB supports ONNX format embedding models. Recommended models include:

1. **all-MiniLM-L6-v2** (384 dimensions)
   - Balanced performance and quality
   - Good for general-purpose text

2. **all-mpnet-base-v2** (768 dimensions)
   - Higher quality embeddings
   - Better for semantic similarity

3. **sentence-transformers models**
   - Various sizes and languages
   - Convert to ONNX format

## Error Handling

The MCP server provides detailed error messages for embedding-related issues:

- **Model not found**: Specified model file doesn't exist
- **Invalid input**: Empty or malformed text content
- **Resource exhaustion**: System resources insufficient
- **Timeout**: Embedding generation took too long

## Performance Considerations

1. **Batch Processing**: The server automatically batches embeddings for efficiency
2. **Caching**: Model warmup reduces latency for subsequent requests
3. **Resource Usage**: Monitor memory usage with large models
4. **Concurrent Requests**: The server handles concurrent embedding requests safely

## Integration Examples

### Python Client

```python
import json
import sys

# Create a collection
create_collection = {
    "tool": "create_collection",
    "arguments": {
        "name": "knowledge_base",
        "dimension": 384,
        "distance": "cosine"
    }
}

# Add documents
add_docs = {
    "tool": "add_vectors",
    "arguments": {
        "collection": "knowledge_base",
        "vectors": [
            {
                "content": "Your document text here",
                "metadata": {"source": "doc1.txt"}
            }
        ]
    }
}

# Search
search = {
    "tool": "search_vectors",
    "arguments": {
        "collection": "knowledge_base",
        "query": "Find relevant documents",
        "limit": 10
    }
}

# Send to MCP server via stdin
print(json.dumps(add_docs))
```

### LLM Integration

The MCP embedding support is designed for seamless LLM integration:

1. Store conversation history as embeddings
2. Retrieve relevant context for RAG
3. Build semantic memory systems
4. Implement agent knowledge bases

## Troubleshooting

### Model Loading Issues

```bash
# Verify model file
ls -la /path/to/model.onnx

# Check model compatibility
# The model should have 'input_ids' input and produce embeddings output
```

### Performance Issues

```bash
# Start with verbose logging
embeddix-mcp --enable-embedding --model ./model.onnx --verbose

# Monitor resource usage
top -p $(pgrep embeddix-mcp)
```

### Connection Issues

```bash
# Test basic MCP connection first
echo '{"tool": "list_collections", "arguments": {}}' | embeddix-mcp
```

## Best Practices

1. **Model Selection**: Choose models based on your use case
2. **Collection Setup**: Match collection dimension to model output
3. **Metadata**: Store additional context in metadata
4. **Error Handling**: Implement retry logic for transient errors
5. **Resource Management**: Close connections properly

## Next Steps

- [MCP Protocol Specification](MCP_PROTOCOL.md)
- [API Reference](API_REFERENCE.md)
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)