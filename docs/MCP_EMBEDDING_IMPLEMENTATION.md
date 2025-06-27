# MCP Embedding Support Implementation

## Summary

The MCP (Model Context Protocol) server in EmbeddixDB now has full embedding support, enabling users to work with text content directly without managing embeddings manually.

## Key Components

### 1. EmbeddingStore (`mcp/embedding_store.go`)
- Wraps the base VectorStore with embedding capabilities
- Implements `EmbedText` and `EmbedTexts` methods
- Thread-safe with proper mutex locking
- Graceful resource cleanup on close

### 2. Command Line Integration (`cmd/embeddix-mcp/main.go`)
- Added flags:
  - `--enable-embedding`: Enable embedding support
  - `--model`: Path to ONNX model file
  - `--model-name`: Model identifier (default: "all-MiniLM-L6-v2")
- Automatic EmbeddingStore creation when enabled
- Seamless integration with existing persistence options

### 3. Handler Integration (`mcp/handlers.go`)
- Handlers use Go interface assertions to detect embedding support
- `SearchVectorsHandler`: Converts text queries to embeddings
- `AddVectorsHandler`: Embeds content field if provided
- Backward compatible - still accepts raw vectors

## How It Works

### Text to Vector Flow
1. User provides `content` field instead of `vector`
2. Handler detects EmbeddingStore interface
3. Calls `EmbedText` to generate embeddings
4. Stores resulting vector with metadata
5. Original content preserved in metadata

### Search Flow
1. User provides natural language `query`
2. Handler embeds query text
3. Performs vector similarity search
4. Returns results with metadata

## Error Handling

The implementation includes comprehensive error handling:
- Model not found errors
- Invalid/empty input validation
- Resource exhaustion handling
- Proper error propagation to MCP responses

## Testing

- Unit tests in `mcp/embedding_test.go`
- Mock embedding engine for testing
- Integration tests verify handler behavior
- All tests passing

## Usage Example

```bash
# Start server with embedding
embeddix-mcp --enable-embedding --model ./model.onnx

# Add document
{
  "tool": "add_vectors",
  "arguments": {
    "collection": "docs",
    "vectors": [{
      "content": "Text to be embedded",
      "metadata": {"source": "doc1"}
    }]
  }
}

# Search
{
  "tool": "search_vectors",
  "arguments": {
    "collection": "docs",
    "query": "Natural language search"
  }
}
```

## Benefits

1. **Simplified API**: Users work with text, not vectors
2. **Flexibility**: Supports both text and raw vectors
3. **Performance**: Batched embedding for efficiency
4. **Integration**: Works with all persistence backends
5. **Standards**: Follows MCP protocol conventions