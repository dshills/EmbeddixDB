# EmbeddixDB MCP Examples

## Basic Workflow

This example demonstrates a complete workflow using the MCP server to store and retrieve semantic memories.

### 1. Start the MCP Server

```bash
# Start with in-memory storage (for testing)
./mcp-server -verbose

# Or with persistent storage
./mcp-server -persistence bolt -data ./data -verbose
```

### 2. Initialize Connection

```json
// --> Request
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "example-client",
      "version": "1.0.0"
    }
  },
  "id": 1
}

// <-- Response
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {"listChanged": false}
    },
    "serverInfo": {
      "name": "EmbeddixDB MCP Server",
      "version": "0.1.0"
    }
  },
  "id": 1
}
```

### 3. Create a Collection

```json
// --> Request
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "create_collection",
    "arguments": {
      "name": "agent_memory",
      "dimension": 384,
      "distance": "cosine",
      "indexType": "hnsw"
    }
  },
  "id": 2
}

// <-- Response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Successfully created collection 'agent_memory'"
      },
      {
        "type": "text",
        "text": "{\"name\":\"agent_memory\",\"dimension\":384,\"distance\":\"cosine\",\"indexType\":\"hnsw\",\"created\":true}"
      }
    ]
  },
  "id": 2
}
```

### 4. Add Memories

```json
// --> Request
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "add_vectors",
    "arguments": {
      "collection": "agent_memory",
      "vectors": [
        {
          "id": "mem_001",
          "vector": [0.1, 0.2, 0.15, /* ... 384 dimensions total ... */],
          "metadata": {
            "content": "User mentioned they work as a software engineer",
            "timestamp": "2024-01-15T10:30:00Z",
            "type": "user_info",
            "confidence": 0.9
          }
        },
        {
          "id": "mem_002",
          "vector": [0.05, 0.3, 0.25, /* ... 384 dimensions total ... */],
          "metadata": {
            "content": "User prefers Python for machine learning projects",
            "timestamp": "2024-01-15T10:35:00Z",
            "type": "preference",
            "topic": "programming",
            "confidence": 0.95
          }
        }
      ]
    }
  },
  "id": 3
}

// <-- Response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Successfully added 2 vectors"
      },
      {
        "type": "text",
        "text": "{\"added\":2,\"ids\":[\"mem_001\",\"mem_002\"],\"errors\":null}"
      }
    ]
  },
  "id": 3
}
```

### 5. Search for Relevant Memories

```json
// --> Request
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_vectors",
    "arguments": {
      "collection": "agent_memory",
      "vector": [0.08, 0.25, 0.2, /* ... query vector ... */],
      "limit": 5,
      "includeMetadata": true,
      "filters": {
        "type": "preference"
      }
    }
  },
  "id": 4
}

// <-- Response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 1 matching vectors"
      },
      {
        "type": "text",
        "text": "[{\"id\":\"mem_002\",\"score\":0.92,\"content\":\"User prefers Python for machine learning projects\",\"metadata\":{\"confidence\":0.95,\"content\":\"User prefers Python for machine learning projects\",\"timestamp\":\"2024-01-15T10:35:00Z\",\"topic\":\"programming\",\"type\":\"preference\"}}]"
      }
    ]
  },
  "id": 4
}
```

## Advanced Examples

### Personalized Search with User Context

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_vectors",
    "arguments": {
      "collection": "shared_knowledge",
      "query": "best practices for API design",
      "limit": 10,
      "userId": "user_123",
      "sessionId": "session_abc",
      "includeMetadata": true
    }
  },
  "id": 5
}
```

### Batch Vector Addition with Mixed Content

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "add_vectors",
    "arguments": {
      "collection": "documents",
      "vectors": [
        {
          "content": "The quick brown fox jumps over the lazy dog",
          "metadata": {
            "source": "example.txt",
            "type": "text"
          }
        },
        {
          "id": "custom_id_123",
          "vector": [0.1, 0.2, 0.3, /* ... pre-computed embedding ... */],
          "metadata": {
            "source": "preprocessed",
            "type": "embedding"
          }
        }
      ]
    }
  },
  "id": 6
}
```

### Metadata Filtering

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_vectors",
    "arguments": {
      "collection": "knowledge_base",
      "query": "machine learning",
      "limit": 20,
      "filters": {
        "category": "technical",
        "difficulty": "intermediate",
        "language": "en"
      },
      "includeMetadata": true
    }
  },
  "id": 7
}
```

## Use Cases

### 1. LLM Memory System

Store conversation context and retrieve relevant memories:

```python
# Store conversation turn
client.call_tool("add_vectors", {
    "collection": "conversations",
    "vectors": [{
        "content": user_message,
        "metadata": {
            "role": "user",
            "timestamp": timestamp,
            "session_id": session_id,
            "turn": turn_number
        }
    }]
})

# Retrieve relevant context
context = client.call_tool("search_vectors", {
    "collection": "conversations",
    "query": current_query,
    "limit": 10,
    "filters": {"session_id": session_id},
    "userId": user_id
})
```

### 2. RAG (Retrieval-Augmented Generation)

Index documents and retrieve relevant chunks:

```python
# Index document chunks
chunks = split_document(document_text)
vectors = []
for i, chunk in enumerate(chunks):
    vectors.append({
        "id": f"{doc_id}_chunk_{i}",
        "content": chunk,
        "metadata": {
            "document_id": doc_id,
            "chunk_index": i,
            "source": document_name,
            "created_at": timestamp
        }
    })

client.call_tool("add_vectors", {
    "collection": "documents",
    "vectors": vectors
})

# Retrieve relevant chunks for generation
relevant_chunks = client.call_tool("search_vectors", {
    "collection": "documents",
    "query": user_question,
    "limit": 5,
    "includeMetadata": true
})
```

### 3. Semantic Search Engine

Build a semantic search system:

```python
# Index items with rich metadata
client.call_tool("add_vectors", {
    "collection": "products",
    "vectors": [{
        "content": product_description,
        "metadata": {
            "product_id": product_id,
            "title": title,
            "category": category,
            "price": price,
            "rating": rating,
            "tags": tags
        }
    }]
})

# Search with filters
results = client.call_tool("search_vectors", {
    "collection": "products",
    "query": "comfortable running shoes",
    "limit": 20,
    "filters": {
        "category": "footwear",
        "price": {"$lt": 150},
        "rating": {"$gte": 4.0}
    }
})
```

## Best Practices

1. **Collection Design**
   - Use separate collections for different data types
   - Choose appropriate dimensions based on your embedding model
   - Select the right distance metric for your use case

2. **Vector Management**
   - Always include meaningful IDs for vectors
   - Use rich metadata for filtering and context
   - Batch vector additions for better performance

3. **Search Optimization**
   - Use appropriate limit values (10-50 for most cases)
   - Apply metadata filters to narrow search space
   - Enable personalization for user-specific results

4. **Error Handling**
   - Check for `isError` in responses
   - Validate collection existence before operations
   - Handle dimension mismatches gracefully

## Performance Tips

1. **Batch Operations**: Group multiple vectors in single `add_vectors` calls
2. **Index Type**: Use HNSW for large collections, flat for small ones (<10k vectors)
3. **Persistence**: Use BoltDB or BadgerDB for production deployments
4. **Caching**: Implement client-side caching for frequently accessed vectors

## Troubleshooting

### Common Issues

1. **"Collection not found"**
   - Ensure collection is created before adding/searching vectors
   - Check collection name spelling

2. **"Dimension mismatch"**
   - Verify vector dimensions match collection configuration
   - Check embedding model output dimensions

3. **"Parse error"**
   - Ensure valid JSON formatting
   - Check for proper escaping of special characters

4. **Performance Issues**
   - Consider using batch operations
   - Switch to persistent storage backend
   - Optimize search limit and filters