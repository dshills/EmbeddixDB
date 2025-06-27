# EmbeddixDB MCP Server API Documentation

## Overview

The EmbeddixDB MCP (Model Context Protocol) server provides a standardized interface for LLMs to interact with the vector database. It exposes vector operations as tools that can be called through the MCP protocol.

## Getting Started

### Building the MCP Server

```bash
go build ./cmd/mcp-server/
```

### Running the MCP Server

```bash
# Basic usage (in-memory storage)
./mcp-server

# With BoltDB persistence
./mcp-server -persistence bolt -data ./data

# With BadgerDB persistence
./mcp-server -persistence badger -data ./data

# Enable verbose logging
./mcp-server -verbose
```

### Command Line Options

- `-persistence`: Storage backend type (`memory`, `bolt`, `badger`). Default: `memory`
- `-data`: Data directory path for persistent backends. Default: `./data`
- `-verbose`: Enable verbose logging to stderr. Default: `false`

## MCP Protocol

The server communicates via JSON-RPC 2.0 over stdio (stdin/stdout). All requests and responses follow the MCP specification.

### Initialization

```json
// Request
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

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {
        "listChanged": false
      }
    },
    "serverInfo": {
      "name": "EmbeddixDB MCP Server",
      "version": "0.1.0"
    }
  },
  "id": 1
}
```

### List Available Tools

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 2
}

// Response contains all available tools with their schemas
```

## Available Tools

### 1. create_collection

Create a new vector collection.

**Parameters:**
- `name` (string, required): Name of the collection
- `dimension` (integer, required): Vector dimension
- `distance` (string, optional): Distance metric (`cosine`, `l2`, `dot`). Default: `cosine`
- `indexType` (string, optional): Index type (`flat`, `hnsw`). Default: `hnsw`

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "create_collection",
    "arguments": {
      "name": "documents",
      "dimension": 384,
      "distance": "cosine",
      "indexType": "hnsw"
    }
  },
  "id": 3
}
```

### 2. add_vectors

Add vectors to a collection. Supports both raw vectors and text content (requires embedding model).

**Parameters:**
- `collection` (string, required): Collection name
- `vectors` (array, required): Array of vector objects
  - `id` (string, optional): Vector ID (auto-generated if not provided)
  - `content` (string, optional): Text to embed
  - `vector` (array, optional): Raw vector data
  - `metadata` (object, optional): Additional metadata

**Example:**
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
          "id": "doc1",
          "vector": [0.1, 0.2, 0.3, ...],
          "metadata": {
            "title": "Introduction to MCP",
            "type": "documentation"
          }
        },
        {
          "content": "This is a text that will be embedded",
          "metadata": {
            "source": "user_input"
          }
        }
      ]
    }
  },
  "id": 4
}
```

### 3. search_vectors

Search for similar vectors using semantic similarity.

**Parameters:**
- `collection` (string, required): Collection to search
- `query` (string, optional): Text query for semantic search
- `vector` (array, optional): Raw vector for similarity search
- `limit` (integer, optional): Max results. Default: 10
- `filters` (object, optional): Metadata filters
- `includeMetadata` (boolean, optional): Include metadata. Default: true
- `sessionId` (string, optional): Session ID for personalization
- `userId` (string, optional): User ID for personalization

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_vectors",
    "arguments": {
      "collection": "documents",
      "query": "What is MCP?",
      "limit": 5,
      "includeMetadata": true,
      "userId": "user123"
    }
  },
  "id": 5
}
```

### 4. get_vector

Retrieve a specific vector by ID.

**Parameters:**
- `collection` (string, required): Collection name
- `id` (string, required): Vector ID

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_vector",
    "arguments": {
      "collection": "documents",
      "id": "doc1"
    }
  },
  "id": 6
}
```

### 5. delete_vector

Delete a vector from a collection.

**Parameters:**
- `collection` (string, required): Collection name
- `id` (string, required): Vector ID

### 6. list_collections

List all available collections.

**Parameters:** None

### 7. delete_collection

Delete a collection and all its vectors.

**Parameters:**
- `name` (string, required): Collection name

## Response Format

All tool responses follow this format:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Human-readable summary"
      },
      {
        "type": "text",
        "text": "JSON data with detailed results"
      }
    ],
    "isError": false
  },
  "id": <request_id>
}
```

Error responses:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Error description"
      }
    ],
    "isError": true
  },
  "id": <request_id>
}
```

## Integration Example

### Python Client Example

```python
import json
import subprocess

class MCPClient:
    def __init__(self, server_path="./mcp-server"):
        self.process = subprocess.Popen(
            [server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.request_id = 0
        
        # Initialize connection
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "python-client", "version": "1.0"}
        })
    
    def _call(self, method, params=None):
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id
        }
        if params:
            request["params"] = params
        
        # Send request
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        
        # Read response
        response = json.loads(self.process.stdout.readline())
        return response.get("result", response.get("error"))
    
    def call_tool(self, name, arguments):
        return self._call("tools/call", {
            "name": name,
            "arguments": arguments
        })

# Usage
client = MCPClient()

# Create collection
client.call_tool("create_collection", {
    "name": "memories",
    "dimension": 384
})

# Add vectors
client.call_tool("add_vectors", {
    "collection": "memories",
    "vectors": [{
        "content": "The user prefers dark themes",
        "metadata": {"type": "preference"}
    }]
})

# Search
results = client.call_tool("search_vectors", {
    "collection": "memories",
    "query": "What are the user's preferences?",
    "limit": 5
})
```

## Performance Considerations

1. **Batch Operations**: Use the `add_vectors` tool with multiple vectors for better performance
2. **Connection Pooling**: The MCP server maintains a single connection to the vector store
3. **Memory Usage**: Monitor memory usage when using in-memory storage backend
4. **Persistence**: Use BoltDB or BadgerDB for production deployments

## Error Handling

The MCP server provides detailed error messages for:
- Invalid parameters
- Missing collections
- Dimension mismatches
- Storage backend errors
- Embedding failures (when using text content)

## Security Considerations

1. The MCP server uses stdio communication - ensure proper process isolation
2. No built-in authentication - implement at the transport layer if needed
3. Input validation is performed on all tool parameters
4. Consider rate limiting for production deployments

## Next Steps

- See [MCP_EXAMPLES.md](./MCP_EXAMPLES.md) for more usage examples
- Review the [MCP Implementation Plan](./MCP_IMPLEMENTATION_PLAN.md) for upcoming features
- Check the [main documentation](../README.md) for general EmbeddixDB information