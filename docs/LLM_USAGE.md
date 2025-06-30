# Best Practices for Instructing LLMs to Use EmbeddixDB via MCP

## 1. Tool Discovery and Initialization

The LLM should first list available MCP tools to understand EmbeddixDB's capabilities:

- **search_vectors**: Semantic similarity search
- **add_vectors**: Store vectors/embeddings
- **get_vector**: Retrieve specific vectors
- **delete_vector**: Remove vectors
- **create_collection**: Create vector collections
- **list_collections**: View all collections
- **delete_collection**: Remove collections

## 2. Collection-First Approach

Always start by creating or verifying collections exist:

1. Use `list_collections` to see what's available
2. Create collections with appropriate dimensions (e.g., 384 for nomic-embed-text, 1024 for mxbai-embed-large)
3. Specify distance metric: `cosine` (default), `l2`, or `dot`
4. Choose index type: `hnsw` (fast approximate) or `flat` (exact but slower)

## 3. Auto-Embedding Feature

EmbeddixDB automatically converts text to vectors when using the `content` field:

```json
{
  "collection": "memories",
  "vectors": [{
    "content": "Text to be automatically embedded",
    "metadata": {"type": "conversation", "timestamp": "2024-01-01"}
  }]
}
```

## 4. Structured Metadata Usage

Encourage rich metadata for better filtering and retrieval:

```json
{
  "metadata": {
    "source": "conversation",
    "topic": "technical",
    "timestamp": "2024-01-01T10:00:00Z",
    "user_id": "user123",
    "session_id": "session456",
    "importance": "high",
    "tags": ["programming", "database"]
  }
}
```

## 5. Search Strategies

### Text-based search (recommended):

```json
{
  "collection": "memories",
  "query": "vector database implementation details",
  "limit": 10,
  "filters": {"topic": "technical"},
  "userId": "user123",
  "sessionId": "session456"
}
```

### Vector-based search (when you have embeddings):

```json
{
  "collection": "memories",
  "vector": [0.1, 0.2, ...],
  "limit": 5,
  "includeMetadata": true
}
```

## 6. Memory Management Patterns

### Conversation Memory:

1. Create a "conversations" collection
2. Store each message with metadata (role, timestamp, session)
3. Search by `session_id` for context retrieval
4. Use `userId` for personalization

### Knowledge Base:

1. Create domain-specific collections ("technical_docs", "user_preferences")
2. Add vectors with detailed metadata
3. Use filters for precise retrieval
4. Implement feedback with `sessionId` tracking

## 7. Example Instruction Template for LLMs

```markdown
You have access to EmbeddixDB, a vector database for persistent memory storage.

To use it effectively:

1. First, check existing collections with list_collections
2. Create collections for different memory types:
   - "conversations" for chat history
   - "knowledge" for learned information
   - "user_preferences" for personalization

3. When storing memories:
   - Use the "content" field for automatic text embedding
   - Add rich metadata for better retrieval
   - Include timestamps, categories, and relevance scores

4. When searching:
   - Use natural language queries with the "query" field
   - Apply filters to narrow results
   - Include userId/sessionId for personalized results
   - Request 5-10 results for context

5. Organize memories hierarchically:
   - Use metadata fields like "category", "subcategory"
   - Tag important memories with "importance": "high"
   - Track memory sources and confidence levels

Example memory storage:
```

```json
{
  "collection": "conversations",
  "vectors": [{
    "content": "User prefers Python for data science projects",
    "metadata": {
      "type": "preference",
      "domain": "programming",
      "confidence": 0.9,
      "timestamp": "2024-01-01T10:00:00Z",
      "user_id": "user123"
    }
  }]
}
```

## 8. Performance Optimization Tips

- **Batch operations**: Add multiple vectors in one call
- **Use appropriate collection sizes**: Dimension matching your embedding model
- **Choose HNSW index**: For large collections (>10k vectors)
- **Set reasonable search limits**: 5-20 results
- **Use metadata filters**: To reduce search space

## 9. Error Handling Guidance

Instruct the LLM to:

- Check if collections exist before operations
- Handle dimension mismatches gracefully
- Verify vector additions were successful
- Implement retry logic for transient failures
- Log operations for debugging

This approach ensures the LLM uses EmbeddixDB effectively as a memory backend while leveraging its advanced features like auto-embedding, personalized search, and metadata filtering.