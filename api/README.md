# EmbeddixDB API Documentation

## Overview

EmbeddixDB provides a RESTful API for managing vector collections and performing similarity searches. The API is documented using OpenAPI/Swagger specification.

## Viewing API Documentation

When the server is running, you can view the interactive API documentation at:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI Spec**: http://localhost:8080/swagger.yaml

## Quick Start

### 1. Create a Collection

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_vectors",
    "dimension": 384,
    "index_type": "hnsw",
    "distance": "cosine"
  }'
```

### 2. Add Vectors

```bash
curl -X POST http://localhost:8080/collections/my_vectors/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "vec1",
    "values": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "category": "example"
    }
  }'
```

### 3. Search for Similar Vectors

```bash
curl -X POST http://localhost:8080/collections/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, ...],
    "top_k": 10,
    "filter": {
      "category": "example"
    }
  }'
```

### 4. Range Search

```bash
curl -X POST http://localhost:8080/collections/my_vectors/search/range \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, ...],
    "radius": 0.5,
    "limit": 100
  }'
```

## API Endpoints

### Collections
- `GET /collections` - List all collections
- `POST /collections` - Create a new collection
- `GET /collections/{name}` - Get collection details
- `DELETE /collections/{name}` - Delete a collection

### Vectors
- `POST /collections/{name}/vectors` - Add a single vector
- `POST /collections/{name}/vectors/batch` - Add multiple vectors
- `GET /collections/{name}/vectors/{id}` - Get a vector
- `PUT /collections/{name}/vectors/{id}` - Update a vector
- `DELETE /collections/{name}/vectors/{id}` - Delete a vector

### Search
- `POST /collections/{name}/search` - K-nearest neighbor search
- `POST /collections/{name}/search/batch` - Batch KNN search
- `POST /collections/{name}/search/range` - Range search

### Statistics
- `GET /stats` - Get overall statistics
- `GET /collections/{name}/stats` - Get collection statistics

### Health
- `GET /health` - Health check

## Response Formats

All responses are in JSON format. Successful responses include the requested data, while errors return:

```json
{
  "error": "Error message"
}
```

## Authentication

Currently, the API does not require authentication. In production, you should implement appropriate authentication and authorization mechanisms.

## Rate Limiting

No rate limiting is currently implemented. Consider adding rate limiting for production deployments.

## Error Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error