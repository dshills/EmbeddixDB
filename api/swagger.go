// Package api EmbeddixDB API
//
// EmbeddixDB is a high-performance vector database for storing and searching embeddings.
// It supports multiple index types, distance metrics, and provides both KNN and range search capabilities.
//
//	Schemes: http, https
//	Host: localhost:8080
//	BasePath: /
//	Version: 1.0.0
//
//	Consumes:
//	- application/json
//
//	Produces:
//	- application/json
//
// swagger:meta
package api

// swagger:response healthResponse
type swaggerHealthResponse struct {
	// in: body
	Body HealthResponse
}

// swagger:response collectionResponse
type swaggerCollectionResponse struct {
	// in: body
	Body CollectionResponse
}

// swagger:response collectionsResponse
type swaggerCollectionsResponse struct {
	// in: body
	Body []CollectionResponse
}

// swagger:response vectorResponse
type swaggerVectorResponse struct {
	// in: body
	Body VectorResponse
}

// swagger:response searchResultsResponse
type swaggerSearchResultsResponse struct {
	// in: body
	Body []SearchResult
}

// swagger:response rangeSearchResponse
type swaggerRangeSearchResponse struct {
	// in: body
	Body RangeSearchResponse
}

// swagger:response statsResponse
type swaggerStatsResponse struct {
	// in: body
	Body StatsResponse
}

// swagger:response collectionStatsResponse
type swaggerCollectionStatsResponse struct {
	// in: body
	Body CollectionStats
}

// swagger:response errorResponse
type swaggerErrorResponse struct {
	// in: body
	Body struct {
		// Error message
		// required: true
		Error string `json:"error"`
	}
}

// swagger:response messageResponse
type swaggerMessageResponse struct {
	// in: body
	Body struct {
		// Success message
		// required: true
		Message string `json:"message"`
	}
}

// swagger:response batchAddResponse
type swaggerBatchAddResponse struct {
	// in: body
	Body struct {
		// Success message
		// required: true
		Message string `json:"message"`
		// Number of vectors added
		// required: true
		Count int `json:"count"`
	}
}

// swagger:parameters getCollection deleteCollection getVector updateVector deleteVector
// swagger:parameters search batchSearch rangeSearch addVector addVectorsBatch getCollectionStats
type collectionParam struct {
	// The name of the collection
	// in: path
	// required: true
	Collection string `json:"collection"`
}

// swagger:parameters getVector updateVector deleteVector
type vectorIDParam struct {
	// The ID of the vector
	// in: path
	// required: true
	ID string `json:"id"`
}

// swagger:parameters createCollection
type createCollectionParams struct {
	// Collection creation request
	// in: body
	// required: true
	Body CreateCollectionRequest
}

// swagger:parameters addVector
type addVectorParams struct {
	// Vector to add
	// in: body
	// required: true
	Body AddVectorRequest
}

// swagger:parameters updateVector
type updateVectorParams struct {
	// Vector update data
	// in: body
	// required: true
	Body AddVectorRequest
}

// swagger:parameters addVectorsBatch
type addVectorsBatchParams struct {
	// Vectors to add in batch
	// in: body
	// required: true
	Body []AddVectorRequest
}

// swagger:parameters search
type searchParams struct {
	// Search request
	// in: body
	// required: true
	Body SearchRequest
}

// swagger:parameters batchSearch
type batchSearchParams struct {
	// Batch search requests
	// in: body
	// required: true
	Body []SearchRequest
}

// swagger:parameters rangeSearch
type rangeSearchParams struct {
	// Range search request
	// in: body
	// required: true
	Body RangeSearchRequest
}

// Models for better documentation

// Collection represents a vector collection
// swagger:model
type Collection struct {
	// The unique name of the collection
	// required: true
	// example: product_embeddings
	Name string `json:"name"`
	
	// The dimension of vectors in this collection
	// required: true
	// minimum: 1
	// example: 384
	Dimension int `json:"dimension"`
	
	// The type of index to use
	// required: true
	// enum: flat,hnsw
	// example: hnsw
	IndexType string `json:"index_type"`
	
	// The distance metric to use
	// required: true
	// enum: l2,cosine,dot
	// example: cosine
	Distance string `json:"distance"`
	
	// Optional metadata for the collection
	// example: {"description": "Product embeddings from BERT model"}
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Vector represents a vector with metadata
// swagger:model
type Vector struct {
	// The unique identifier for the vector
	// required: true
	// example: product_123
	ID string `json:"id"`
	
	// The vector values
	// required: true
	// example: [0.1, 0.2, 0.3]
	Values []float32 `json:"values"`
	
	// Optional metadata for the vector
	// example: {"category": "electronics", "price": 99.99}
	Metadata map[string]string `json:"metadata,omitempty"`
}

// SearchParams represents search parameters
// swagger:model
type SearchParams struct {
	// The query vector
	// required: true
	// example: [0.1, 0.2, 0.3]
	Query []float32 `json:"query"`
	
	// The number of nearest neighbors to return
	// required: true
	// minimum: 1
	// maximum: 1000
	// example: 10
	TopK int `json:"top_k"`
	
	// Optional metadata filters
	// example: {"category": "electronics"}
	Filter map[string]string `json:"filter,omitempty"`
	
	// Whether to include the full vector values in results
	// example: false
	IncludeVectors bool `json:"include_vectors"`
}

// RangeSearchParams represents range search parameters
// swagger:model
type RangeSearchParams struct {
	// The query vector
	// required: true
	// example: [0.1, 0.2, 0.3]
	Query []float32 `json:"query"`
	
	// The maximum distance threshold
	// required: true
	// minimum: 0
	// example: 0.5
	Radius float32 `json:"radius"`
	
	// Optional metadata filters
	// example: {"category": "electronics"}
	Filter map[string]string `json:"filter,omitempty"`
	
	// Whether to include the full vector values in results
	// example: false
	IncludeVectors bool `json:"include_vectors"`
	
	// Optional limit on the number of results (0 = no limit)
	// minimum: 0
	// example: 100
	Limit int `json:"limit,omitempty"`
}