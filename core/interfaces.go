package core

import "context"

// IndexFactory creates index instances based on type and configuration
type IndexFactory interface {
	CreateIndex(indexType string, dimension int, distanceMetric DistanceMetric) (Index, error)
}

// VectorStore is the main interface for vector database operations
type VectorStore interface {
	// Vector operations
	AddVector(ctx context.Context, collection string, vec Vector) error
	GetVector(ctx context.Context, collection, id string) (Vector, error)
	DeleteVector(ctx context.Context, collection, id string) error
	Search(ctx context.Context, collection string, req SearchRequest) ([]SearchResult, error)
	
	// Collection operations
	CreateCollection(ctx context.Context, spec Collection) error
	DeleteCollection(ctx context.Context, name string) error
	ListCollections(ctx context.Context) ([]Collection, error)
	GetCollection(ctx context.Context, name string) (Collection, error)
	
	// Lifecycle
	Close() error
}

// Index represents a search index for vectors
type Index interface {
	// Add a vector to the index
	Add(vector Vector) error
	
	// Search for similar vectors
	Search(query []float32, k int, filter map[string]string) ([]SearchResult, error)
	
	// Remove a vector from the index
	Delete(id string) error
	
	// Rebuild the index (useful after batch operations)
	Rebuild() error
	
	// Get index statistics
	Size() int
	
	// Get index type
	Type() string
	
	// Serialize index state to bytes
	Serialize() ([]byte, error)
	
	// Deserialize index state from bytes
	Deserialize(data []byte) error
}// Persistence handles durable storage of vectors and collections
type Persistence interface {
	// Vector operations
	SaveVector(ctx context.Context, collection string, vec Vector) error
	LoadVector(ctx context.Context, collection, id string) (Vector, error)
	LoadVectors(ctx context.Context, collection string) ([]Vector, error)
	DeleteVector(ctx context.Context, collection, id string) error
	
	// Collection operations
	SaveCollection(ctx context.Context, collection Collection) error
	LoadCollection(ctx context.Context, name string) (Collection, error)
	LoadCollections(ctx context.Context) ([]Collection, error)
	DeleteCollection(ctx context.Context, name string) error
	
	// Index state operations
	SaveIndexState(ctx context.Context, collection string, indexData []byte) error
	LoadIndexState(ctx context.Context, collection string) ([]byte, error)
	DeleteIndexState(ctx context.Context, collection string) error
	
	// Batch operations for performance
	SaveVectorsBatch(ctx context.Context, collection string, vectors []Vector) error
	
	// Lifecycle
	Close() error
}