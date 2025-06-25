package core

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// VectorStoreImpl implements the VectorStore interface
type VectorStoreImpl struct {
	mu           sync.RWMutex
	persistence  Persistence
	indexes      map[string]Index // collection name -> index
	indexFactory IndexFactory
}



// NewVectorStore creates a new vector store with the given persistence layer and index factory
func NewVectorStore(persistence Persistence, indexFactory IndexFactory) *VectorStoreImpl {
	return &VectorStoreImpl{
		persistence:  persistence,
		indexes:      make(map[string]Index),
		indexFactory: indexFactory,
	}
}

// CreateCollection creates a new collection with the specified configuration
func (vs *VectorStoreImpl) CreateCollection(ctx context.Context, spec Collection) error {
	if err := ValidateCollection(spec); err != nil {
		return fmt.Errorf("invalid collection specification: %w", err)
	}
	
	// Set creation time if not provided
	if spec.CreatedAt.IsZero() {
		spec.CreatedAt = time.Now()
	}
	
	// Check if collection already exists
	_, err := vs.persistence.LoadCollection(ctx, spec.Name)
	if err == nil {
		return fmt.Errorf("collection %s already exists", spec.Name)
	}
	
	// Save collection metadata
	if err := vs.persistence.SaveCollection(ctx, spec); err != nil {
		return fmt.Errorf("failed to save collection: %w", err)
	}
	
	// Create index for the collection
	if err := vs.createIndex(spec); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}
	
	return nil
}// AddVector adds a vector to the specified collection
func (vs *VectorStoreImpl) AddVector(ctx context.Context, collection string, vec Vector) error {
	if err := ValidateVector(vec); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}
	
	// Get collection to validate dimension
	coll, err := vs.persistence.LoadCollection(ctx, collection)
	if err != nil {
		return fmt.Errorf("collection %s not found: %w", collection, err)
	}
	
	if err := ValidateVectorDimension(vec, coll.Dimension); err != nil {
		return fmt.Errorf("dimension mismatch: %w", err)
	}
	
	// Save to persistence layer
	if err := vs.persistence.SaveVector(ctx, collection, vec); err != nil {
		return fmt.Errorf("failed to save vector: %w", err)
	}
	
	// Add to index
	vs.mu.RLock()
	index, exists := vs.indexes[collection]
	vs.mu.RUnlock()
	
	if exists {
		if err := index.Add(vec); err != nil {
			return fmt.Errorf("failed to add to index: %w", err)
		}
		
		// Save updated index state (best effort - don't fail the operation if this fails)
		if err := vs.saveIndexState(ctx, collection, index); err != nil {
			// Note: We don't return this error because the vector was successfully added
			// and the index was updated. The state can be recovered later if needed.
			// In a production system, this should be logged
			_ = err // Acknowledge we're intentionally ignoring this error
		}
	}
	
	return nil
}

// GetVector retrieves a vector by ID from the specified collection
func (vs *VectorStoreImpl) GetVector(ctx context.Context, collection, id string) (Vector, error) {
	return vs.persistence.LoadVector(ctx, collection, id)
}// DeleteVector removes a vector from the specified collection
func (vs *VectorStoreImpl) DeleteVector(ctx context.Context, collection, id string) error {
	// Remove from persistence layer
	if err := vs.persistence.DeleteVector(ctx, collection, id); err != nil {
		return fmt.Errorf("failed to delete vector: %w", err)
	}
	
	// Remove from index
	vs.mu.RLock()
	index, exists := vs.indexes[collection]
	vs.mu.RUnlock()
	
	if exists {
		if err := index.Delete(id); err != nil {
			return fmt.Errorf("failed to delete from index: %w", err)
		}
		
		// Save updated index state (best effort - don't fail the operation if this fails)
		if err := vs.saveIndexState(ctx, collection, index); err != nil {
			// Note: We don't return this error because the vector was successfully deleted
			// and the index was updated. The state can be recovered later if needed.
			// In a production system, this should be logged
			_ = err // Acknowledge we're intentionally ignoring this error
		}
	}
	
	return nil
}

// Search performs vector similarity search in the specified collection
func (vs *VectorStoreImpl) Search(ctx context.Context, collection string, req SearchRequest) ([]SearchResult, error) {
	// Get collection to validate search request
	coll, err := vs.persistence.LoadCollection(ctx, collection)
	if err != nil {
		return nil, fmt.Errorf("collection %s not found: %w", collection, err)
	}
	
	if err := ValidateSearchRequest(req, coll.Dimension); err != nil {
		return nil, fmt.Errorf("invalid search request: %w", err)
	}
	
	// Use index for search
	vs.mu.RLock()
	index, exists := vs.indexes[collection]
	vs.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("index not found for collection %s", collection)
	}
	
	results, err := index.Search(req.Query, req.TopK, req.Filter)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	
	// Include full vectors if requested
	if req.IncludeVectors {
		for i := range results {
			vector, err := vs.persistence.LoadVector(ctx, collection, results[i].ID)
			if err != nil {
				continue // Skip if vector not found
			}
			results[i].Vector = &vector
		}
	}
	
	return results, nil
}// DeleteCollection removes a collection and all its vectors
func (vs *VectorStoreImpl) DeleteCollection(ctx context.Context, name string) error {
	// Remove index
	vs.mu.Lock()
	delete(vs.indexes, name)
	vs.mu.Unlock()
	
	// Remove from persistence
	if err := vs.persistence.DeleteCollection(ctx, name); err != nil {
		return fmt.Errorf("failed to delete collection: %w", err)
	}
	
	return nil
}

// ListCollections returns all collections
func (vs *VectorStoreImpl) ListCollections(ctx context.Context) ([]Collection, error) {
	return vs.persistence.LoadCollections(ctx)
}

// GetCollection returns collection metadata
func (vs *VectorStoreImpl) GetCollection(ctx context.Context, name string) (Collection, error) {
	return vs.persistence.LoadCollection(ctx, name)
}

// Close closes the vector store and its resources
func (vs *VectorStoreImpl) Close() error {
	return vs.persistence.Close()
}

// createIndex creates an index for a collection based on its configuration
func (vs *VectorStoreImpl) createIndex(collection Collection) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	
	distanceMetric := DistanceMetric(collection.Distance)
	
	index, err := vs.indexFactory.CreateIndex(collection.IndexType, collection.Dimension, distanceMetric)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}
	
	vs.indexes[collection.Name] = index
	return nil
}

// saveIndexState serializes and saves the current state of an index
func (vs *VectorStoreImpl) saveIndexState(ctx context.Context, collection string, index Index) error {
	// Serialize the index state
	indexData, err := index.Serialize()
	if err != nil {
		return fmt.Errorf("failed to serialize index state: %w", err)
	}
	
	// Save to persistence layer
	return vs.persistence.SaveIndexState(ctx, collection, indexData)
}