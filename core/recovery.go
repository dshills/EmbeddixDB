package core

import (
	"context"
	"fmt"
)

// RecoverIndexes rebuilds all indexes from persisted data
func (vs *VectorStoreImpl) RecoverIndexes(ctx context.Context) error {
	// Get all collections
	collections, err := vs.persistence.LoadCollections(ctx)
	if err != nil {
		return fmt.Errorf("failed to load collections for recovery: %w", err)
	}
	
	// Rebuild index for each collection
	for _, collection := range collections {
		if err := vs.rebuildCollectionIndex(ctx, collection); err != nil {
			return fmt.Errorf("failed to rebuild index for collection %s: %w", collection.Name, err)
		}
	}
	
	return nil
}

// rebuildCollectionIndex rebuilds the index for a specific collection
func (vs *VectorStoreImpl) rebuildCollectionIndex(ctx context.Context, collection Collection) error {
	// Create index for the collection
	if err := vs.createIndex(collection); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}
	
	vs.mu.RLock()
	index, exists := vs.indexes[collection.Name]
	vs.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("index not found after creation for collection %s", collection.Name)
	}
	
	// Try to load serialized index state first
	if indexData, err := vs.persistence.LoadIndexState(ctx, collection.Name); err == nil {
		// Deserialize index state
		if err := index.Deserialize(indexData); err != nil {
			// If deserialization fails, fall back to rebuilding from vectors
			return vs.rebuildFromVectors(ctx, collection, index)
		}
		// Successfully loaded from serialized state
		return nil
	}
	
	// No serialized state found, rebuild from vectors
	return vs.rebuildFromVectors(ctx, collection, index)
}

// rebuildFromVectors rebuilds an index by adding all vectors from persistence
func (vs *VectorStoreImpl) rebuildFromVectors(ctx context.Context, collection Collection, index Index) error {
	// Load all vectors for this collection
	vectors, err := vs.persistence.LoadVectors(ctx, collection.Name)
	if err != nil {
		return fmt.Errorf("failed to load vectors: %w", err)
	}
	
	// Add all vectors to the index
	for _, vector := range vectors {
		if err := index.Add(vector); err != nil {
			return fmt.Errorf("failed to add vector %s to index: %w", vector.ID, err)
		}
	}
	
	return nil
}

// NewVectorStoreWithRecovery creates a vector store and automatically recovers indexes
func NewVectorStoreWithRecovery(persistence Persistence, indexFactory IndexFactory) (*VectorStoreImpl, error) {
	store := NewVectorStore(persistence, indexFactory)
	
	// Attempt to recover indexes from persisted data
	if err := store.RecoverIndexes(context.Background()); err != nil {
		// If recovery fails, we still return the store but log the error
		// This allows the store to work for new data even if old data can't be recovered
		return store, fmt.Errorf("index recovery failed (store still usable for new data): %w", err)
	}
	
	return store, nil
}