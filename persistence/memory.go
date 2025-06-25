package persistence

import (
	"context"
	"fmt"
	"sync"
	
	"github.com/dshills/EmbeddixDB/core"
)

// MemoryPersistence implements in-memory storage (non-persistent)
type MemoryPersistence struct {
	mu          sync.RWMutex
	vectors     map[string]map[string]core.Vector // collection -> id -> vector
	collections map[string]core.Collection        // collection name -> collection
	indexStates map[string][]byte                 // collection name -> serialized index state
}

// NewMemoryPersistence creates a new in-memory persistence layer
func NewMemoryPersistence() *MemoryPersistence {
	return &MemoryPersistence{
		vectors:     make(map[string]map[string]core.Vector),
		collections: make(map[string]core.Collection),
		indexStates: make(map[string][]byte),
	}
}

// SaveVector stores a vector in memory
func (m *MemoryPersistence) SaveVector(ctx context.Context, collection string, vec core.Vector) error {
	if err := core.ValidateVector(vec); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.vectors[collection] == nil {
		m.vectors[collection] = make(map[string]core.Vector)
	}
	
	m.vectors[collection][vec.ID] = vec
	return nil
}// LoadVector retrieves a vector by ID from memory
func (m *MemoryPersistence) LoadVector(ctx context.Context, collection, id string) (core.Vector, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	collectionVectors, exists := m.vectors[collection]
	if !exists {
		return core.Vector{}, fmt.Errorf("collection %s not found", collection)
	}
	
	vector, exists := collectionVectors[id]
	if !exists {
		return core.Vector{}, fmt.Errorf("vector %s not found in collection %s", id, collection)
	}
	
	return vector, nil
}

// LoadVectors retrieves all vectors from a collection
func (m *MemoryPersistence) LoadVectors(ctx context.Context, collection string) ([]core.Vector, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	collectionVectors, exists := m.vectors[collection]
	if !exists {
		return nil, fmt.Errorf("collection %s not found", collection)
	}
	
	vectors := make([]core.Vector, 0, len(collectionVectors))
	for _, vec := range collectionVectors {
		vectors = append(vectors, vec)
	}
	
	return vectors, nil
}

// DeleteVector removes a vector from memory
func (m *MemoryPersistence) DeleteVector(ctx context.Context, collection, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	collectionVectors, exists := m.vectors[collection]
	if !exists {
		return fmt.Errorf("collection %s not found", collection)
	}
	
	if _, exists := collectionVectors[id]; !exists {
		return fmt.Errorf("vector %s not found in collection %s", id, collection)
	}
	
	delete(collectionVectors, id)
	return nil
}// SaveCollection stores collection metadata
func (m *MemoryPersistence) SaveCollection(ctx context.Context, collection core.Collection) error {
	if err := core.ValidateCollection(collection); err != nil {
		return fmt.Errorf("invalid collection: %w", err)
	}
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.collections[collection.Name] = collection
	return nil
}

// LoadCollection retrieves collection metadata
func (m *MemoryPersistence) LoadCollection(ctx context.Context, name string) (core.Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	collection, exists := m.collections[name]
	if !exists {
		return core.Collection{}, fmt.Errorf("collection %s not found", name)
	}
	
	return collection, nil
}

// LoadCollections retrieves all collection metadata
func (m *MemoryPersistence) LoadCollections(ctx context.Context) ([]core.Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	collections := make([]core.Collection, 0, len(m.collections))
	for _, collection := range m.collections {
		collections = append(collections, collection)
	}
	
	return collections, nil
}// DeleteCollection removes a collection and all its vectors
func (m *MemoryPersistence) DeleteCollection(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if _, exists := m.collections[name]; !exists {
		return fmt.Errorf("collection %s not found", name)
	}
	
	delete(m.collections, name)
	delete(m.vectors, name)
	return nil
}

// SaveVectorsBatch stores multiple vectors efficiently
func (m *MemoryPersistence) SaveVectorsBatch(ctx context.Context, collection string, vectors []core.Vector) error {
	for _, vec := range vectors {
		if err := core.ValidateVector(vec); err != nil {
			return fmt.Errorf("invalid vector %s: %w", vec.ID, err)
		}
	}
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.vectors[collection] == nil {
		m.vectors[collection] = make(map[string]core.Vector)
	}
	
	for _, vec := range vectors {
		m.vectors[collection][vec.ID] = vec
	}
	
	return nil
}

// SaveIndexState stores serialized index state
func (m *MemoryPersistence) SaveIndexState(ctx context.Context, collection string, indexData []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.indexStates[collection] = indexData
	return nil
}

// LoadIndexState retrieves serialized index state
func (m *MemoryPersistence) LoadIndexState(ctx context.Context, collection string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	indexData, exists := m.indexStates[collection]
	if !exists {
		return nil, fmt.Errorf("index state for collection %s not found", collection)
	}
	
	return indexData, nil
}

// DeleteIndexState removes serialized index state
func (m *MemoryPersistence) DeleteIndexState(ctx context.Context, collection string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	delete(m.indexStates, collection)
	return nil
}

// Close is a no-op for memory persistence
func (m *MemoryPersistence) Close() error {
	return nil
}