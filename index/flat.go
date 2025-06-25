package index

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	
	"github.com/dshills/EmbeddixDB/core"
)

// FlatIndex implements brute-force exact search
type FlatIndex struct {
	mu             sync.RWMutex
	vectors        map[string]core.Vector
	dimension      int
	distanceMetric core.DistanceMetric
}

// NewFlatIndex creates a new flat index
func NewFlatIndex(dimension int, distanceMetric core.DistanceMetric) *FlatIndex {
	return &FlatIndex{
		vectors:        make(map[string]core.Vector),
		dimension:      dimension,
		distanceMetric: distanceMetric,
	}
}

// Add adds a vector to the index
func (f *FlatIndex) Add(vector core.Vector) error {
	if err := core.ValidateVector(vector); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}
	
	if err := core.ValidateVectorDimension(vector, f.dimension); err != nil {
		return fmt.Errorf("dimension mismatch: %w", err)
	}
	
	f.mu.Lock()
	defer f.mu.Unlock()
	
	f.vectors[vector.ID] = vector
	return nil
}// Search performs brute-force search for k nearest neighbors
func (f *FlatIndex) Search(query []float32, k int, filter map[string]string) ([]core.SearchResult, error) {
	if len(query) != f.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", 
			len(query), f.dimension)
	}
	
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	var results []core.SearchResult
	
	// Calculate distances for all vectors
	for _, vector := range f.vectors {
		// Apply metadata filter if specified
		if !matchesFilter(vector.Metadata, filter) {
			continue
		}
		
		distance, err := core.CalculateDistance(query, vector.Values, f.distanceMetric)
		if err != nil {
			return nil, fmt.Errorf("distance calculation failed: %w", err)
		}
		
		result := core.SearchResult{
			ID:       vector.ID,
			Score:    distance,
			Metadata: vector.Metadata,
		}
		
		results = append(results, result)
	}
	
	// Sort by distance (ascending for distance metrics)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})
	
	// Return top k results
	if k > len(results) {
		k = len(results)
	}
	
	return results[:k], nil
}// Delete removes a vector from the index
func (f *FlatIndex) Delete(id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	
	if _, exists := f.vectors[id]; !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}
	
	delete(f.vectors, id)
	return nil
}

// RangeSearch finds all vectors within a distance threshold
func (f *FlatIndex) RangeSearch(query []float32, radius float32, filter map[string]string, limit int) ([]core.SearchResult, error) {
	if len(query) != f.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), f.dimension)
	}

	f.mu.RLock()
	defer f.mu.RUnlock()

	results := make([]core.SearchResult, 0)
	
	// Check all vectors
	for id, vec := range f.vectors {
		// Apply metadata filter if provided
		if filter != nil && len(filter) > 0 {
			match := true
			for key, value := range filter {
				if vec.Metadata[key] != value {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}

		// Calculate distance
		distance, err := core.CalculateDistance(query, vec.Values, f.distanceMetric)
		if err != nil {
			continue
		}

		// Check if within radius
		if distance <= radius {
			results = append(results, core.SearchResult{
				ID:       id,
				Score:    distance,
				Vector:   &vec,
				Metadata: vec.Metadata,
			})

			// Check limit
			if limit > 0 && len(results) >= limit {
				break
			}
		}
	}

	// Sort results by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	return results, nil
}

// Rebuild is a no-op for flat index (no structure to rebuild)
func (f *FlatIndex) Rebuild() error {
	return nil
}

// Size returns the number of vectors in the index
func (f *FlatIndex) Size() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.vectors)
}

// Type returns the index type
func (f *FlatIndex) Type() string {
	return "flat"
}

// flatIndexState represents the serializable state of a FlatIndex
type flatIndexState struct {
	Vectors        map[string]core.Vector   `json:"vectors"`
	Dimension      int                      `json:"dimension"`
	DistanceMetric core.DistanceMetric      `json:"distance_metric"`
}

// Serialize converts the index state to bytes
func (f *FlatIndex) Serialize() ([]byte, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	// Create a deep copy of vectors to avoid concurrent modification during JSON marshaling
	vectorsCopy := make(map[string]core.Vector, len(f.vectors))
	for id, vec := range f.vectors {
		vectorsCopy[id] = vec
	}
	
	state := flatIndexState{
		Vectors:        vectorsCopy,
		Dimension:      f.dimension,
		DistanceMetric: f.distanceMetric,
	}
	
	return json.Marshal(state)
}

// Deserialize restores the index state from bytes
func (f *FlatIndex) Deserialize(data []byte) error {
	var state flatIndexState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to unmarshal flat index state: %w", err)
	}
	
	f.mu.Lock()
	defer f.mu.Unlock()
	
	f.vectors = state.Vectors
	f.dimension = state.Dimension
	f.distanceMetric = state.DistanceMetric
	
	return nil
}

// matchesFilter checks if vector metadata matches the given filter
func matchesFilter(metadata, filter map[string]string) bool {
	if len(filter) == 0 {
		return true
	}
	
	for key, value := range filter {
		if metaValue, exists := metadata[key]; !exists || metaValue != value {
			return false
		}
	}
	
	return true
}