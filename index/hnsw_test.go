package index

import (
	"fmt"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
)

func TestHNSWBasicOperations(t *testing.T) {
	config := DefaultHNSWConfig()
	config.M = 4 // Smaller M for test
	config.EfConstruction = 10
	config.EfSearch = 5

	index := NewHNSWIndex(3, core.DistanceCosine, config)

	// Test adding vectors
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{1.0, 0.0, 0.0}},
		{ID: "v2", Values: []float32{0.0, 1.0, 0.0}},
		{ID: "v3", Values: []float32{0.0, 0.0, 1.0}},
		{ID: "v4", Values: []float32{0.7, 0.7, 0.0}},
	}

	for _, vec := range vectors {
		err := index.Add(vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	if index.Size() != len(vectors) {
		t.Errorf("Expected size %d, got %d", len(vectors), index.Size())
	}

	// Test search
	query := []float32{1.0, 0.1, 0.0}
	results, err := index.Search(query, 2, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected search results, got none")
	}

	// First result should be v1 (closest to query)
	if results[0].ID != "v1" {
		t.Errorf("Expected first result to be v1, got %s", results[0].ID)
	}
}

func TestHNSWDelete(t *testing.T) {
	config := DefaultHNSWConfig()
	index := NewHNSWIndex(2, core.DistanceL2, config)

	// Add vectors
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{1.0, 0.0}},
		{ID: "v2", Values: []float32{0.0, 1.0}},
	}

	for _, vec := range vectors {
		index.Add(vec)
	}

	// Delete a vector
	err := index.Delete("v1")
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	if index.Size() != 1 {
		t.Errorf("Expected size 1 after deletion, got %d", index.Size())
	}

	// Try to delete non-existent vector
	err = index.Delete("nonexistent")
	if err == nil {
		t.Error("Expected error when deleting non-existent vector")
	}
}

func TestHNSWWithFilter(t *testing.T) {
	config := DefaultHNSWConfig()
	index := NewHNSWIndex(2, core.DistanceCosine, config)

	// Add vectors with metadata
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{1.0, 0.0}, Metadata: map[string]string{"type": "A"}},
		{ID: "v2", Values: []float32{0.9, 0.1}, Metadata: map[string]string{"type": "A"}},
		{ID: "v3", Values: []float32{0.0, 1.0}, Metadata: map[string]string{"type": "B"}},
	}

	for _, vec := range vectors {
		index.Add(vec)
	}

	// Search with filter
	query := []float32{1.0, 0.0}
	filter := map[string]string{"type": "A"}

	results, err := index.Search(query, 5, filter)
	if err != nil {
		t.Fatalf("Filtered search failed: %v", err)
	}

	// Should only return vectors with type "A"
	for _, result := range results {
		if result.Metadata["type"] != "A" {
			t.Errorf("Filter failed: got vector with type %s", result.Metadata["type"])
		}
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 filtered results, got %d", len(results))
	}
}

func TestHNSWValidation(t *testing.T) {
	config := DefaultHNSWConfig()
	index := NewHNSWIndex(2, core.DistanceCosine, config)

	// Test invalid vector (wrong dimension)
	vec := core.Vector{ID: "v1", Values: []float32{1.0, 0.0, 0.0}} // 3D instead of 2D
	err := index.Add(vec)
	if err == nil {
		t.Error("Expected error for wrong dimension vector")
	}

	// Test invalid search query
	query := []float32{1.0} // 1D instead of 2D
	_, err = index.Search(query, 1, nil)
	if err == nil {
		t.Error("Expected error for wrong dimension query")
	}
}

func BenchmarkHNSWAdd(b *testing.B) {
	config := DefaultHNSWConfig()
	index := NewHNSWIndex(128, core.DistanceCosine, config)

	// Pre-generate vectors
	vectors := make([]core.Vector, b.N)
	for i := 0; i < b.N; i++ {
		values := make([]float32, 128)
		for j := range values {
			values[j] = float32(i*j) * 0.001
		}
		vectors[i] = core.Vector{
			ID:     fmt.Sprintf("v%d", i),
			Values: values,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index.Add(vectors[i])
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	config := DefaultHNSWConfig()
	index := NewHNSWIndex(128, core.DistanceCosine, config)

	// Add 1000 vectors first
	for i := 0; i < 1000; i++ {
		values := make([]float32, 128)
		for j := range values {
			values[j] = float32(i*j) * 0.001
		}
		vec := core.Vector{
			ID:     fmt.Sprintf("v%d", i),
			Values: values,
		}
		index.Add(vec)
	}

	// Create query vector
	query := make([]float32, 128)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = index.Search(query, 10, nil)
	}
}
