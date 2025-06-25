package index

import (
	"testing"
	
	"github.com/dshills/EmbeddixDB/core"
)

func TestFlatIndexSerialization(t *testing.T) {
	// Create and populate flat index
	original := NewFlatIndex(3, core.DistanceCosine)
	
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{1.0, 0.0, 0.0}, Metadata: map[string]string{"type": "A"}},
		{ID: "v2", Values: []float32{0.0, 1.0, 0.0}, Metadata: map[string]string{"type": "B"}},
		{ID: "v3", Values: []float32{0.0, 0.0, 1.0}, Metadata: map[string]string{"type": "A"}},
	}
	
	for _, vec := range vectors {
		err := original.Add(vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}
	
	// Serialize
	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("Failed to serialize index: %v", err)
	}
	
	// Create new index and deserialize
	restored := NewFlatIndex(3, core.DistanceCosine)
	err = restored.Deserialize(data)
	if err != nil {
		t.Fatalf("Failed to deserialize index: %v", err)
	}
	
	// Verify the index works correctly
	if restored.Size() != original.Size() {
		t.Errorf("Size mismatch: original %d, restored %d", original.Size(), restored.Size())
	}
	
	// Test search functionality
	query := []float32{1.0, 0.1, 0.0}
	
	originalResults, err := original.Search(query, 2, nil)
	if err != nil {
		t.Fatalf("Original search failed: %v", err)
	}
	
	restoredResults, err := restored.Search(query, 2, nil)
	if err != nil {
		t.Fatalf("Restored search failed: %v", err)
	}
	
	if len(originalResults) != len(restoredResults) {
		t.Errorf("Result count mismatch: original %d, restored %d", 
			len(originalResults), len(restoredResults))
	}
	
	// Verify same results (order and content)
	for i := 0; i < len(originalResults) && i < len(restoredResults); i++ {
		if originalResults[i].ID != restoredResults[i].ID {
			t.Errorf("Result %d ID mismatch: original %s, restored %s", 
				i, originalResults[i].ID, restoredResults[i].ID)
		}
		if abs(originalResults[i].Score - restoredResults[i].Score) > 0.0001 {
			t.Errorf("Result %d score mismatch: original %f, restored %f", 
				i, originalResults[i].Score, restoredResults[i].Score)
		}
	}
}

func TestHNSWIndexSerialization(t *testing.T) {
	// Create and populate HNSW index with smaller parameters for testing
	config := DefaultHNSWConfig()
	config.M = 4
	config.EfConstruction = 10
	config.EfSearch = 5
	
	original := NewHNSWIndex(3, core.DistanceCosine, config)
	
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{1.0, 0.0, 0.0}, Metadata: map[string]string{"type": "A"}},
		{ID: "v2", Values: []float32{0.0, 1.0, 0.0}, Metadata: map[string]string{"type": "B"}},
		{ID: "v3", Values: []float32{0.0, 0.0, 1.0}, Metadata: map[string]string{"type": "A"}},
		{ID: "v4", Values: []float32{0.7, 0.7, 0.0}, Metadata: map[string]string{"type": "C"}},
	}
	
	for _, vec := range vectors {
		err := original.Add(vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}
	
	// Serialize
	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("Failed to serialize HNSW index: %v", err)
	}
	
	// Create new index and deserialize
	restored := NewHNSWIndex(3, core.DistanceCosine, config)
	err = restored.Deserialize(data)
	if err != nil {
		t.Fatalf("Failed to deserialize HNSW index: %v", err)
	}
	
	// Verify the index works correctly
	if restored.Size() != original.Size() {
		t.Errorf("Size mismatch: original %d, restored %d", original.Size(), restored.Size())
	}
	
	// Test search functionality
	query := []float32{1.0, 0.1, 0.0}
	
	originalResults, err := original.Search(query, 3, nil)
	if err != nil {
		t.Fatalf("Original HNSW search failed: %v", err)
	}
	
	restoredResults, err := restored.Search(query, 3, nil)
	if err != nil {
		t.Fatalf("Restored HNSW search failed: %v", err)
	}
	
	// For HNSW, we just verify that we get some results and basic functionality works
	// The exact order might differ due to the probabilistic nature of HNSW
	if len(originalResults) == 0 {
		t.Error("Original HNSW search returned no results")
	}
	
	if len(restoredResults) == 0 {
		t.Error("Restored HNSW search returned no results")
	}
	
	// Verify that the closest vector is found (should be v1 for this query)
	found := false
	for _, result := range restoredResults {
		if result.ID == "v1" {
			found = true
			break
		}
	}
	
	if !found {
		t.Error("Restored HNSW index did not find the expected closest vector (v1)")
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}