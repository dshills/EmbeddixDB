package persistence

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

func TestBoltPersistence(t *testing.T) {
	// Create temporary directory for test
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.bolt")

	// Create BoltDB persistence
	persistence, err := NewBoltPersistence(dbPath)
	if err != nil {
		t.Fatalf("Failed to create BoltDB persistence: %v", err)
	}
	defer persistence.Close()

	testPersistenceOperations(t, persistence)
}

func TestBadgerPersistence(t *testing.T) {
	// Create temporary directory for test
	tmpDir := t.TempDir()

	// Create BadgerDB persistence
	persistence, err := NewBadgerPersistence(tmpDir)
	if err != nil {
		t.Fatalf("Failed to create BadgerDB persistence: %v", err)
	}
	defer persistence.Close()

	testPersistenceOperations(t, persistence)
}

func TestMemoryPersistence(t *testing.T) {
	persistence := NewMemoryPersistence()
	defer persistence.Close()

	testPersistenceOperations(t, persistence)
}

// testPersistenceOperations runs a comprehensive test suite on any persistence implementation
func testPersistenceOperations(t *testing.T, persistence core.Persistence) {
	ctx := context.Background()

	// Test collection operations
	collection := core.Collection{
		Name:      "test_collection",
		Dimension: 3,
		IndexType: "flat",
		Distance:  "cosine",
		CreatedAt: time.Now(),
	}

	// Save collection
	err := persistence.SaveCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to save collection: %v", err)
	}

	// Load collection
	loadedCollection, err := persistence.LoadCollection(ctx, "test_collection")
	if err != nil {
		t.Fatalf("Failed to load collection: %v", err)
	}

	if loadedCollection.Name != collection.Name {
		t.Errorf("Collection name mismatch: expected %s, got %s", collection.Name, loadedCollection.Name)
	}

	// Test vector operations
	vectors := []core.Vector{
		{ID: "vec1", Values: []float32{1.0, 0.0, 0.0}, Metadata: map[string]string{"type": "test"}},
		{ID: "vec2", Values: []float32{0.0, 1.0, 0.0}, Metadata: map[string]string{"type": "test"}},
		{ID: "vec3", Values: []float32{0.0, 0.0, 1.0}, Metadata: map[string]string{"type": "different"}},
	}

	// Save vectors individually
	for _, vec := range vectors[:2] {
		err := persistence.SaveVector(ctx, "test_collection", vec)
		if err != nil {
			t.Fatalf("Failed to save vector %s: %v", vec.ID, err)
		}
	}

	// Save vector batch
	err = persistence.SaveVectorsBatch(ctx, "test_collection", vectors[2:])
	if err != nil {
		t.Fatalf("Failed to save vector batch: %v", err)
	}

	// Load individual vector
	loadedVector, err := persistence.LoadVector(ctx, "test_collection", "vec1")
	if err != nil {
		t.Fatalf("Failed to load vector: %v", err)
	}

	if loadedVector.ID != "vec1" {
		t.Errorf("Vector ID mismatch: expected vec1, got %s", loadedVector.ID)
	}

	// Load all vectors
	allVectors, err := persistence.LoadVectors(ctx, "test_collection")
	if err != nil {
		t.Fatalf("Failed to load all vectors: %v", err)
	}

	if len(allVectors) != 3 {
		t.Errorf("Expected 3 vectors, got %d", len(allVectors))
	}

	// Delete a vector
	err = persistence.DeleteVector(ctx, "test_collection", "vec2")
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	// Verify vector is deleted
	_, err = persistence.LoadVector(ctx, "test_collection", "vec2")
	if err == nil {
		t.Error("Expected error when loading deleted vector")
	}

	// List collections
	collections, err := persistence.LoadCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to load collections: %v", err)
	}

	if len(collections) != 1 {
		t.Errorf("Expected 1 collection, got %d", len(collections))
	}

	// Delete collection
	err = persistence.DeleteCollection(ctx, "test_collection")
	if err != nil {
		t.Fatalf("Failed to delete collection: %v", err)
	}

	// Verify collection is deleted
	_, err = persistence.LoadCollection(ctx, "test_collection")
	if err == nil {
		t.Error("Expected error when loading deleted collection")
	}
}
