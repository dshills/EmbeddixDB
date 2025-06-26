package persistence

import (
	"context"
	"os"
	"testing"
)

func TestMemoryPersistenceIndexState(t *testing.T) {
	ctx := context.Background()
	persistence := NewMemoryPersistence()
	defer persistence.Close()

	// Test data
	collection := "test_collection"
	indexData := []byte(`{"nodes":{"v1":{"id":"v1","vector":{"id":"v1","values":[1,0,0]},"level":0,"connections":{}}},"entry_point_id":"v1","config":{"m":16,"ef_construction":200,"ef_search":100,"max_level":4,"seed":42},"dimension":3,"distance_metric":"cosine","size":1}`)

	// Save index state
	err := persistence.SaveIndexState(ctx, collection, indexData)
	if err != nil {
		t.Fatalf("Failed to save index state: %v", err)
	}

	// Load index state
	loadedData, err := persistence.LoadIndexState(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to load index state: %v", err)
	}

	// Verify data matches
	if string(loadedData) != string(indexData) {
		t.Error("Loaded index state does not match saved data")
	}

	// Delete index state
	err = persistence.DeleteIndexState(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to delete index state: %v", err)
	}

	// Verify deletion
	_, err = persistence.LoadIndexState(ctx, collection)
	if err == nil {
		t.Error("Expected error when loading deleted index state")
	}
}

func TestBoltPersistenceIndexState(t *testing.T) {
	ctx := context.Background()

	// Create temp database
	dbPath := "/tmp/test_bolt_index_state.db"
	defer os.RemoveAll(dbPath)

	persistence, err := NewBoltPersistence(dbPath)
	if err != nil {
		t.Fatalf("Failed to create BoltDB persistence: %v", err)
	}
	defer persistence.Close()

	// Test data
	collection := "test_collection"
	indexData := []byte(`{"nodes":{"v1":{"id":"v1","vector":{"id":"v1","values":[1,0,0]},"level":0,"connections":{}}},"entry_point_id":"v1","config":{"m":16,"ef_construction":200,"ef_search":100,"max_level":4,"seed":42},"dimension":3,"distance_metric":"cosine","size":1}`)

	// Save index state
	err = persistence.SaveIndexState(ctx, collection, indexData)
	if err != nil {
		t.Fatalf("Failed to save index state: %v", err)
	}

	// Load index state
	loadedData, err := persistence.LoadIndexState(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to load index state: %v", err)
	}

	// Verify data matches
	if string(loadedData) != string(indexData) {
		t.Error("Loaded index state does not match saved data")
	}

	// Delete index state
	err = persistence.DeleteIndexState(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to delete index state: %v", err)
	}

	// Verify deletion
	_, err = persistence.LoadIndexState(ctx, collection)
	if err == nil {
		t.Error("Expected error when loading deleted index state")
	}
}

func TestBadgerPersistenceIndexState(t *testing.T) {
	ctx := context.Background()

	// Create temp database
	dbPath := "/tmp/test_badger_index_state"
	defer os.RemoveAll(dbPath)

	persistence, err := NewBadgerPersistence(dbPath)
	if err != nil {
		t.Fatalf("Failed to create BadgerDB persistence: %v", err)
	}
	defer persistence.Close()

	// Test data
	collection := "test_collection"
	indexData := []byte(`{"nodes":{"v1":{"id":"v1","vector":{"id":"v1","values":[1,0,0]},"level":0,"connections":{}}},"entry_point_id":"v1","config":{"m":16,"ef_construction":200,"ef_search":100,"max_level":4,"seed":42},"dimension":3,"distance_metric":"cosine","size":1}`)

	// Save index state
	err = persistence.SaveIndexState(ctx, collection, indexData)
	if err != nil {
		t.Fatalf("Failed to save index state: %v", err)
	}

	// Load index state
	loadedData, err := persistence.LoadIndexState(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to load index state: %v", err)
	}

	// Verify data matches
	if string(loadedData) != string(indexData) {
		t.Error("Loaded index state does not match saved data")
	}

	// Delete index state
	err = persistence.DeleteIndexState(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to delete index state: %v", err)
	}

	// Verify deletion
	_, err = persistence.LoadIndexState(ctx, collection)
	if err == nil {
		t.Error("Expected error when loading deleted index state")
	}
}
