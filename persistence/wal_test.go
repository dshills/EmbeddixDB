package persistence

import (
	"context"
	"os"
	"testing"
	"time"
	
	"github.com/dshills/EmbeddixDB/core"
)

func TestWALBasicOperations(t *testing.T) {
	// Create temporary WAL directory
	walDir := "/tmp/test_wal_basic"
	defer os.RemoveAll(walDir)
	
	config := WALConfig{
		Path:     walDir,
		MaxSize:  1024 * 1024, // 1MB for testing
		SyncMode: true,
	}
	
	wal, err := NewWAL(config)
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()
	
	ctx := context.Background()
	
	// Test writing entries
	testData := []byte(`{"id":"test1","values":[1,2,3]}`)
	
	err = wal.WriteEntry(ctx, WALOpAddVector, "test_collection", "test1", testData)
	if err != nil {
		t.Fatalf("Failed to write WAL entry: %v", err)
	}
	
	err = wal.WriteEntry(ctx, WALOpDeleteVector, "test_collection", "test2", nil)
	if err != nil {
		t.Fatalf("Failed to write WAL entry: %v", err)
	}
	
	// Test reading entries
	entries, err := wal.ReadEntries(ctx, 1)
	if err != nil {
		t.Fatalf("Failed to read WAL entries: %v", err)
	}
	
	if len(entries) != 2 {
		t.Errorf("Expected 2 entries, got %d", len(entries))
	}
	
	// Verify first entry
	if entries[0].Operation != WALOpAddVector {
		t.Errorf("Expected operation %s, got %s", WALOpAddVector, entries[0].Operation)
	}
	
	if entries[0].Collection != "test_collection" {
		t.Errorf("Expected collection 'test_collection', got %s", entries[0].Collection)
	}
	
	if string(entries[0].Data) != string(testData) {
		t.Errorf("Data mismatch: expected %s, got %s", testData, entries[0].Data)
	}
	
	// Verify second entry
	if entries[1].Operation != WALOpDeleteVector {
		t.Errorf("Expected operation %s, got %s", WALOpDeleteVector, entries[1].Operation)
	}
	
	if entries[1].VectorID != "test2" {
		t.Errorf("Expected vector ID 'test2', got %s", entries[1].VectorID)
	}
}

func TestWALPersistenceWrapper(t *testing.T) {
	// Create temporary directories
	walDir := "/tmp/test_wal_persistence_wal"
	boltPath := "/tmp/test_wal_persistence.db"
	defer os.RemoveAll(walDir)
	defer os.RemoveAll(boltPath)
	
	// Create underlying BoltDB persistence
	boltPersistence, err := NewBoltPersistence(boltPath)
	if err != nil {
		t.Fatalf("Failed to create BoltDB persistence: %v", err)
	}
	
	// Create WAL config
	walConfig := WALConfig{
		Path:     walDir,
		MaxSize:  1024 * 1024,
		SyncMode: true,
	}
	
	// Create WAL-enabled persistence
	walPersistence, err := NewWALPersistence(boltPersistence, walConfig)
	if err != nil {
		t.Fatalf("Failed to create WAL persistence: %v", err)
	}
	defer walPersistence.Close()
	
	ctx := context.Background()
	
	// Test collection operations
	collection := core.Collection{
		Name:      "wal_test",
		Dimension: 3,
		IndexType: "flat",
		Distance:  "cosine",
		CreatedAt: time.Now(),
	}
	
	err = walPersistence.SaveCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to save collection: %v", err)
	}
	
	// Test vector operations
	vector := core.Vector{
		ID:     "v1",
		Values: []float32{1.0, 2.0, 3.0},
		Metadata: map[string]string{"type": "test"},
	}
	
	err = walPersistence.SaveVector(ctx, "wal_test", vector)
	if err != nil {
		t.Fatalf("Failed to save vector: %v", err)
	}
	
	// Verify data was saved
	loadedVector, err := walPersistence.LoadVector(ctx, "wal_test", "v1")
	if err != nil {
		t.Fatalf("Failed to load vector: %v", err)
	}
	
	if loadedVector.ID != vector.ID {
		t.Errorf("Vector ID mismatch: expected %s, got %s", vector.ID, loadedVector.ID)
	}
	
	// Test deletion
	err = walPersistence.DeleteVector(ctx, "wal_test", "v1")
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}
	
	// Verify deletion
	_, err = walPersistence.LoadVector(ctx, "wal_test", "v1")
	if err == nil {
		t.Error("Expected error when loading deleted vector")
	}
	
	// Check WAL stats
	stats := walPersistence.GetWALStats()
	if stats["last_wal_id"] == nil {
		t.Error("WAL stats should include last_wal_id")
	}
}

func TestWALRecovery(t *testing.T) {
	// Create temporary directories
	walDir := "/tmp/test_wal_recovery_wal"
	boltPath := "/tmp/test_wal_recovery.db"
	defer os.RemoveAll(walDir)
	defer os.RemoveAll(boltPath)
	
	ctx := context.Background()
	
	// Create WAL config
	walConfig := WALConfig{
		Path:     walDir,
		MaxSize:  1024 * 1024,
		SyncMode: true,
	}
	
	var walPersistence *WALPersistence
	
	// Phase 1: Write some data and simulate crash (close without proper shutdown)
	{
		boltPersistence, err := NewBoltPersistence(boltPath)
		if err != nil {
			t.Fatalf("Failed to create BoltDB persistence: %v", err)
		}
		
		walPersistence, err = NewWALPersistence(boltPersistence, walConfig)
		if err != nil {
			t.Fatalf("Failed to create WAL persistence: %v", err)
		}
		
		// Create collection
		collection := core.Collection{
			Name:      "recovery_test",
			Dimension: 2,
			IndexType: "flat",
			Distance:  "l2",
			CreatedAt: time.Now(),
		}
		
		err = walPersistence.SaveCollection(ctx, collection)
		if err != nil {
			t.Fatalf("Failed to save collection: %v", err)
		}
		
		// Add some vectors
		vectors := []core.Vector{
			{ID: "r1", Values: []float32{1.0, 2.0}, Metadata: map[string]string{"group": "A"}},
			{ID: "r2", Values: []float32{3.0, 4.0}, Metadata: map[string]string{"group": "B"}},
		}
		
		for _, vec := range vectors {
			err = walPersistence.SaveVector(ctx, "recovery_test", vec)
			if err != nil {
				t.Fatalf("Failed to save vector %s: %v", vec.ID, err)
			}
		}
		
		// Simulate crash - close WAL but keep the underlying persistence open
		walPersistence.wal.Close()
		walPersistence.underlying.Close()
	}
	
	// Phase 2: Restart and verify recovery
	{
		// Create new BoltDB persistence (data is persisted)
		boltPersistence, err := NewBoltPersistence(boltPath)
		if err != nil {
			t.Fatalf("Failed to create BoltDB persistence for recovery: %v", err)
		}
		
		// Create new WAL persistence - this should trigger recovery
		walPersistence, err := NewWALPersistence(boltPersistence, walConfig)
		if err != nil {
			t.Fatalf("Failed to create WAL persistence for recovery: %v", err)
		}
		defer walPersistence.Close()
		
		// Verify collection was recovered
		collections, err := walPersistence.LoadCollections(ctx)
		if err != nil {
			t.Fatalf("Failed to load collections after recovery: %v", err)
		}
		
		if len(collections) != 1 {
			t.Errorf("Expected 1 collection after recovery, got %d", len(collections))
		}
		
		if collections[0].Name != "recovery_test" {
			t.Errorf("Expected collection 'recovery_test', got %s", collections[0].Name)
		}
		
		// Verify vectors were recovered
		vectors, err := walPersistence.LoadVectors(ctx, "recovery_test")
		if err != nil {
			t.Fatalf("Failed to load vectors after recovery: %v", err)
		}
		
		if len(vectors) != 2 {
			t.Errorf("Expected 2 vectors after recovery, got %d", len(vectors))
		}
		
		// Verify specific vectors
		foundR1, foundR2 := false, false
		for _, vec := range vectors {
			if vec.ID == "r1" {
				foundR1 = true
				if vec.Values[0] != 1.0 || vec.Values[1] != 2.0 {
					t.Errorf("Vector r1 values mismatch")
				}
			}
			if vec.ID == "r2" {
				foundR2 = true
				if vec.Values[0] != 3.0 || vec.Values[1] != 4.0 {
					t.Errorf("Vector r2 values mismatch")
				}
			}
		}
		
		if !foundR1 {
			t.Error("Vector r1 not found after recovery")
		}
		if !foundR2 {
			t.Error("Vector r2 not found after recovery")
		}
	}
}

func TestWALRotation(t *testing.T) {
	// Create temporary WAL directory
	walDir := "/tmp/test_wal_rotation"
	defer os.RemoveAll(walDir)
	
	config := WALConfig{
		Path:     walDir,
		MaxSize:  100, // Very small size to trigger rotation
		SyncMode: true,
	}
	
	wal, err := NewWAL(config)
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()
	
	ctx := context.Background()
	
	// Write enough entries to trigger rotation
	largeData := make([]byte, 50) // Each entry will be sizeable
	for i := 0; i < len(largeData); i++ {
		largeData[i] = byte('A' + (i % 26))
	}
	
	for i := 0; i < 10; i++ {
		err = wal.WriteEntry(ctx, WALOpAddVector, "test", "v1", largeData)
		if err != nil {
			t.Fatalf("Failed to write WAL entry %d: %v", i, err)
		}
	}
	
	// Check that WAL file was rotated (archived file should exist)
	entries, err := os.ReadDir(walDir)
	if err != nil {
		t.Fatalf("Failed to read WAL directory: %v", err)
	}
	
	hasRotatedFile := false
	for _, entry := range entries {
		if entry.Name() != "wal.log" && !entry.IsDir() {
			hasRotatedFile = true
			break
		}
	}
	
	if !hasRotatedFile {
		t.Error("Expected to find rotated WAL file, but none found")
	}
}