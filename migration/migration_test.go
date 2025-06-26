package migration

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/persistence"
)

func TestMigrationBasics(t *testing.T) {
	// Create temporary directory
	tempDir := "/tmp/test_migration"
	defer os.RemoveAll(tempDir)

	// Create memory persistence for testing
	memPersistence := persistence.NewMemoryPersistence()
	migrator := NewMigrator(memPersistence)

	ctx := context.Background()

	// Test initial state
	version, err := migrator.GetCurrentVersion(ctx)
	if err != nil {
		t.Fatalf("Failed to get current version: %v", err)
	}

	if version != 0 {
		t.Errorf("Expected initial version 0, got %d", version)
	}

	// Add test migrations
	migration1 := &Migration{
		Version:     1,
		Name:        "create_users_collection",
		Description: "Create users collection for testing",
		UpFunc: func(ctx context.Context, db core.Persistence) error {
			collection := core.Collection{
				Name:      "users",
				Dimension: 128,
				IndexType: "flat",
				Distance:  "cosine",
				CreatedAt: time.Now(),
			}
			return db.SaveCollection(ctx, collection)
		},
		DownFunc: func(ctx context.Context, db core.Persistence) error {
			return db.DeleteCollection(ctx, "users")
		},
		CreatedAt: time.Now(),
	}

	migration2 := &Migration{
		Version:     2,
		Name:        "create_documents_collection",
		Description: "Create documents collection for testing",
		UpFunc: func(ctx context.Context, db core.Persistence) error {
			collection := core.Collection{
				Name:      "documents",
				Dimension: 256,
				IndexType: "hnsw",
				Distance:  "l2",
				CreatedAt: time.Now(),
			}
			return db.SaveCollection(ctx, collection)
		},
		DownFunc: func(ctx context.Context, db core.Persistence) error {
			return db.DeleteCollection(ctx, "documents")
		},
		CreatedAt: time.Now(),
	}

	migrator.AddMigration(migration1)
	migrator.AddMigration(migration2)

	// Test migration up
	err = migrator.MigrateUp(ctx)
	if err != nil {
		t.Fatalf("Failed to migrate up: %v", err)
	}

	// Verify current version
	version, err = migrator.GetCurrentVersion(ctx)
	if err != nil {
		t.Fatalf("Failed to get current version after migration: %v", err)
	}

	if version != 2 {
		t.Errorf("Expected version 2 after migration, got %d", version)
	}

	// Verify collections were created
	collections, err := memPersistence.LoadCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to load collections: %v", err)
	}

	if len(collections) != 2 {
		t.Errorf("Expected 2 collections, got %d", len(collections))
	}

	// Test migration down
	err = migrator.Migrate(ctx, 1)
	if err != nil {
		t.Fatalf("Failed to migrate down: %v", err)
	}

	// Verify version after rollback
	version, err = migrator.GetCurrentVersion(ctx)
	if err != nil {
		t.Fatalf("Failed to get current version after rollback: %v", err)
	}

	if version != 1 {
		t.Errorf("Expected version 1 after rollback, got %d", version)
	}

	// Verify one collection was removed
	collections, err = memPersistence.LoadCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to load collections after rollback: %v", err)
	}

	if len(collections) != 1 {
		t.Errorf("Expected 1 collection after rollback, got %d", len(collections))
	}

	if collections[0].Name != "users" {
		t.Errorf("Expected 'users' collection to remain, got %s", collections[0].Name)
	}
}

func TestExportImport(t *testing.T) {
	// Create temporary directories
	exportDir := "/tmp/test_export"
	importDir := "/tmp/test_import"
	defer os.RemoveAll(exportDir)
	defer os.RemoveAll(importDir)

	ctx := context.Background()

	// Create source persistence with test data
	sourcePersistence := persistence.NewMemoryPersistence()

	// Create test collection
	collection := core.Collection{
		Name:      "test_export",
		Dimension: 3,
		IndexType: "flat",
		Distance:  "cosine",
		CreatedAt: time.Now(),
	}

	err := sourcePersistence.SaveCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}

	// Add test vectors
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{1.0, 0.0, 0.0}, Metadata: map[string]string{"type": "test"}},
		{ID: "v2", Values: []float32{0.0, 1.0, 0.0}, Metadata: map[string]string{"type": "test"}},
		{ID: "v3", Values: []float32{0.0, 0.0, 1.0}, Metadata: map[string]string{"type": "test"}},
	}

	for _, vec := range vectors {
		err := sourcePersistence.SaveVector(ctx, "test_export", vec)
		if err != nil {
			t.Fatalf("Failed to save vector %s: %v", vec.ID, err)
		}
	}

	// Export data
	exporter := NewExporter(sourcePersistence)
	exportOptions := DefaultExportOptions(exportDir)

	metadata, err := exporter.Export(ctx, exportOptions)
	if err != nil {
		t.Fatalf("Failed to export data: %v", err)
	}

	if len(metadata.Collections) != 1 {
		t.Errorf("Expected 1 collection in export, got %d", len(metadata.Collections))
	}

	if metadata.VectorCounts["test_export"] != 3 {
		t.Errorf("Expected 3 vectors in export, got %d", metadata.VectorCounts["test_export"])
	}

	// Create target persistence for import
	targetPersistence := persistence.NewMemoryPersistence()

	// Import data
	importer := NewImporter(targetPersistence)
	importOptions := DefaultImportOptions(exportDir)

	importResult, err := importer.Import(ctx, importOptions)
	if err != nil {
		t.Fatalf("Failed to import data: %v", err)
	}

	if importResult.TotalVectors != 3 {
		t.Errorf("Expected 3 vectors imported, got %d", importResult.TotalVectors)
	}

	// Verify imported data
	importedVectors, err := targetPersistence.LoadVectors(ctx, "test_export")
	if err != nil {
		t.Fatalf("Failed to load imported vectors: %v", err)
	}

	if len(importedVectors) != 3 {
		t.Errorf("Expected 3 imported vectors, got %d", len(importedVectors))
	}

	// Verify vector content
	vectorMap := make(map[string]core.Vector)
	for _, vec := range importedVectors {
		vectorMap[vec.ID] = vec
	}

	for _, originalVec := range vectors {
		importedVec, exists := vectorMap[originalVec.ID]
		if !exists {
			t.Errorf("Vector %s not found in imported data", originalVec.ID)
			continue
		}

		if len(importedVec.Values) != len(originalVec.Values) {
			t.Errorf("Vector %s dimension mismatch", originalVec.ID)
			continue
		}

		for i, val := range originalVec.Values {
			if importedVec.Values[i] != val {
				t.Errorf("Vector %s value mismatch at index %d", originalVec.ID, i)
			}
		}
	}
}

func TestRecoveryManager(t *testing.T) {
	// Create temporary directories
	dbDir := "/tmp/test_recovery_db"
	backupDir := "/tmp/test_recovery_backup"
	defer os.RemoveAll(dbDir)
	defer os.RemoveAll(backupDir)

	ctx := context.Background()

	// Create source database with test data
	sourcePersistence := persistence.NewMemoryPersistence()

	// Create test data
	collection := core.Collection{
		Name:      "recovery_test",
		Dimension: 2,
		IndexType: "flat",
		Distance:  "l2",
		CreatedAt: time.Now(),
	}

	err := sourcePersistence.SaveCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}

	vectors := []core.Vector{
		{ID: "r1", Values: []float32{1.0, 2.0}, Metadata: map[string]string{"group": "A"}},
		{ID: "r2", Values: []float32{3.0, 4.0}, Metadata: map[string]string{"group": "B"}},
	}

	for _, vec := range vectors {
		err := sourcePersistence.SaveVector(ctx, "recovery_test", vec)
		if err != nil {
			t.Fatalf("Failed to save vector %s: %v", vec.ID, err)
		}
	}

	// Create recovery manager
	recoveryManager := NewRecoveryManager(sourcePersistence, backupDir)

	// Create backup
	backupInfo, err := recoveryManager.CreateBackup(ctx, "test_backup")
	if err != nil {
		t.Fatalf("Failed to create backup: %v", err)
	}

	if backupInfo.Name != "test_backup" {
		t.Errorf("Expected backup name 'test_backup', got %s", backupInfo.Name)
	}

	if len(backupInfo.Collections) != 1 {
		t.Errorf("Expected 1 collection in backup, got %d", len(backupInfo.Collections))
	}

	if backupInfo.VectorCounts["recovery_test"] != 2 {
		t.Errorf("Expected 2 vectors in backup, got %d", backupInfo.VectorCounts["recovery_test"])
	}

	// Verify backup
	verification, err := recoveryManager.VerifyBackup(ctx, "test_backup")
	if err != nil {
		t.Fatalf("Failed to verify backup: %v", err)
	}

	if !verification.Valid {
		t.Errorf("Backup verification failed: %v", verification.Issues)
	}

	// List backups
	backups, err := recoveryManager.ListBackups()
	if err != nil {
		t.Fatalf("Failed to list backups: %v", err)
	}

	if len(backups) != 1 {
		t.Errorf("Expected 1 backup, got %d", len(backups))
	}

	// Test restore (to new persistence)
	targetPersistence := persistence.NewMemoryPersistence()
	targetRecoveryManager := NewRecoveryManager(targetPersistence, backupDir)

	recoveryOptions := RecoveryOptions{
		RestoreIndexes: true,
		ValidateData:   true,
		Force:          true,
	}

	restoreResult, err := targetRecoveryManager.RestoreFromBackup(ctx, "test_backup", recoveryOptions)
	if err != nil {
		t.Fatalf("Failed to restore from backup: %v", err)
	}

	if restoreResult.VectorsRestored != 2 {
		t.Errorf("Expected 2 vectors restored, got %d", restoreResult.VectorsRestored)
	}

	if restoreResult.Collections != 1 {
		t.Errorf("Expected 1 collection restored, got %d", restoreResult.Collections)
	}

	// Verify restored data
	restoredVectors, err := targetPersistence.LoadVectors(ctx, "recovery_test")
	if err != nil {
		t.Fatalf("Failed to load restored vectors: %v", err)
	}

	if len(restoredVectors) != 2 {
		t.Errorf("Expected 2 restored vectors, got %d", len(restoredVectors))
	}
}
