package migration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
	
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/persistence"
)

// RecoveryOptions contains options for database recovery
type RecoveryOptions struct {
	BackupDirectory   string                 `json:"backup_directory"`
	TargetDirectory   string                 `json:"target_directory"`
	PersistenceType   persistence.PersistenceType `json:"persistence_type"`
	RestoreIndexes    bool                   `json:"restore_indexes"`
	ValidateData      bool                   `json:"validate_data"`
	CreateBackup      bool                   `json:"create_backup"`
	Force             bool                   `json:"force"`
}

// RecoveryManager handles disaster recovery operations
type RecoveryManager struct {
	sourceDB    core.Persistence
	backupDir   string
}

// NewRecoveryManager creates a new recovery manager
func NewRecoveryManager(sourceDB core.Persistence, backupDir string) *RecoveryManager {
	return &RecoveryManager{
		sourceDB:  sourceDB,
		backupDir: backupDir,
	}
}

// CreateBackup creates a complete backup of the database
func (r *RecoveryManager) CreateBackup(ctx context.Context, backupName string) (*BackupInfo, error) {
	if backupName == "" {
		backupName = fmt.Sprintf("backup_%s", time.Now().Format("20060102_150405"))
	}
	
	backupPath := filepath.Join(r.backupDir, backupName)
	
	// Create backup directory
	if err := os.MkdirAll(backupPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create backup directory: %w", err)
	}
	
	fmt.Printf("Creating backup: %s\n", backupName)
	
	// Export all data
	exporter := NewExporter(r.sourceDB)
	exportOptions := DefaultExportOptions(backupPath)
	exportOptions.IncludeIndexes = true
	
	metadata, err := exporter.Export(ctx, exportOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to export data for backup: %w", err)
	}
	
	// Create backup info
	backupInfo := &BackupInfo{
		Name:         backupName,
		Path:         backupPath,
		CreatedAt:    time.Now(),
		Collections:  metadata.Collections,
		VectorCounts: metadata.VectorCounts,
		Format:       metadata.Format,
		Size:         r.calculateBackupSize(backupPath),
	}
	
	// Save backup info
	if err := r.saveBackupInfo(backupInfo); err != nil {
		return nil, fmt.Errorf("failed to save backup info: %w", err)
	}
	
	fmt.Printf("Backup created successfully: %s\n", backupPath)
	return backupInfo, nil
}

// RestoreFromBackup restores database from a backup
func (r *RecoveryManager) RestoreFromBackup(ctx context.Context, backupName string, options RecoveryOptions) (*RecoveryResult, error) {
	backupPath := filepath.Join(r.backupDir, backupName)
	
	// Verify backup exists
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("backup not found: %s", backupName)
	}
	
	fmt.Printf("Restoring from backup: %s\n", backupName)
	
	// Load backup info
	_, err := r.loadBackupInfo(backupName)
	if err != nil {
		return nil, fmt.Errorf("failed to load backup info: %w", err)
	}
	
	// Create target persistence if specified
	var targetDB core.Persistence
	if options.TargetDirectory != "" {
		targetDB, err = r.createTargetPersistence(options)
		if err != nil {
			return nil, fmt.Errorf("failed to create target persistence: %w", err)
		}
		defer targetDB.Close()
	} else {
		targetDB = r.sourceDB
	}
	
	// Create backup of current data if requested
	var currentBackupName string
	if options.CreateBackup && targetDB == r.sourceDB {
		currentBackupName = fmt.Sprintf("pre_restore_%s", time.Now().Format("20060102_150405"))
		_, err := r.CreateBackup(ctx, currentBackupName)
		if err != nil {
			return nil, fmt.Errorf("failed to create pre-restore backup: %w", err)
		}
	}
	
	// Import data from backup
	importer := NewImporter(targetDB)
	importOptions := DefaultImportOptions(backupPath)
	importOptions.RestoreIndexes = options.RestoreIndexes
	importOptions.ValidateData = options.ValidateData
	importOptions.OverwriteData = options.Force
	
	importResult, err := importer.Import(ctx, importOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to import backup data: %w", err)
	}
	
	result := &RecoveryResult{
		BackupName:        backupName,
		RestoredAt:        time.Now(),
		Collections:       len(importResult.CollectionResults),
		VectorsRestored:   importResult.TotalVectors,
		IndexesRestored:   r.countRestoredIndexes(importResult),
		PreRestoreBackup:  currentBackupName,
	}
	
	fmt.Printf("Restore completed successfully. Restored %d vectors across %d collections\n", 
		result.VectorsRestored, result.Collections)
	
	return result, nil
}

// ListBackups lists available backups
func (r *RecoveryManager) ListBackups() ([]BackupInfo, error) {
	entries, err := os.ReadDir(r.backupDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []BackupInfo{}, nil
		}
		return nil, fmt.Errorf("failed to read backup directory: %w", err)
	}
	
	var backups []BackupInfo
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		
		backupInfo, err := r.loadBackupInfo(entry.Name())
		if err != nil {
			continue // Skip invalid backups
		}
		
		backups = append(backups, *backupInfo)
	}
	
	return backups, nil
}

// DeleteBackup removes a backup
func (r *RecoveryManager) DeleteBackup(backupName string) error {
	backupPath := filepath.Join(r.backupDir, backupName)
	
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		return fmt.Errorf("backup not found: %s", backupName)
	}
	
	return os.RemoveAll(backupPath)
}

// VerifyBackup verifies the integrity of a backup
func (r *RecoveryManager) VerifyBackup(ctx context.Context, backupName string) (*BackupVerification, error) {
	backupPath := filepath.Join(r.backupDir, backupName)
	
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("backup not found: %s", backupName)
	}
	
	// Load backup info
	backupInfo, err := r.loadBackupInfo(backupName)
	if err != nil {
		return nil, fmt.Errorf("failed to load backup info: %w", err)
	}
	
	verification := &BackupVerification{
		BackupName: backupName,
		VerifiedAt: time.Now(),
		Valid:      true,
		Issues:     make([]string, 0),
	}
	
	// Verify export metadata exists
	metadataPath := filepath.Join(backupPath, "export_metadata.json")
	if _, err := os.Stat(metadataPath); os.IsNotExist(err) {
		verification.Valid = false
		verification.Issues = append(verification.Issues, "export metadata file missing")
	}
	
	// Verify collections metadata exists
	collectionsPath := filepath.Join(backupPath, "collections.json")
	if _, err := os.Stat(collectionsPath); os.IsNotExist(err) {
		verification.Valid = false
		verification.Issues = append(verification.Issues, "collections metadata file missing")
	}
	
	// Verify vector files exist for each collection
	for _, collection := range backupInfo.Collections {
		vectorFile := filepath.Join(backupPath, fmt.Sprintf("%s_vectors.%s", collection, backupInfo.Format))
		if _, err := os.Stat(vectorFile); os.IsNotExist(err) {
			verification.Valid = false
			verification.Issues = append(verification.Issues, fmt.Sprintf("vector file missing for collection %s", collection))
		}
	}
	
	// Try to load and validate some data
	if verification.Valid {
		if err := r.validateBackupData(ctx, backupPath, backupInfo); err != nil {
			verification.Valid = false
			verification.Issues = append(verification.Issues, fmt.Sprintf("data validation failed: %v", err))
		}
	}
	
	return verification, nil
}

// validateBackupData performs basic validation on backup data
func (r *RecoveryManager) validateBackupData(ctx context.Context, backupPath string, backupInfo *BackupInfo) error {
	// Create a temporary memory persistence for validation
	memPersistence := persistence.NewMemoryPersistence()
	importer := NewImporter(memPersistence)
	
	// Try to import one collection to verify data integrity
	if len(backupInfo.Collections) > 0 {
		importOptions := DefaultImportOptions(backupPath)
		importOptions.Collections = []string{backupInfo.Collections[0]}
		importOptions.ValidateData = true
		
		_, err := importer.Import(ctx, importOptions)
		if err != nil {
			return fmt.Errorf("failed to validate collection %s: %w", backupInfo.Collections[0], err)
		}
	}
	
	return nil
}

// createTargetPersistence creates a new persistence instance for restore target
func (r *RecoveryManager) createTargetPersistence(options RecoveryOptions) (core.Persistence, error) {
	config := persistence.PersistenceConfig{
		Type: options.PersistenceType,
		Path: options.TargetDirectory,
	}
	
	factory := persistence.NewDefaultFactory()
	return factory.CreatePersistence(config)
}

// calculateBackupSize calculates the size of a backup directory
func (r *RecoveryManager) calculateBackupSize(backupPath string) int64 {
	var size int64
	filepath.Walk(backupPath, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}

// countRestoredIndexes counts how many indexes were restored
func (r *RecoveryManager) countRestoredIndexes(result *ImportResult) int {
	count := 0
	for _, collectionResult := range result.CollectionResults {
		if collectionResult.IndexRestored {
			count++
		}
	}
	return count
}

// saveBackupInfo saves backup information
func (r *RecoveryManager) saveBackupInfo(info *BackupInfo) error {
	infoPath := filepath.Join(r.backupDir, info.Name, "backup_info.json")
	return r.saveJSON(infoPath, info)
}

// loadBackupInfo loads backup information
func (r *RecoveryManager) loadBackupInfo(backupName string) (*BackupInfo, error) {
	infoPath := filepath.Join(r.backupDir, backupName, "backup_info.json")
	
	var info BackupInfo
	if err := r.loadJSON(infoPath, &info); err != nil {
		return nil, err
	}
	
	return &info, nil
}

// saveJSON saves an object as JSON
func (r *RecoveryManager) saveJSON(path string, obj interface{}) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(obj)
}

// loadJSON loads an object from JSON
func (r *RecoveryManager) loadJSON(path string, obj interface{}) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	decoder := json.NewDecoder(file)
	return decoder.Decode(obj)
}

// BackupInfo contains information about a backup
type BackupInfo struct {
	Name         string            `json:"name"`
	Path         string            `json:"path"`
	CreatedAt    time.Time         `json:"created_at"`
	Collections  []string          `json:"collections"`
	VectorCounts map[string]int    `json:"vector_counts"`
	Format       ExportFormat      `json:"format"`
	Size         int64             `json:"size"`
}

// RecoveryResult contains the results of a recovery operation
type RecoveryResult struct {
	BackupName       string    `json:"backup_name"`
	RestoredAt       time.Time `json:"restored_at"`
	Collections      int       `json:"collections"`
	VectorsRestored  int       `json:"vectors_restored"`
	IndexesRestored  int       `json:"indexes_restored"`
	PreRestoreBackup string    `json:"pre_restore_backup,omitempty"`
}

// BackupVerification contains the results of backup verification
type BackupVerification struct {
	BackupName string    `json:"backup_name"`
	VerifiedAt time.Time `json:"verified_at"`
	Valid      bool      `json:"valid"`
	Issues     []string  `json:"issues"`
}