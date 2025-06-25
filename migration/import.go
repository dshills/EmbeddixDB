package migration

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
	
	"github.com/dshills/EmbeddixDB/core"
)

// Importer handles data import operations
type Importer struct {
	persistence core.Persistence
}

// NewImporter creates a new data importer
func NewImporter(persistence core.Persistence) *Importer {
	return &Importer{
		persistence: persistence,
	}
}

// Import imports database data from files
func (i *Importer) Import(ctx context.Context, options ImportOptions) (*ImportResult, error) {
	// Validate options
	if err := i.validateImportOptions(options); err != nil {
		return nil, fmt.Errorf("invalid import options: %w", err)
	}
	
	// Load export metadata
	metadata, err := i.loadExportMetadata(options.InputDirectory)
	if err != nil {
		return nil, fmt.Errorf("failed to load export metadata: %w", err)
	}
	
	fmt.Printf("Importing data exported at %s\n", metadata.ExportedAt.Format("2006-01-02 15:04:05"))
	
	result := &ImportResult{
		ImportedAt:     metadata.ExportedAt,
		Format:         metadata.Format,
		CollectionResults: make(map[string]CollectionImportResult),
	}
	
	// Import collections metadata first
	collections, err := i.importCollectionsMetadata(ctx, options)
	if err != nil {
		return nil, fmt.Errorf("failed to import collections metadata: %w", err)
	}
	
	// Determine which collections to import
	collectionsToImport := i.getCollectionsToImport(collections, options.Collections)
	
	// Import each collection
	for _, collectionName := range collectionsToImport {
		fmt.Printf("Importing collection: %s\n", collectionName)
		
		collectionResult, err := i.importCollection(ctx, collectionName, metadata, options)
		if err != nil {
			return nil, fmt.Errorf("failed to import collection %s: %w", collectionName, err)
		}
		
		result.CollectionResults[collectionName] = *collectionResult
		result.TotalVectors += collectionResult.VectorCount
		
		// Import index state if available and requested
		if options.RestoreIndexes && metadata.IndexStates[collectionName] {
			if err := i.importIndexState(ctx, collectionName, options); err != nil {
				return nil, fmt.Errorf("failed to import index state for %s: %w", collectionName, err)
			}
			collectionResult.IndexRestored = true
		}
	}
	
	fmt.Printf("Import completed successfully. Imported %d vectors across %d collections\n", 
		result.TotalVectors, len(result.CollectionResults))
	
	return result, nil
}

// importCollection imports vectors for a single collection
func (i *Importer) importCollection(ctx context.Context, collectionName string, metadata *ExportMetadata, options ImportOptions) (*CollectionImportResult, error) {
	filename := fmt.Sprintf("%s_vectors.%s", collectionName, metadata.Format)
	filePath := filepath.Join(options.InputDirectory, filename)
	
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filePath, err)
	}
	defer file.Close()
	
	var vectors []core.Vector
	switch metadata.Format {
	case ExportFormatJSON:
		vectors, err = i.importVectorsJSON(file)
	case ExportFormatBinary:
		vectors, err = i.importVectorsBinary(file)
	default:
		return nil, fmt.Errorf("unsupported format: %s", metadata.Format)
	}
	
	if err != nil {
		return nil, fmt.Errorf("failed to read vectors: %w", err)
	}
	
	// Validate data if requested
	if options.ValidateData {
		if err := i.validateVectors(vectors); err != nil {
			return nil, fmt.Errorf("vector validation failed: %w", err)
		}
	}
	
	// Check if we should overwrite existing data
	if !options.OverwriteData {
		existingVectors, err := i.persistence.LoadVectors(ctx, collectionName)
		if err == nil && len(existingVectors) > 0 {
			return nil, fmt.Errorf("collection %s already has data, use OverwriteData=true to replace", collectionName)
		}
	}
	
	// Import vectors in batches
	batchSize := options.BatchSize
	if batchSize <= 0 {
		batchSize = 1000
	}
	
	importedCount := 0
	for idx := 0; idx < len(vectors); idx += batchSize {
		end := idx + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		
		batch := vectors[idx:end]
		if err := i.persistence.SaveVectorsBatch(ctx, collectionName, batch); err != nil {
			return nil, fmt.Errorf("failed to save vectors batch: %w", err)
		}
		
		importedCount += len(batch)
	}
	
	return &CollectionImportResult{
		VectorCount:   importedCount,
		IndexRestored: false,
	}, nil
}

// importVectorsJSON imports vectors from JSON format
func (i *Importer) importVectorsJSON(reader io.Reader) ([]core.Vector, error) {
	decoder := json.NewDecoder(reader)
	
	var vectors []core.Vector
	for decoder.More() {
		var batch []core.Vector
		if err := decoder.Decode(&batch); err != nil {
			return nil, fmt.Errorf("failed to decode JSON batch: %w", err)
		}
		vectors = append(vectors, batch...)
	}
	
	return vectors, nil
}

// importVectorsBinary imports vectors from binary format
func (i *Importer) importVectorsBinary(reader io.Reader) ([]core.Vector, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read binary data: %w", err)
	}
	
	var vectors []core.Vector
	if err := json.Unmarshal(data, &vectors); err != nil {
		return nil, fmt.Errorf("failed to unmarshal binary data: %w", err)
	}
	
	return vectors, nil
}

// importIndexState imports index state for a collection
func (i *Importer) importIndexState(ctx context.Context, collectionName string, options ImportOptions) error {
	filename := fmt.Sprintf("%s_index.json", collectionName)
	filePath := filepath.Join(options.InputDirectory, filename)
	
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // Index state file doesn't exist, which is fine
		}
		return fmt.Errorf("failed to read index state file: %w", err)
	}
	
	return i.persistence.SaveIndexState(ctx, collectionName, data)
}

// importCollectionsMetadata imports collection definitions
func (i *Importer) importCollectionsMetadata(ctx context.Context, options ImportOptions) ([]core.Collection, error) {
	filePath := filepath.Join(options.InputDirectory, "collections.json")
	
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open collections metadata file: %w", err)
	}
	defer file.Close()
	
	var collections []core.Collection
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&collections); err != nil {
		return nil, fmt.Errorf("failed to decode collections metadata: %w", err)
	}
	
	// Create/update collections
	for _, collection := range collections {
		if err := i.persistence.SaveCollection(ctx, collection); err != nil {
			return nil, fmt.Errorf("failed to save collection %s: %w", collection.Name, err)
		}
	}
	
	return collections, nil
}

// loadExportMetadata loads export metadata from a directory
func (i *Importer) loadExportMetadata(inputDir string) (*ExportMetadata, error) {
	filePath := filepath.Join(inputDir, "export_metadata.json")
	
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open export metadata file: %w", err)
	}
	defer file.Close()
	
	var metadata ExportMetadata
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&metadata); err != nil {
		return nil, fmt.Errorf("failed to decode export metadata: %w", err)
	}
	
	return &metadata, nil
}

// getCollectionsToImport determines which collections to import
func (i *Importer) getCollectionsToImport(available []core.Collection, requested []string) []string {
	if len(requested) == 0 {
		// Import all available collections
		names := make([]string, len(available))
		for i, collection := range available {
			names[i] = collection.Name
		}
		return names
	}
	
	// Filter to only requested collections that are available
	availableMap := make(map[string]bool)
	for _, collection := range available {
		availableMap[collection.Name] = true
	}
	
	var toImport []string
	for _, name := range requested {
		if availableMap[name] {
			toImport = append(toImport, name)
		}
	}
	
	return toImport
}

// validateVectors validates imported vectors
func (i *Importer) validateVectors(vectors []core.Vector) error {
	for idx, vector := range vectors {
		if err := core.ValidateVector(vector); err != nil {
			return fmt.Errorf("invalid vector at index %d: %w", idx, err)
		}
	}
	return nil
}

// validateImportOptions validates import options
func (i *Importer) validateImportOptions(options ImportOptions) error {
	if options.InputDirectory == "" {
		return fmt.Errorf("input directory is required")
	}
	
	if _, err := os.Stat(options.InputDirectory); os.IsNotExist(err) {
		return fmt.Errorf("input directory does not exist: %s", options.InputDirectory)
	}
	
	if options.BatchSize <= 0 {
		options.BatchSize = 1000
	}
	
	return nil
}

// ImportResult contains the results of an import operation
type ImportResult struct {
	ImportedAt        time.Time                           `json:"imported_at"`
	Format            ExportFormat                        `json:"format"`
	TotalVectors      int                                 `json:"total_vectors"`
	CollectionResults map[string]CollectionImportResult   `json:"collection_results"`
}

// CollectionImportResult contains the results for a single collection import
type CollectionImportResult struct {
	VectorCount   int  `json:"vector_count"`
	IndexRestored bool `json:"index_restored"`
}

// DefaultImportOptions returns default import options
func DefaultImportOptions(inputDir string) ImportOptions {
	return ImportOptions{
		Format:         ExportFormatJSON,
		RestoreIndexes: true,
		OverwriteData:  false,
		BatchSize:      1000,
		InputDirectory: inputDir,
		ValidateData:   true,
	}
}

// ListAvailableExports lists available exports in a directory
func ListAvailableExports(baseDir string) ([]ExportInfo, error) {
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}
	
	var exports []ExportInfo
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		
		exportDir := filepath.Join(baseDir, entry.Name())
		metadataPath := filepath.Join(exportDir, "export_metadata.json")
		
		if _, err := os.Stat(metadataPath); err != nil {
			continue // Not a valid export directory
		}
		
		// Load metadata
		file, err := os.Open(metadataPath)
		if err != nil {
			continue
		}
		
		var metadata ExportMetadata
		decoder := json.NewDecoder(file)
		if err := decoder.Decode(&metadata); err != nil {
			file.Close()
			continue
		}
		file.Close()
		
		totalVectors := 0
		for _, count := range metadata.VectorCounts {
			totalVectors += count
		}
		
		exports = append(exports, ExportInfo{
			Directory:     exportDir,
			ExportedAt:    metadata.ExportedAt,
			Format:        metadata.Format,
			Collections:   metadata.Collections,
			TotalVectors:  totalVectors,
			HasIndexes:    len(metadata.IndexStates) > 0,
		})
	}
	
	return exports, nil
}

// ExportInfo provides information about an available export
type ExportInfo struct {
	Directory     string       `json:"directory"`
	ExportedAt    time.Time    `json:"exported_at"`
	Format        ExportFormat `json:"format"`
	Collections   []string     `json:"collections"`
	TotalVectors  int          `json:"total_vectors"`
	HasIndexes    bool         `json:"has_indexes"`
}