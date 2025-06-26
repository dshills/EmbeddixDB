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

// ExportFormat represents the export file format
type ExportFormat string

const (
	ExportFormatJSON   ExportFormat = "json"
	ExportFormatBinary ExportFormat = "binary"
)

// ExportOptions contains options for data export
type ExportOptions struct {
	Format          ExportFormat `json:"format"`
	IncludeIndexes  bool         `json:"include_indexes"`
	CompressOutput  bool         `json:"compress_output"`
	Collections     []string     `json:"collections,omitempty"` // Empty means all collections
	BatchSize       int          `json:"batch_size"`
	OutputDirectory string       `json:"output_directory"`
}

// ImportOptions contains options for data import
type ImportOptions struct {
	Format         ExportFormat `json:"format"`
	RestoreIndexes bool         `json:"restore_indexes"`
	OverwriteData  bool         `json:"overwrite_data"`
	Collections    []string     `json:"collections,omitempty"` // Empty means all collections
	BatchSize      int          `json:"batch_size"`
	InputDirectory string       `json:"input_directory"`
	ValidateData   bool         `json:"validate_data"`
}

// ExportMetadata contains metadata about an export
type ExportMetadata struct {
	Version      string                 `json:"version"`
	ExportedAt   time.Time              `json:"exported_at"`
	Format       ExportFormat           `json:"format"`
	Collections  []string               `json:"collections"`
	VectorCounts map[string]int         `json:"vector_counts"`
	IndexStates  map[string]bool        `json:"index_states"`
	Options      ExportOptions          `json:"options"`
	DatabaseInfo map[string]interface{} `json:"database_info"`
}

// Exporter handles data export operations
type Exporter struct {
	persistence core.Persistence
}

// NewExporter creates a new data exporter
func NewExporter(persistence core.Persistence) *Exporter {
	return &Exporter{
		persistence: persistence,
	}
}

// Export exports database data to files
func (e *Exporter) Export(ctx context.Context, options ExportOptions) (*ExportMetadata, error) {
	// Validate options
	if err := e.validateExportOptions(options); err != nil {
		return nil, fmt.Errorf("invalid export options: %w", err)
	}

	// Create output directory
	if err := os.MkdirAll(options.OutputDirectory, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}

	// Get collections to export
	collectionsToExport, err := e.getCollectionsToExport(ctx, options.Collections)
	if err != nil {
		return nil, fmt.Errorf("failed to get collections: %w", err)
	}

	metadata := &ExportMetadata{
		Version:      "1.0",
		ExportedAt:   time.Now(),
		Format:       options.Format,
		Collections:  collectionsToExport,
		VectorCounts: make(map[string]int),
		IndexStates:  make(map[string]bool),
		Options:      options,
		DatabaseInfo: make(map[string]interface{}),
	}

	// Export each collection
	for _, collectionName := range collectionsToExport {
		fmt.Printf("Exporting collection: %s\n", collectionName)

		count, err := e.exportCollection(ctx, collectionName, options)
		if err != nil {
			return nil, fmt.Errorf("failed to export collection %s: %w", collectionName, err)
		}

		metadata.VectorCounts[collectionName] = count

		// Export index state if requested
		if options.IncludeIndexes {
			hasIndex, err := e.exportIndexState(ctx, collectionName, options)
			if err != nil {
				return nil, fmt.Errorf("failed to export index state for %s: %w", collectionName, err)
			}
			metadata.IndexStates[collectionName] = hasIndex
		}
	}

	// Export collections metadata
	if err := e.exportCollectionsMetadata(ctx, collectionsToExport, options); err != nil {
		return nil, fmt.Errorf("failed to export collections metadata: %w", err)
	}

	// Save export metadata
	if err := e.saveExportMetadata(metadata, options.OutputDirectory); err != nil {
		return nil, fmt.Errorf("failed to save export metadata: %w", err)
	}

	fmt.Printf("Export completed successfully to %s\n", options.OutputDirectory)
	return metadata, nil
}

// exportCollection exports vectors from a single collection
func (e *Exporter) exportCollection(ctx context.Context, collectionName string, options ExportOptions) (int, error) {
	vectors, err := e.persistence.LoadVectors(ctx, collectionName)
	if err != nil {
		return 0, err
	}

	filename := fmt.Sprintf("%s_vectors.%s", collectionName, options.Format)
	filePath := filepath.Join(options.OutputDirectory, filename)

	file, err := os.Create(filePath)
	if err != nil {
		return 0, fmt.Errorf("failed to create file %s: %w", filePath, err)
	}
	defer file.Close()

	switch options.Format {
	case ExportFormatJSON:
		return e.exportVectorsJSON(vectors, file, options.BatchSize)
	case ExportFormatBinary:
		return e.exportVectorsBinary(vectors, file, options.BatchSize)
	default:
		return 0, fmt.Errorf("unsupported export format: %s", options.Format)
	}
}

// exportVectorsJSON exports vectors in JSON format
func (e *Exporter) exportVectorsJSON(vectors []core.Vector, writer io.Writer, batchSize int) (int, error) {
	encoder := json.NewEncoder(writer)
	encoder.SetIndent("", "  ")

	if batchSize <= 0 {
		batchSize = 1000
	}

	totalCount := 0
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		batch := vectors[i:end]
		if err := encoder.Encode(batch); err != nil {
			return totalCount, fmt.Errorf("failed to encode vectors batch: %w", err)
		}

		totalCount += len(batch)
	}

	return totalCount, nil
}

// exportVectorsBinary exports vectors in binary format
func (e *Exporter) exportVectorsBinary(vectors []core.Vector, writer io.Writer, batchSize int) (int, error) {
	// For binary format, we'll use JSON encoding for simplicity
	// In a production system, you might want a more efficient binary format
	data, err := json.Marshal(vectors)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal vectors: %w", err)
	}

	_, err = writer.Write(data)
	if err != nil {
		return 0, fmt.Errorf("failed to write binary data: %w", err)
	}

	return len(vectors), nil
}

// exportIndexState exports index state for a collection
func (e *Exporter) exportIndexState(ctx context.Context, collectionName string, options ExportOptions) (bool, error) {
	indexData, err := e.persistence.LoadIndexState(ctx, collectionName)
	if err != nil {
		// Index state might not exist, which is fine
		return false, nil
	}

	filename := fmt.Sprintf("%s_index.json", collectionName)
	filePath := filepath.Join(options.OutputDirectory, filename)

	return true, os.WriteFile(filePath, indexData, 0644)
}

// exportCollectionsMetadata exports collection definitions
func (e *Exporter) exportCollectionsMetadata(ctx context.Context, collectionNames []string, options ExportOptions) error {
	collections := make([]core.Collection, 0, len(collectionNames))

	for _, name := range collectionNames {
		collection, err := e.persistence.LoadCollection(ctx, name)
		if err != nil {
			return fmt.Errorf("failed to load collection %s: %w", name, err)
		}
		collections = append(collections, collection)
	}

	filename := "collections.json"
	filePath := filepath.Join(options.OutputDirectory, filename)

	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create collections file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(collections)
}

// getCollectionsToExport determines which collections to export
func (e *Exporter) getCollectionsToExport(ctx context.Context, requested []string) ([]string, error) {
	if len(requested) > 0 {
		return requested, nil
	}

	// Export all collections
	collections, err := e.persistence.LoadCollections(ctx)
	if err != nil {
		return nil, err
	}

	names := make([]string, len(collections))
	for i, collection := range collections {
		names[i] = collection.Name
	}

	return names, nil
}

// saveExportMetadata saves export metadata to a file
func (e *Exporter) saveExportMetadata(metadata *ExportMetadata, outputDir string) error {
	filePath := filepath.Join(outputDir, "export_metadata.json")

	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(metadata)
}

// validateExportOptions validates export options
func (e *Exporter) validateExportOptions(options ExportOptions) error {
	if options.OutputDirectory == "" {
		return fmt.Errorf("output directory is required")
	}

	if options.Format != ExportFormatJSON && options.Format != ExportFormatBinary {
		return fmt.Errorf("unsupported format: %s", options.Format)
	}

	if options.BatchSize <= 0 {
		options.BatchSize = 1000
	}

	return nil
}

// DefaultExportOptions returns default export options
func DefaultExportOptions(outputDir string) ExportOptions {
	return ExportOptions{
		Format:          ExportFormatJSON,
		IncludeIndexes:  true,
		CompressOutput:  false,
		BatchSize:       1000,
		OutputDirectory: outputDir,
	}
}
