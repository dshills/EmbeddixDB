package persistence

import (
	"fmt"
	"time"
)

// PersistenceType represents the type of persistence backend
type PersistenceType string

const (
	PersistenceMemory PersistenceType = "memory"
	PersistenceBolt   PersistenceType = "bolt"
	PersistenceBadger PersistenceType = "badger"
)

// PersistenceConfig holds configuration for persistence layers
type PersistenceConfig struct {
	// Type of persistence backend
	Type PersistenceType `json:"type" yaml:"type"`
	
	// Path to database directory/file
	Path string `json:"path" yaml:"path"`
	
	// Additional options specific to each backend
	Options map[string]interface{} `json:"options,omitempty" yaml:"options,omitempty"`
}

// BoltConfig holds BoltDB-specific configuration
type BoltConfig struct {
	// Timeout for opening the database
	Timeout time.Duration `json:"timeout" yaml:"timeout"`
	
	// NoGrowSync disables growing file size synchronization
	NoGrowSync bool `json:"no_grow_sync" yaml:"no_grow_sync"`
	
	// NoFreelistSync disables freelist synchronization
	NoFreelistSync bool `json:"no_freelist_sync" yaml:"no_freelist_sync"`
	
	// FreelistType sets the backend freelist type
	FreelistType string `json:"freelist_type" yaml:"freelist_type"`
	
	// ReadOnly opens the database in read-only mode
	ReadOnly bool `json:"read_only" yaml:"read_only"`
	
	// MmapFlags additional flags for memory mapping
	MmapFlags int `json:"mmap_flags" yaml:"mmap_flags"`
}

// BadgerConfig holds BadgerDB-specific configuration
type BadgerConfig struct {
	// Dir is the directory to store data
	Dir string `json:"dir" yaml:"dir"`
	
	// ValueDir is the directory to store values (can be same as Dir)
	ValueDir string `json:"value_dir" yaml:"value_dir"`
	
	// SyncWrites enables synchronous writes
	SyncWrites bool `json:"sync_writes" yaml:"sync_writes"`
	
	// NumVersionsToKeep sets how many versions to keep per key
	NumVersionsToKeep int `json:"num_versions_to_keep" yaml:"num_versions_to_keep"`
	
	// ReadOnly opens the database in read-only mode
	ReadOnly bool `json:"read_only" yaml:"read_only"`
	
	// Compression algorithm to use
	Compression string `json:"compression" yaml:"compression"`
	
	// InMemory creates a purely in-memory database
	InMemory bool `json:"in_memory" yaml:"in_memory"`
	
	// MemTableSize sets the maximum size of memtable
	MemTableSize int64 `json:"mem_table_size" yaml:"mem_table_size"`
	
	// BaseTableSize sets the maximum size of base level tables
	BaseTableSize int64 `json:"base_table_size" yaml:"base_table_size"`
	
	// LevelSizeMultiplier sets the ratio between consecutive levels
	LevelSizeMultiplier int `json:"level_size_multiplier" yaml:"level_size_multiplier"`
	
	// MaxLevels sets the maximum number of levels
	MaxLevels int `json:"max_levels" yaml:"max_levels"`
	
	// VLogPercentile sets the percentile for value log GC
	VLogPercentile float64 `json:"vlog_percentile" yaml:"vlog_percentile"`
}// DefaultPersistenceConfig returns a default configuration for the specified type
func DefaultPersistenceConfig(persistenceType PersistenceType, path string) PersistenceConfig {
	config := PersistenceConfig{
		Type: persistenceType,
		Path: path,
		Options: make(map[string]interface{}),
	}
	
	switch persistenceType {
	case PersistenceBolt:
		config.Options = map[string]interface{}{
			"timeout":           "1s",
			"no_grow_sync":      false,
			"no_freelist_sync":  false,
			"freelist_type":     "map",
			"read_only":         false,
			"mmap_flags":        0,
		}
	case PersistenceBadger:
		config.Options = map[string]interface{}{
			"sync_writes":             false,
			"num_versions_to_keep":    1,
			"read_only":               false,
			"compression":             "none",
			"in_memory":               false,
			"mem_table_size":          64 << 20, // 64MB
			"base_table_size":         2 << 20,  // 2MB
			"level_size_multiplier":   10,
			"max_levels":              7,
			"vlog_percentile":         0.5,
		}
	}
	
	return config
}

// ValidateConfig validates a persistence configuration
func ValidateConfig(config PersistenceConfig) error {
	switch config.Type {
	case PersistenceMemory:
		// Memory persistence doesn't need a path
		return nil
	case PersistenceBolt, PersistenceBadger:
		if config.Path == "" {
			return fmt.Errorf("path is required for %s persistence", config.Type)
		}
		return nil
	default:
		return fmt.Errorf("unsupported persistence type: %s", config.Type)
	}
}