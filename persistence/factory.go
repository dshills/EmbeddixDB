package persistence

import (
	"fmt"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// DefaultFactory implements core.PersistenceFactory
type DefaultFactory struct{}

// NewDefaultFactory creates a new default persistence factory
func NewDefaultFactory() *DefaultFactory {
	return &DefaultFactory{}
}

// CreatePersistence creates a persistence instance based on configuration
func (f *DefaultFactory) CreatePersistence(config PersistenceConfig) (core.Persistence, error) {
	if err := ValidateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid persistence configuration: %w", err)
	}

	// Create the base persistence layer
	var basePersistence core.Persistence
	var err error

	switch config.Type {
	case PersistenceMemory:
		basePersistence = NewMemoryPersistence()

	case PersistenceBolt:
		basePersistence, err = f.createBoltPersistence(config)
		if err != nil {
			return nil, err
		}

	case PersistenceBadger:
		basePersistence, err = f.createBadgerPersistence(config)
		if err != nil {
			return nil, err
		}

	default:
		return nil, fmt.Errorf("unsupported persistence type: %s", config.Type)
	}

	// Wrap with WAL if configured
	if config.WAL != nil {
		walPersistence, err := NewWALPersistence(basePersistence, *config.WAL)
		if err != nil {
			basePersistence.Close() // Clean up on error
			return nil, fmt.Errorf("failed to create WAL persistence: %w", err)
		}
		return walPersistence, nil
	}

	return basePersistence, nil
}

// createBoltPersistence creates a BoltDB persistence with configuration
func (f *DefaultFactory) createBoltPersistence(config PersistenceConfig) (core.Persistence, error) {
	// For now, use the simple constructor
	// In the future, this could apply BoltConfig options
	return NewBoltPersistence(config.Path)
}

// createBadgerPersistence creates a BadgerDB persistence with configuration
func (f *DefaultFactory) createBadgerPersistence(config PersistenceConfig) (core.Persistence, error) {
	// For now, use the simple constructor
	// In the future, this could apply BadgerConfig options
	return NewBadgerPersistence(config.Path)
}

// Helper function to parse duration from config options
func parseDuration(options map[string]interface{}, key string, defaultValue time.Duration) time.Duration {
	if val, exists := options[key]; exists {
		if str, ok := val.(string); ok {
			if duration, err := time.ParseDuration(str); err == nil {
				return duration
			}
		}
	}
	return defaultValue
}

// Helper function to parse bool from config options
func parseBool(options map[string]interface{}, key string, defaultValue bool) bool {
	if val, exists := options[key]; exists {
		if b, ok := val.(bool); ok {
			return b
		}
	}
	return defaultValue
}

// Helper function to parse int from config options
func parseInt(options map[string]interface{}, key string, defaultValue int) int {
	if val, exists := options[key]; exists {
		if i, ok := val.(int); ok {
			return i
		}
		if f, ok := val.(float64); ok {
			return int(f)
		}
	}
	return defaultValue
}

// Helper function to parse int64 from config options
func parseInt64(options map[string]interface{}, key string, defaultValue int64) int64 {
	if val, exists := options[key]; exists {
		if i, ok := val.(int64); ok {
			return i
		}
		if f, ok := val.(float64); ok {
			return int64(f)
		}
		if i, ok := val.(int); ok {
			return int64(i)
		}
	}
	return defaultValue
}

// Helper function to parse float64 from config options
func parseFloat64(options map[string]interface{}, key string, defaultValue float64) float64 {
	if val, exists := options[key]; exists {
		if f, ok := val.(float64); ok {
			return f
		}
	}
	return defaultValue
}

// Helper function to parse string from config options
func parseString(options map[string]interface{}, key string, defaultValue string) string {
	if val, exists := options[key]; exists {
		if s, ok := val.(string); ok {
			return s
		}
	}
	return defaultValue
}
