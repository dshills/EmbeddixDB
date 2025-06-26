package persistence

import (
	"fmt"

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
