package persistence

import "github.com/dshills/EmbeddixDB/core"

// PersistenceFactory creates persistence instances based on type and configuration
type PersistenceFactory interface {
	CreatePersistence(persistenceType string, config map[string]interface{}) (core.Persistence, error)
}

// PersistenceConfig holds configuration for persistence creation
type PersistenceConfig struct {
	Type    string
	Options map[string]interface{}
}