package persistence

import "github.com/dshills/EmbeddixDB/core"

// PersistenceFactory creates persistence instances based on type and configuration
type PersistenceFactory interface {
	CreatePersistence(config PersistenceConfig) (core.Persistence, error)
}
