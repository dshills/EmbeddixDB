package persistence

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/dgraph-io/badger/v4"
	"github.com/dshills/EmbeddixDB/core"
)

const (
	// Key prefixes for different data types
	vectorKeyPrefix     = "v:"
	collectionKeyPrefix = "c:"
	indexStateKeyPrefix = "i:"
	metadataKeyPrefix   = "m:"
)

// BadgerPersistence implements persistence using BadgerDB
type BadgerPersistence struct {
	db   *badger.DB
	path string
}

// NewBadgerPersistence creates a new BadgerDB persistence layer
func NewBadgerPersistence(dbPath string) (*BadgerPersistence, error) {
	// Ensure directory exists
	if err := os.MkdirAll(dbPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory %s: %w", dbPath, err)
	}

	// Configure BadgerDB options
	opts := badger.DefaultOptions(dbPath)
	opts.Logger = nil // Disable logging for cleaner output

	// Open BadgerDB database
	db, err := badger.Open(opts)
	if err != nil {
		return nil, fmt.Errorf("failed to open BadgerDB at %s: %w", dbPath, err)
	}

	return &BadgerPersistence{
		db:   db,
		path: dbPath,
	}, nil
}

// makeVectorKey creates a key for storing vectors
func (b *BadgerPersistence) makeVectorKey(collection, id string) string {
	return vectorKeyPrefix + collection + ":" + id
}

// makeCollectionKey creates a key for storing collection metadata
func (b *BadgerPersistence) makeCollectionKey(name string) string {
	return collectionKeyPrefix + name
}

// parseVectorKey extracts collection and ID from a vector key
func (b *BadgerPersistence) parseVectorKey(key string) (collection, id string, valid bool) {
	if !strings.HasPrefix(key, vectorKeyPrefix) {
		return "", "", false
	}

	remainder := key[len(vectorKeyPrefix):]
	parts := strings.SplitN(remainder, ":", 2)
	if len(parts) != 2 {
		return "", "", false
	}

	return parts[0], parts[1], true
} // SaveVector stores a vector in BadgerDB
func (b *BadgerPersistence) SaveVector(ctx context.Context, collection string, vec core.Vector) error {
	if err := core.ValidateVector(vec); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}

	data, err := json.Marshal(vec)
	if err != nil {
		return fmt.Errorf("failed to marshal vector: %w", err)
	}

	key := b.makeVectorKey(collection, vec.ID)

	return b.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(key), data)
	})
}

// LoadVector retrieves a vector by ID from BadgerDB
func (b *BadgerPersistence) LoadVector(ctx context.Context, collection, id string) (core.Vector, error) {
	var vec core.Vector
	key := b.makeVectorKey(collection, id)

	err := b.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			if err == badger.ErrKeyNotFound {
				return fmt.Errorf("vector %s not found in collection %s", id, collection)
			}
			return err
		}

		return item.Value(func(val []byte) error {
			return json.Unmarshal(val, &vec)
		})
	})

	if err != nil {
		return core.Vector{}, err
	}

	return vec, nil
}

// LoadVectors retrieves all vectors from a collection
func (b *BadgerPersistence) LoadVectors(ctx context.Context, collection string) ([]core.Vector, error) {
	var vectors []core.Vector
	prefix := vectorKeyPrefix + collection + ":"

	err := b.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchSize = 10
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Seek([]byte(prefix)); it.ValidForPrefix([]byte(prefix)); it.Next() {
			item := it.Item()

			err := item.Value(func(val []byte) error {
				var vec core.Vector
				if err := json.Unmarshal(val, &vec); err != nil {
					return fmt.Errorf("failed to unmarshal vector: %w", err)
				}
				vectors = append(vectors, vec)
				return nil
			})

			if err != nil {
				return err
			}
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return vectors, nil
} // DeleteVector removes a vector from BadgerDB
func (b *BadgerPersistence) DeleteVector(ctx context.Context, collection, id string) error {
	key := b.makeVectorKey(collection, id)

	return b.db.Update(func(txn *badger.Txn) error {
		// Check if key exists first
		_, err := txn.Get([]byte(key))
		if err != nil {
			if err == badger.ErrKeyNotFound {
				return fmt.Errorf("vector %s not found in collection %s", id, collection)
			}
			return err
		}

		return txn.Delete([]byte(key))
	})
}

// SaveCollection stores collection metadata
func (b *BadgerPersistence) SaveCollection(ctx context.Context, collection core.Collection) error {
	if err := core.ValidateCollection(collection); err != nil {
		return fmt.Errorf("invalid collection: %w", err)
	}

	data, err := json.Marshal(collection)
	if err != nil {
		return fmt.Errorf("failed to marshal collection: %w", err)
	}

	key := b.makeCollectionKey(collection.Name)

	return b.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(key), data)
	})
}

// LoadCollection retrieves collection metadata
func (b *BadgerPersistence) LoadCollection(ctx context.Context, name string) (core.Collection, error) {
	var collection core.Collection
	key := b.makeCollectionKey(name)

	err := b.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			if err == badger.ErrKeyNotFound {
				return fmt.Errorf("collection %s not found", name)
			}
			return err
		}

		return item.Value(func(val []byte) error {
			return json.Unmarshal(val, &collection)
		})
	})

	if err != nil {
		return core.Collection{}, err
	}

	return collection, nil
}

// LoadCollections retrieves all collection metadata
func (b *BadgerPersistence) LoadCollections(ctx context.Context) ([]core.Collection, error) {
	var collections []core.Collection
	prefix := collectionKeyPrefix

	err := b.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchSize = 10
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Seek([]byte(prefix)); it.ValidForPrefix([]byte(prefix)); it.Next() {
			item := it.Item()

			err := item.Value(func(val []byte) error {
				var collection core.Collection
				if err := json.Unmarshal(val, &collection); err != nil {
					return fmt.Errorf("failed to unmarshal collection: %w", err)
				}
				collections = append(collections, collection)
				return nil
			})

			if err != nil {
				return err
			}
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return collections, nil
} // DeleteCollection removes a collection and all its vectors
func (b *BadgerPersistence) DeleteCollection(ctx context.Context, name string) error {
	return b.db.Update(func(txn *badger.Txn) error {
		// Check if collection exists
		collectionKey := b.makeCollectionKey(name)
		_, err := txn.Get([]byte(collectionKey))
		if err != nil {
			if err == badger.ErrKeyNotFound {
				return fmt.Errorf("collection %s not found", name)
			}
			return err
		}

		// Delete collection metadata
		if err := txn.Delete([]byte(collectionKey)); err != nil {
			return fmt.Errorf("failed to delete collection metadata: %w", err)
		}

		// Delete all vectors in the collection
		vectorPrefix := vectorKeyPrefix + name + ":"
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false // We only need keys
		it := txn.NewIterator(opts)
		defer it.Close()

		var keysToDelete [][]byte
		for it.Seek([]byte(vectorPrefix)); it.ValidForPrefix([]byte(vectorPrefix)); it.Next() {
			key := it.Item().KeyCopy(nil)
			keysToDelete = append(keysToDelete, key)
		}

		// Delete all vector keys
		for _, key := range keysToDelete {
			if err := txn.Delete(key); err != nil {
				return fmt.Errorf("failed to delete vector key %s: %w", string(key), err)
			}
		}

		return nil
	})
}

// SaveVectorsBatch stores multiple vectors efficiently in a single transaction
func (b *BadgerPersistence) SaveVectorsBatch(ctx context.Context, collection string, vectors []core.Vector) error {
	for _, vec := range vectors {
		if err := core.ValidateVector(vec); err != nil {
			return fmt.Errorf("invalid vector %s: %w", vec.ID, err)
		}
	}

	return b.db.Update(func(txn *badger.Txn) error {
		for _, vec := range vectors {
			data, err := json.Marshal(vec)
			if err != nil {
				return fmt.Errorf("failed to marshal vector %s: %w", vec.ID, err)
			}

			key := b.makeVectorKey(collection, vec.ID)
			if err := txn.Set([]byte(key), data); err != nil {
				return fmt.Errorf("failed to store vector %s: %w", vec.ID, err)
			}
		}

		return nil
	})
}

// SaveIndexState stores serialized index state in BadgerDB
func (b *BadgerPersistence) SaveIndexState(ctx context.Context, collection string, indexData []byte) error {
	key := indexStateKeyPrefix + collection

	return b.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(key), indexData)
	})
}

// LoadIndexState retrieves serialized index state from BadgerDB
func (b *BadgerPersistence) LoadIndexState(ctx context.Context, collection string) ([]byte, error) {
	key := indexStateKeyPrefix + collection
	var indexData []byte

	err := b.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			if err == badger.ErrKeyNotFound {
				return fmt.Errorf("index state for collection %s not found", collection)
			}
			return err
		}

		return item.Value(func(val []byte) error {
			// Copy the value since it's only valid during the transaction
			indexData = make([]byte, len(val))
			copy(indexData, val)
			return nil
		})
	})

	return indexData, err
}

// DeleteIndexState removes serialized index state from BadgerDB
func (b *BadgerPersistence) DeleteIndexState(ctx context.Context, collection string) error {
	key := indexStateKeyPrefix + collection

	return b.db.Update(func(txn *badger.Txn) error {
		return txn.Delete([]byte(key))
	})
}

// Close closes the BadgerDB database
func (b *BadgerPersistence) Close() error {
	if b.db != nil {
		return b.db.Close()
	}
	return nil
}

// RunGarbageCollection manually triggers BadgerDB garbage collection
func (b *BadgerPersistence) RunGarbageCollection() error {
	for {
		err := b.db.RunValueLogGC(0.5)
		if err != nil {
			if err == badger.ErrNoRewrite {
				break // No more GC needed
			}
			return fmt.Errorf("garbage collection failed: %w", err)
		}
	}
	return nil
}

// Stats returns database statistics
func (b *BadgerPersistence) Stats() (map[string]interface{}, error) {
	stats := make(map[string]interface{})

	// Get BadgerDB LSM stats
	lsmSize, vlogSize := b.db.Size()
	stats["lsm_size"] = lsmSize
	stats["vlog_size"] = vlogSize
	stats["total_size"] = lsmSize + vlogSize

	// Count collections and vectors
	err := b.db.View(func(txn *badger.Txn) error {
		collectionCount := 0
		vectorCounts := make(map[string]int)

		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			key := string(it.Item().Key())

			if strings.HasPrefix(key, collectionKeyPrefix) {
				collectionCount++
			} else if strings.HasPrefix(key, vectorKeyPrefix) {
				if collection, _, valid := b.parseVectorKey(key); valid {
					vectorCounts[collection]++
				}
			}
		}

		stats["collection_count"] = collectionCount
		stats["vector_counts"] = vectorCounts

		return nil
	})

	if err != nil {
		return nil, err
	}

	return stats, nil
}
