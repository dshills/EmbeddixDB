package persistence

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"go.etcd.io/bbolt"
)

const (
	// Bucket names for different data types
	vectorsBucketPrefix = "vectors_"
	collectionsBucket   = "collections"
	indexStatesBucket   = "index_states"
	metadataBucket      = "metadata"
)

// BoltPersistence implements persistence using BoltDB
type BoltPersistence struct {
	db   *bbolt.DB
	path string
}

// NewBoltPersistence creates a new BoltDB persistence layer
func NewBoltPersistence(dbPath string) (*BoltPersistence, error) {
	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Open BoltDB database
	db, err := bbolt.Open(dbPath, 0600, &bbolt.Options{
		Timeout: 1 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to open BoltDB at %s: %w", dbPath, err)
	}

	persistence := &BoltPersistence{
		db:   db,
		path: dbPath,
	}

	// Initialize required buckets
	if err := persistence.initBuckets(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize buckets: %w", err)
	}

	return persistence, nil
}

// initBuckets creates the required buckets if they don't exist
func (b *BoltPersistence) initBuckets() error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		// Create collections bucket
		if _, err := tx.CreateBucketIfNotExists([]byte(collectionsBucket)); err != nil {
			return fmt.Errorf("failed to create collections bucket: %w", err)
		}

		// Create index states bucket
		if _, err := tx.CreateBucketIfNotExists([]byte(indexStatesBucket)); err != nil {
			return fmt.Errorf("failed to create index states bucket: %w", err)
		}

		// Create metadata bucket
		if _, err := tx.CreateBucketIfNotExists([]byte(metadataBucket)); err != nil {
			return fmt.Errorf("failed to create metadata bucket: %w", err)
		}

		return nil
	})
} // SaveVector stores a vector in BoltDB
func (b *BoltPersistence) SaveVector(ctx context.Context, collection string, vec core.Vector) error {
	if err := core.ValidateVector(vec); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}

	data, err := json.Marshal(vec)
	if err != nil {
		return fmt.Errorf("failed to marshal vector: %w", err)
	}

	bucketName := vectorsBucketPrefix + collection

	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket, err := tx.CreateBucketIfNotExists([]byte(bucketName))
		if err != nil {
			return fmt.Errorf("failed to create/get bucket %s: %w", bucketName, err)
		}

		return bucket.Put([]byte(vec.ID), data)
	})
}

// LoadVector retrieves a vector by ID from BoltDB
func (b *BoltPersistence) LoadVector(ctx context.Context, collection, id string) (core.Vector, error) {
	var vec core.Vector
	bucketName := vectorsBucketPrefix + collection

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(bucketName))
		if bucket == nil {
			return fmt.Errorf("collection %s not found", collection)
		}

		data := bucket.Get([]byte(id))
		if data == nil {
			return fmt.Errorf("vector %s not found in collection %s", id, collection)
		}

		return json.Unmarshal(data, &vec)
	})

	if err != nil {
		return core.Vector{}, err
	}

	return vec, nil
}

// LoadVectors retrieves all vectors from a collection
func (b *BoltPersistence) LoadVectors(ctx context.Context, collection string) ([]core.Vector, error) {
	var vectors []core.Vector
	bucketName := vectorsBucketPrefix + collection

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(bucketName))
		if bucket == nil {
			return fmt.Errorf("collection %s not found", collection)
		}

		return bucket.ForEach(func(k, v []byte) error {
			var vec core.Vector
			if err := json.Unmarshal(v, &vec); err != nil {
				return fmt.Errorf("failed to unmarshal vector %s: %w", string(k), err)
			}
			vectors = append(vectors, vec)
			return nil
		})
	})

	if err != nil {
		return nil, err
	}

	return vectors, nil
} // DeleteVector removes a vector from BoltDB
func (b *BoltPersistence) DeleteVector(ctx context.Context, collection, id string) error {
	bucketName := vectorsBucketPrefix + collection

	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(bucketName))
		if bucket == nil {
			return fmt.Errorf("collection %s not found", collection)
		}

		if bucket.Get([]byte(id)) == nil {
			return fmt.Errorf("vector %s not found in collection %s", id, collection)
		}

		return bucket.Delete([]byte(id))
	})
}

// SaveCollection stores collection metadata
func (b *BoltPersistence) SaveCollection(ctx context.Context, collection core.Collection) error {
	if err := core.ValidateCollection(collection); err != nil {
		return fmt.Errorf("invalid collection: %w", err)
	}

	data, err := json.Marshal(collection)
	if err != nil {
		return fmt.Errorf("failed to marshal collection: %w", err)
	}

	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(collectionsBucket))
		return bucket.Put([]byte(collection.Name), data)
	})
}

// LoadCollection retrieves collection metadata
func (b *BoltPersistence) LoadCollection(ctx context.Context, name string) (core.Collection, error) {
	var collection core.Collection

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(collectionsBucket))
		data := bucket.Get([]byte(name))
		if data == nil {
			return fmt.Errorf("collection %s not found", name)
		}

		return json.Unmarshal(data, &collection)
	})

	if err != nil {
		return core.Collection{}, err
	}

	return collection, nil
}

// LoadCollections retrieves all collection metadata
func (b *BoltPersistence) LoadCollections(ctx context.Context) ([]core.Collection, error) {
	var collections []core.Collection

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(collectionsBucket))

		return bucket.ForEach(func(k, v []byte) error {
			var collection core.Collection
			if err := json.Unmarshal(v, &collection); err != nil {
				return fmt.Errorf("failed to unmarshal collection %s: %w", string(k), err)
			}
			collections = append(collections, collection)
			return nil
		})
	})

	if err != nil {
		return nil, err
	}

	return collections, nil
} // DeleteCollection removes a collection and all its vectors
func (b *BoltPersistence) DeleteCollection(ctx context.Context, name string) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		// Delete collection metadata
		bucket := tx.Bucket([]byte(collectionsBucket))
		if bucket.Get([]byte(name)) == nil {
			return fmt.Errorf("collection %s not found", name)
		}

		if err := bucket.Delete([]byte(name)); err != nil {
			return fmt.Errorf("failed to delete collection metadata: %w", err)
		}

		// Delete vectors bucket
		bucketName := vectorsBucketPrefix + name
		if err := tx.DeleteBucket([]byte(bucketName)); err != nil && err != bbolt.ErrBucketNotFound {
			return fmt.Errorf("failed to delete vectors bucket: %w", err)
		}

		return nil
	})
}

// SaveVectorsBatch stores multiple vectors efficiently in a single transaction
func (b *BoltPersistence) SaveVectorsBatch(ctx context.Context, collection string, vectors []core.Vector) error {
	for _, vec := range vectors {
		if err := core.ValidateVector(vec); err != nil {
			return fmt.Errorf("invalid vector %s: %w", vec.ID, err)
		}
	}

	bucketName := vectorsBucketPrefix + collection

	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket, err := tx.CreateBucketIfNotExists([]byte(bucketName))
		if err != nil {
			return fmt.Errorf("failed to create/get bucket %s: %w", bucketName, err)
		}

		for _, vec := range vectors {
			data, err := json.Marshal(vec)
			if err != nil {
				return fmt.Errorf("failed to marshal vector %s: %w", vec.ID, err)
			}

			if err := bucket.Put([]byte(vec.ID), data); err != nil {
				return fmt.Errorf("failed to store vector %s: %w", vec.ID, err)
			}
		}

		return nil
	})
}

// SaveIndexState stores serialized index state in BoltDB
func (b *BoltPersistence) SaveIndexState(ctx context.Context, collection string, indexData []byte) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(indexStatesBucket))
		if bucket == nil {
			return fmt.Errorf("index states bucket not found")
		}

		return bucket.Put([]byte(collection), indexData)
	})
}

// LoadIndexState retrieves serialized index state from BoltDB
func (b *BoltPersistence) LoadIndexState(ctx context.Context, collection string) ([]byte, error) {
	var indexData []byte

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(indexStatesBucket))
		if bucket == nil {
			return fmt.Errorf("index states bucket not found")
		}

		data := bucket.Get([]byte(collection))
		if data == nil {
			return fmt.Errorf("index state for collection %s not found", collection)
		}

		// Copy the data since it's only valid during the transaction
		indexData = make([]byte, len(data))
		copy(indexData, data)
		return nil
	})

	return indexData, err
}

// DeleteIndexState removes serialized index state from BoltDB
func (b *BoltPersistence) DeleteIndexState(ctx context.Context, collection string) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte(indexStatesBucket))
		if bucket == nil {
			return fmt.Errorf("index states bucket not found")
		}

		return bucket.Delete([]byte(collection))
	})
}

// Close closes the BoltDB database
func (b *BoltPersistence) Close() error {
	if b.db != nil {
		return b.db.Close()
	}
	return nil
}

// Stats returns database statistics
func (b *BoltPersistence) Stats() (map[string]interface{}, error) {
	stats := make(map[string]interface{})

	err := b.db.View(func(tx *bbolt.Tx) error {
		dbStats := b.db.Stats()
		stats["bolt_stats"] = dbStats

		// Count collections
		collectionCount := 0
		bucket := tx.Bucket([]byte(collectionsBucket))
		if bucket != nil {
			bucket.ForEach(func(k, v []byte) error {
				collectionCount++
				return nil
			})
		}
		stats["collection_count"] = collectionCount

		// Count vectors per collection
		vectorCounts := make(map[string]int)
		tx.ForEach(func(name []byte, bucket *bbolt.Bucket) error {
			bucketName := string(name)
			if len(bucketName) > len(vectorsBucketPrefix) &&
				bucketName[:len(vectorsBucketPrefix)] == vectorsBucketPrefix {
				collectionName := bucketName[len(vectorsBucketPrefix):]
				count := 0
				bucket.ForEach(func(k, v []byte) error {
					count++
					return nil
				})
				vectorCounts[collectionName] = count
			}
			return nil
		})
		stats["vector_counts"] = vectorCounts

		return nil
	})

	if err != nil {
		return nil, err
	}

	return stats, nil
}
