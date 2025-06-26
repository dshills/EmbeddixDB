package persistence

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/dshills/EmbeddixDB/core"
)

// WALPersistence wraps any persistence backend with Write-Ahead Logging
type WALPersistence struct {
	mu         sync.RWMutex
	underlying core.Persistence
	wal        *WAL
	lastWALID  int64
}

// NewWALPersistence creates a new WAL-enabled persistence wrapper
func NewWALPersistence(underlying core.Persistence, walConfig WALConfig) (*WALPersistence, error) {
	wal, err := NewWAL(walConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create WAL: %w", err)
	}

	walPersistence := &WALPersistence{
		underlying: underlying,
		wal:        wal,
		lastWALID:  wal.GetLastID(),
	}

	// Attempt to recover any uncommitted WAL entries
	if err := walPersistence.RecoverFromWAL(context.Background()); err != nil {
		return nil, fmt.Errorf("WAL recovery failed: %w", err)
	}

	return walPersistence, nil
}

// SaveVector saves a vector with WAL logging
func (w *WALPersistence) SaveVector(ctx context.Context, collection string, vec core.Vector) error {
	// Serialize vector for WAL
	data, err := json.Marshal(vec)
	if err != nil {
		return fmt.Errorf("failed to marshal vector for WAL: %w", err)
	}

	// Write to WAL first
	if err := w.wal.WriteEntry(ctx, WALOpAddVector, collection, vec.ID, data); err != nil {
		return fmt.Errorf("failed to write to WAL: %w", err)
	}

	// Apply to underlying persistence
	if err := w.underlying.SaveVector(ctx, collection, vec); err != nil {
		return fmt.Errorf("failed to save vector to persistence: %w", err)
	}

	return nil
}

// LoadVector loads a vector from the underlying persistence
func (w *WALPersistence) LoadVector(ctx context.Context, collection, id string) (core.Vector, error) {
	return w.underlying.LoadVector(ctx, collection, id)
}

// LoadVectors loads all vectors from the underlying persistence
func (w *WALPersistence) LoadVectors(ctx context.Context, collection string) ([]core.Vector, error) {
	return w.underlying.LoadVectors(ctx, collection)
}

// DeleteVector deletes a vector with WAL logging
func (w *WALPersistence) DeleteVector(ctx context.Context, collection, id string) error {
	// Write deletion to WAL first
	if err := w.wal.WriteEntry(ctx, WALOpDeleteVector, collection, id, nil); err != nil {
		return fmt.Errorf("failed to write deletion to WAL: %w", err)
	}

	// Apply to underlying persistence
	if err := w.underlying.DeleteVector(ctx, collection, id); err != nil {
		return fmt.Errorf("failed to delete vector from persistence: %w", err)
	}

	return nil
}

// SaveCollection saves a collection with WAL logging
func (w *WALPersistence) SaveCollection(ctx context.Context, collection core.Collection) error {
	// Serialize collection for WAL
	data, err := json.Marshal(collection)
	if err != nil {
		return fmt.Errorf("failed to marshal collection for WAL: %w", err)
	}

	// Write to WAL first
	if err := w.wal.WriteEntry(ctx, WALOpCreateCollection, collection.Name, "", data); err != nil {
		return fmt.Errorf("failed to write collection to WAL: %w", err)
	}

	// Apply to underlying persistence
	if err := w.underlying.SaveCollection(ctx, collection); err != nil {
		return fmt.Errorf("failed to save collection to persistence: %w", err)
	}

	return nil
}

// LoadCollection loads a collection from the underlying persistence
func (w *WALPersistence) LoadCollection(ctx context.Context, name string) (core.Collection, error) {
	return w.underlying.LoadCollection(ctx, name)
}

// LoadCollections loads all collections from the underlying persistence
func (w *WALPersistence) LoadCollections(ctx context.Context) ([]core.Collection, error) {
	return w.underlying.LoadCollections(ctx)
}

// DeleteCollection deletes a collection with WAL logging
func (w *WALPersistence) DeleteCollection(ctx context.Context, name string) error {
	// Write deletion to WAL first
	if err := w.wal.WriteEntry(ctx, WALOpDeleteCollection, name, "", nil); err != nil {
		return fmt.Errorf("failed to write collection deletion to WAL: %w", err)
	}

	// Apply to underlying persistence
	if err := w.underlying.DeleteCollection(ctx, name); err != nil {
		return fmt.Errorf("failed to delete collection from persistence: %w", err)
	}

	return nil
}

// SaveIndexState saves index state with WAL logging
func (w *WALPersistence) SaveIndexState(ctx context.Context, collection string, indexData []byte) error {
	// Write to WAL first
	if err := w.wal.WriteEntry(ctx, WALOpSaveIndexState, collection, "", indexData); err != nil {
		return fmt.Errorf("failed to write index state to WAL: %w", err)
	}

	// Apply to underlying persistence
	if err := w.underlying.SaveIndexState(ctx, collection, indexData); err != nil {
		return fmt.Errorf("failed to save index state to persistence: %w", err)
	}

	return nil
}

// LoadIndexState loads index state from the underlying persistence
func (w *WALPersistence) LoadIndexState(ctx context.Context, collection string) ([]byte, error) {
	return w.underlying.LoadIndexState(ctx, collection)
}

// DeleteIndexState deletes index state with WAL logging
func (w *WALPersistence) DeleteIndexState(ctx context.Context, collection string) error {
	// Write deletion to WAL first
	if err := w.wal.WriteEntry(ctx, WALOpDeleteIndexState, collection, "", nil); err != nil {
		return fmt.Errorf("failed to write index state deletion to WAL: %w", err)
	}

	// Apply to underlying persistence
	if err := w.underlying.DeleteIndexState(ctx, collection); err != nil {
		return fmt.Errorf("failed to delete index state from persistence: %w", err)
	}

	return nil
}

// SaveVectorsBatch saves multiple vectors with WAL logging
func (w *WALPersistence) SaveVectorsBatch(ctx context.Context, collection string, vectors []core.Vector) error {
	// For batch operations, we log each vector individually
	// In a more sophisticated implementation, we could optimize this
	for _, vec := range vectors {
		if err := w.SaveVector(ctx, collection, vec); err != nil {
			return fmt.Errorf("failed to save vector %s in batch: %w", vec.ID, err)
		}
	}

	return nil
}

// RecoverFromWAL recovers any uncommitted operations from the WAL
func (w *WALPersistence) RecoverFromWAL(ctx context.Context) error {
	// Get all entries after the last processed ID
	entries, err := w.wal.ReadEntries(ctx, w.lastWALID+1)
	if err != nil {
		return fmt.Errorf("failed to read WAL entries: %w", err)
	}

	if len(entries) == 0 {
		return nil // Nothing to recover
	}

	// Apply each entry to the underlying persistence
	for _, entry := range entries {
		if err := w.applyWALEntry(ctx, entry); err != nil {
			return fmt.Errorf("failed to apply WAL entry %d: %w", entry.ID, err)
		}
		w.lastWALID = entry.ID
	}

	// Truncate applied entries from WAL
	if err := w.wal.Truncate(ctx, w.lastWALID); err != nil {
		return fmt.Errorf("failed to truncate WAL: %w", err)
	}

	return nil
}

// applyWALEntry applies a single WAL entry to the underlying persistence
func (w *WALPersistence) applyWALEntry(ctx context.Context, entry WALEntry) error {
	switch entry.Operation {
	case WALOpAddVector:
		var vec core.Vector
		if err := json.Unmarshal(entry.Data, &vec); err != nil {
			return fmt.Errorf("failed to unmarshal vector: %w", err)
		}
		return w.underlying.SaveVector(ctx, entry.Collection, vec)

	case WALOpDeleteVector:
		return w.underlying.DeleteVector(ctx, entry.Collection, entry.VectorID)

	case WALOpCreateCollection:
		var collection core.Collection
		if err := json.Unmarshal(entry.Data, &collection); err != nil {
			return fmt.Errorf("failed to unmarshal collection: %w", err)
		}
		return w.underlying.SaveCollection(ctx, collection)

	case WALOpDeleteCollection:
		return w.underlying.DeleteCollection(ctx, entry.Collection)

	case WALOpSaveIndexState:
		return w.underlying.SaveIndexState(ctx, entry.Collection, entry.Data)

	case WALOpDeleteIndexState:
		return w.underlying.DeleteIndexState(ctx, entry.Collection)

	default:
		return fmt.Errorf("unknown WAL operation: %s", entry.Operation)
	}
}

// Close closes the WAL and underlying persistence
func (w *WALPersistence) Close() error {
	var errs []error

	if err := w.wal.Close(); err != nil {
		errs = append(errs, fmt.Errorf("WAL close error: %w", err))
	}

	if err := w.underlying.Close(); err != nil {
		errs = append(errs, fmt.Errorf("underlying persistence close error: %w", err))
	}

	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}

	return nil
}

// Sync forces a sync of the WAL to disk
func (w *WALPersistence) Sync() error {
	return w.wal.Sync()
}

// GetWALStats returns statistics about the WAL
func (w *WALPersistence) GetWALStats() map[string]interface{} {
	return map[string]interface{}{
		"last_wal_id": w.lastWALID,
		"current_id":  w.wal.GetLastID(),
		"wal_path":    w.wal.path,
	}
}
