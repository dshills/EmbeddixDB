package persistence

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// WALOperationType represents the type of operation in the WAL
type WALOperationType string

const (
	WALOpAddVector       WALOperationType = "add_vector"
	WALOpDeleteVector    WALOperationType = "delete_vector"
	WALOpCreateCollection WALOperationType = "create_collection"
	WALOpDeleteCollection WALOperationType = "delete_collection"
	WALOpSaveIndexState  WALOperationType = "save_index_state"
	WALOpDeleteIndexState WALOperationType = "delete_index_state"
)

// WALEntry represents a single entry in the Write-Ahead Log
type WALEntry struct {
	ID         int64            `json:"id"`
	Timestamp  time.Time        `json:"timestamp"`
	Operation  WALOperationType `json:"operation"`
	Collection string           `json:"collection"`
	VectorID   string           `json:"vector_id,omitempty"`
	Data       []byte           `json:"data,omitempty"`
	Checksum   uint32           `json:"checksum"`
}

// WAL implements Write-Ahead Logging for persistence operations
type WAL struct {
	mu       sync.RWMutex
	file     *os.File
	writer   *bufio.Writer
	path     string
	nextID   int64
	entries  []WALEntry // In-memory cache of recent entries
	maxSize  int64      // Maximum WAL file size before rotation
	syncMode bool       // Whether to fsync after each write
}

// WALConfig contains configuration for the WAL
type WALConfig struct {
	Path     string // Directory to store WAL files
	MaxSize  int64  // Maximum size before rotation (default: 100MB)
	SyncMode bool   // Sync after each write (default: true for durability)
}

// NewWAL creates a new Write-Ahead Log
func NewWAL(config WALConfig) (*WAL, error) {
	if config.MaxSize == 0 {
		config.MaxSize = 100 * 1024 * 1024 // 100MB default
	}
	
	// Ensure WAL directory exists
	if err := os.MkdirAll(config.Path, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}
	
	walPath := filepath.Join(config.Path, "wal.log")
	
	// Open or create WAL file
	file, err := os.OpenFile(walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL file: %w", err)
	}
	
	wal := &WAL{
		file:     file,
		writer:   bufio.NewWriter(file),
		path:     walPath,
		nextID:   1,
		entries:  make([]WALEntry, 0),
		maxSize:  config.MaxSize,
		syncMode: config.SyncMode,
	}
	
	// Determine next ID by reading existing entries
	if err := wal.loadExistingEntries(); err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to load existing WAL entries: %w", err)
	}
	
	return wal, nil
}

// loadExistingEntries reads the WAL file to determine the next ID
func (w *WAL) loadExistingEntries() error {
	// Open file for reading
	readFile, err := os.Open(w.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // New WAL file
		}
		return err
	}
	defer readFile.Close()
	
	scanner := bufio.NewScanner(readFile)
	maxID := int64(0)
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		var entry WALEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			// Skip malformed entries but continue
			continue
		}
		
		if entry.ID > maxID {
			maxID = entry.ID
		}
	}
	
	w.nextID = maxID + 1
	return scanner.Err()
}

// WriteEntry writes a new entry to the WAL
func (w *WAL) WriteEntry(ctx context.Context, operation WALOperationType, collection, vectorID string, data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	
	entry := WALEntry{
		ID:         w.nextID,
		Timestamp:  time.Now(),
		Operation:  operation,
		Collection: collection,
		VectorID:   vectorID,
		Data:       data,
		Checksum:   w.calculateChecksum(data),
	}
	
	// Serialize entry to JSON
	entryData, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal WAL entry: %w", err)
	}
	
	// Write to file (one entry per line)
	if _, err := w.writer.Write(entryData); err != nil {
		return fmt.Errorf("failed to write WAL entry: %w", err)
	}
	
	if _, err := w.writer.WriteString("\n"); err != nil {
		return fmt.Errorf("failed to write WAL newline: %w", err)
	}
	
	// Flush buffer
	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush WAL buffer: %w", err)
	}
	
	// Sync to disk if enabled
	if w.syncMode {
		if err := w.file.Sync(); err != nil {
			return fmt.Errorf("failed to sync WAL to disk: %w", err)
		}
	}
	
	// Add to in-memory cache
	w.entries = append(w.entries, entry)
	w.nextID++
	
	// Check if rotation is needed
	if err := w.checkRotation(); err != nil {
		return fmt.Errorf("WAL rotation check failed: %w", err)
	}
	
	return nil
}

// calculateChecksum calculates a simple checksum for data integrity
func (w *WAL) calculateChecksum(data []byte) uint32 {
	var checksum uint32
	for _, b := range data {
		checksum = checksum*31 + uint32(b)
	}
	return checksum
}

// checkRotation checks if the WAL file needs to be rotated
func (w *WAL) checkRotation() error {
	stat, err := w.file.Stat()
	if err != nil {
		return err
	}
	
	if stat.Size() >= w.maxSize {
		return w.rotate()
	}
	
	return nil
}

// rotate creates a new WAL file and archives the current one
func (w *WAL) rotate() error {
	// Close current file
	w.writer.Flush()
	w.file.Close()
	
	// Archive current file with timestamp
	timestamp := time.Now().Format("20060102-150405")
	archivePath := fmt.Sprintf("%s.%s", w.path, timestamp)
	
	if err := os.Rename(w.path, archivePath); err != nil {
		return fmt.Errorf("failed to archive WAL file: %w", err)
	}
	
	// Create new file
	file, err := os.OpenFile(w.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to create new WAL file: %w", err)
	}
	
	w.file = file
	w.writer = bufio.NewWriter(file)
	
	return nil
}

// ReadEntries reads WAL entries for recovery purposes
func (w *WAL) ReadEntries(ctx context.Context, fromID int64) ([]WALEntry, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	
	var entries []WALEntry
	
	// First check in-memory cache
	for _, entry := range w.entries {
		if entry.ID >= fromID {
			entries = append(entries, entry)
		}
	}
	
	// If we need older entries, read from file
	if len(entries) == 0 || (len(entries) > 0 && entries[0].ID > fromID) {
		fileEntries, err := w.readEntriesFromFile(fromID)
		if err != nil {
			return nil, err
		}
		entries = append(fileEntries, entries...)
	}
	
	return entries, nil
}

// readEntriesFromFile reads entries from the WAL file
func (w *WAL) readEntriesFromFile(fromID int64) ([]WALEntry, error) {
	file, err := os.Open(w.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No WAL file yet
		}
		return nil, err
	}
	defer file.Close()
	
	var entries []WALEntry
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		var entry WALEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			// Skip malformed entries but continue
			continue
		}
		
		if entry.ID >= fromID {
			entries = append(entries, entry)
		}
	}
	
	return entries, scanner.Err()
}

// GetLastID returns the ID of the last written entry
func (w *WAL) GetLastID() int64 {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.nextID - 1
}

// Truncate removes entries up to the specified ID (after successful application)
func (w *WAL) Truncate(ctx context.Context, upToID int64) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	
	// Remove from in-memory cache
	newEntries := make([]WALEntry, 0)
	for _, entry := range w.entries {
		if entry.ID > upToID {
			newEntries = append(newEntries, entry)
		}
	}
	w.entries = newEntries
	
	// For file truncation, we could implement a more sophisticated approach
	// For now, we rely on rotation to keep file sizes manageable
	
	return nil
}

// Close closes the WAL and ensures all data is flushed
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	
	if w.writer != nil {
		w.writer.Flush()
	}
	
	if w.file != nil {
		return w.file.Close()
	}
	
	return nil
}

// Sync forces a sync of the WAL to disk
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	
	if err := w.writer.Flush(); err != nil {
		return err
	}
	
	return w.file.Sync()
}