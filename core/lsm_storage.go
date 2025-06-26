package core

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// LSMStorage implements write-optimized storage using LSM-tree structure
type LSMStorage struct {
	// Write-ahead log for immediate writes
	wal *WriteAheadLog

	// In-memory table (memtable)
	memtable      *MemTable
	memtableMutex sync.RWMutex

	// Immutable memtables waiting to be flushed
	immutableTables []*MemTable
	immutableMutex  sync.RWMutex

	// SSTable levels
	levels      []*SSTableLevel
	levelsMutex sync.RWMutex

	// Configuration
	config LSMConfig

	// Background workers
	flushChan   chan *MemTable
	compactChan chan int // level to compact
	stopChan    chan bool
	wg          sync.WaitGroup

	// Statistics
	stats LSMStats

	// Base directory
	baseDir string
}

// LSMConfig configures the LSM storage
type LSMConfig struct {
	MemTableSizeThreshold int64         // Size to trigger memtable flush
	MaxLevels             int           // Maximum number of SSTable levels
	LevelSizeMultiplier   int           // Size multiplier between levels
	CompactionTrigger     int           // Number of SSTables to trigger compaction
	WALSyncInterval       time.Duration // WAL sync interval
	BackgroundWorkers     int           // Number of background workers
	CompressionEnabled    bool          // Enable compression for SSTables
}

// DefaultLSMConfig returns sensible defaults
func DefaultLSMConfig() LSMConfig {
	return LSMConfig{
		MemTableSizeThreshold: 64 * 1024 * 1024, // 64MB
		MaxLevels:             7,                // 7 levels
		LevelSizeMultiplier:   10,               // 10x growth per level
		CompactionTrigger:     4,                // Compact when 4+ SSTables
		WALSyncInterval:       100 * time.Millisecond,
		BackgroundWorkers:     2,
		CompressionEnabled:    true,
	}
}

// MemTable represents an in-memory sorted table
type MemTable struct {
	data      map[string]*VectorEntry
	size      int64
	createdAt time.Time
	readOnly  bool
	mutex     sync.RWMutex
}

// VectorEntry represents a stored vector with metadata
type VectorEntry struct {
	Vector    Vector
	Timestamp time.Time
	Deleted   bool
	Size      int64
}

// SSTableLevel represents a level in the LSM tree
type SSTableLevel struct {
	level   int
	tables  []*SSTable
	maxSize int64
	mutex   sync.RWMutex
}

// SSTable represents a sorted string table on disk
type SSTable struct {
	id       string
	filepath string
	size     int64
	minKey   string
	maxKey   string
	metadata SSTableMetadata
}

// SSTableMetadata contains metadata about an SSTable
type SSTableMetadata struct {
	NumEntries      int64
	CreatedAt       time.Time
	Level           int
	CompressionType string
	BloomFilter     []byte // Bloom filter for existence checks
}

// LSMStats tracks LSM storage statistics
type LSMStats struct {
	MemTableFlushes  int64
	Compactions      int64
	WritesTotal      int64
	ReadsTotal       int64
	WALSyncs         int64
	DiskBytesRead    int64
	DiskBytesWritten int64
	mutex            sync.RWMutex
}

// NewLSMStorage creates a new LSM storage instance
func NewLSMStorage(baseDir string, config LSMConfig) (*LSMStorage, error) {
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	wal, err := NewWriteAheadLog(filepath.Join(baseDir, "wal"), config.WALSyncInterval)
	if err != nil {
		return nil, fmt.Errorf("failed to create WAL: %w", err)
	}

	lsm := &LSMStorage{
		wal:         wal,
		memtable:    NewMemTable(),
		levels:      make([]*SSTableLevel, config.MaxLevels),
		config:      config,
		flushChan:   make(chan *MemTable, 100),
		compactChan: make(chan int, 100),
		stopChan:    make(chan bool),
		baseDir:     baseDir,
	}

	// Initialize levels
	for i := 0; i < config.MaxLevels; i++ {
		lsm.levels[i] = &SSTableLevel{
			level:   i,
			tables:  make([]*SSTable, 0),
			maxSize: int64(config.MemTableSizeThreshold) * int64(intPow(config.LevelSizeMultiplier, i)),
		}
	}

	// Start background workers
	for i := 0; i < config.BackgroundWorkers; i++ {
		lsm.wg.Add(1)
		go lsm.backgroundWorker()
	}

	// Load existing SSTables
	if err := lsm.loadExistingSSTables(); err != nil {
		return nil, fmt.Errorf("failed to load existing SSTables: %w", err)
	}

	return lsm, nil
}

// Put stores a vector in the LSM storage
func (lsm *LSMStorage) Put(key string, vector Vector) error {
	// Write to WAL first
	if err := lsm.wal.WriteEntry(key, vector, false); err != nil {
		return fmt.Errorf("failed to write to WAL: %w", err)
	}

	// Add to memtable
	entry := &VectorEntry{
		Vector:    vector,
		Timestamp: time.Now(),
		Deleted:   false,
		Size:      lsm.estimateVectorSize(vector),
	}

	lsm.memtableMutex.Lock()
	lsm.memtable.Put(key, entry)
	needsFlush := lsm.memtable.size >= lsm.config.MemTableSizeThreshold
	lsm.memtableMutex.Unlock()

	lsm.stats.mutex.Lock()
	lsm.stats.WritesTotal++
	lsm.stats.mutex.Unlock()

	// Trigger flush if memtable is full
	if needsFlush {
		lsm.triggerMemTableFlush()
	}

	return nil
}

// Get retrieves a vector from LSM storage
func (lsm *LSMStorage) Get(key string) (Vector, bool, error) {
	lsm.stats.mutex.Lock()
	lsm.stats.ReadsTotal++
	lsm.stats.mutex.Unlock()

	// Check memtable first
	lsm.memtableMutex.RLock()
	if entry, exists := lsm.memtable.Get(key); exists {
		lsm.memtableMutex.RUnlock()
		if entry.Deleted {
			return Vector{}, false, nil
		}
		return entry.Vector, true, nil
	}
	lsm.memtableMutex.RUnlock()

	// Check immutable memtables
	lsm.immutableMutex.RLock()
	for i := len(lsm.immutableTables) - 1; i >= 0; i-- {
		if entry, exists := lsm.immutableTables[i].Get(key); exists {
			lsm.immutableMutex.RUnlock()
			if entry.Deleted {
				return Vector{}, false, nil
			}
			return entry.Vector, true, nil
		}
	}
	lsm.immutableMutex.RUnlock()

	// Check SSTables from newest to oldest
	lsm.levelsMutex.RLock()
	defer lsm.levelsMutex.RUnlock()

	for _, level := range lsm.levels {
		level.mutex.RLock()
		for i := len(level.tables) - 1; i >= 0; i-- {
			table := level.tables[i]
			if lsm.keyInRange(key, table.minKey, table.maxKey) {
				level.mutex.RUnlock()
				if entry, exists, err := lsm.getFromSSTable(table, key); err != nil {
					return Vector{}, false, err
				} else if exists {
					if entry.Deleted {
						return Vector{}, false, nil
					}
					return entry.Vector, true, nil
				}
				level.mutex.RLock()
			}
		}
		level.mutex.RUnlock()
	}

	return Vector{}, false, nil
}

// Delete marks a vector as deleted
func (lsm *LSMStorage) Delete(key string) error {
	// Write tombstone to WAL
	if err := lsm.wal.WriteEntry(key, Vector{}, true); err != nil {
		return fmt.Errorf("failed to write tombstone to WAL: %w", err)
	}

	// Add tombstone to memtable
	entry := &VectorEntry{
		Vector:    Vector{},
		Timestamp: time.Now(),
		Deleted:   true,
		Size:      0,
	}

	lsm.memtableMutex.Lock()
	lsm.memtable.Put(key, entry)
	needsFlush := lsm.memtable.size >= lsm.config.MemTableSizeThreshold
	lsm.memtableMutex.Unlock()

	if needsFlush {
		lsm.triggerMemTableFlush()
	}

	return nil
}

// Scan iterates over vectors in key range
func (lsm *LSMStorage) Scan(startKey, endKey string, callback func(key string, vector Vector) error) error {
	// This is a simplified implementation - a full implementation would
	// need to merge data from all levels while avoiding duplicates

	// Collect all entries from memtable and immutable tables
	entries := make(map[string]*VectorEntry)

	lsm.memtableMutex.RLock()
	for key, entry := range lsm.memtable.data {
		if key >= startKey && (endKey == "" || key <= endKey) {
			entries[key] = entry
		}
	}
	lsm.memtableMutex.RUnlock()

	lsm.immutableMutex.RLock()
	for _, table := range lsm.immutableTables {
		for key, entry := range table.data {
			if key >= startKey && (endKey == "" || key <= endKey) {
				if existing, exists := entries[key]; !exists || entry.Timestamp.After(existing.Timestamp) {
					entries[key] = entry
				}
			}
		}
	}
	lsm.immutableMutex.RUnlock()

	// Sort keys and call callback
	keys := make([]string, 0, len(entries))
	for key := range entries {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		entry := entries[key]
		if !entry.Deleted {
			if err := callback(key, entry.Vector); err != nil {
				return err
			}
		}
	}

	return nil
}

// Close shuts down the LSM storage
func (lsm *LSMStorage) Close() error {
	// Stop background workers
	close(lsm.stopChan)
	lsm.wg.Wait()

	// Flush remaining memtable
	if lsm.memtable.size > 0 {
		if err := lsm.flushMemTable(lsm.memtable); err != nil {
			return fmt.Errorf("failed to flush final memtable: %w", err)
		}
	}

	// Close WAL
	return lsm.wal.Close()
}

// GetStats returns LSM storage statistics
func (lsm *LSMStorage) GetStats() LSMStats {
	lsm.stats.mutex.RLock()
	defer lsm.stats.mutex.RUnlock()

	return LSMStats{
		MemTableFlushes:  lsm.stats.MemTableFlushes,
		Compactions:      lsm.stats.Compactions,
		WritesTotal:      lsm.stats.WritesTotal,
		ReadsTotal:       lsm.stats.ReadsTotal,
		WALSyncs:         lsm.stats.WALSyncs,
		DiskBytesRead:    lsm.stats.DiskBytesRead,
		DiskBytesWritten: lsm.stats.DiskBytesWritten,
	}
}

// Private methods

func (lsm *LSMStorage) triggerMemTableFlush() {
	lsm.memtableMutex.Lock()

	// Move current memtable to immutable list
	lsm.memtable.readOnly = true

	lsm.immutableMutex.Lock()
	lsm.immutableTables = append(lsm.immutableTables, lsm.memtable)
	lsm.immutableMutex.Unlock()

	// Create new memtable
	lsm.memtable = NewMemTable()

	lsm.memtableMutex.Unlock()

	// Trigger background flush
	select {
	case lsm.flushChan <- lsm.immutableTables[len(lsm.immutableTables)-1]:
	default:
		// Channel full, flush will happen eventually
	}
}

func (lsm *LSMStorage) backgroundWorker() {
	defer lsm.wg.Done()

	for {
		select {
		case <-lsm.stopChan:
			return
		case memtable := <-lsm.flushChan:
			if err := lsm.flushMemTable(memtable); err != nil {
				// Log error but continue
				fmt.Printf("Failed to flush memtable: %v\n", err)
			}
		case level := <-lsm.compactChan:
			if err := lsm.compactLevel(level); err != nil {
				// Log error but continue
				fmt.Printf("Failed to compact level %d: %v\n", level, err)
			}
		}
	}
}

func (lsm *LSMStorage) flushMemTable(memtable *MemTable) error {
	sstable, err := lsm.writeSSTable(memtable, 0)
	if err != nil {
		return err
	}

	// Add to level 0
	lsm.levelsMutex.Lock()
	lsm.levels[0].mutex.Lock()
	lsm.levels[0].tables = append(lsm.levels[0].tables, sstable)
	needsCompaction := len(lsm.levels[0].tables) >= lsm.config.CompactionTrigger
	lsm.levels[0].mutex.Unlock()
	lsm.levelsMutex.Unlock()

	// Remove from immutable tables
	lsm.immutableMutex.Lock()
	for i, table := range lsm.immutableTables {
		if table == memtable {
			lsm.immutableTables = append(lsm.immutableTables[:i], lsm.immutableTables[i+1:]...)
			break
		}
	}
	lsm.immutableMutex.Unlock()

	lsm.stats.mutex.Lock()
	lsm.stats.MemTableFlushes++
	lsm.stats.mutex.Unlock()

	// Trigger compaction if needed
	if needsCompaction {
		select {
		case lsm.compactChan <- 0:
		default:
		}
	}

	return nil
}

func (lsm *LSMStorage) writeSSTable(memtable *MemTable, level int) (*SSTable, error) {
	// Generate SSTable ID
	id := fmt.Sprintf("sst_%d_%d", level, time.Now().UnixNano())
	filePath := filepath.Join(lsm.baseDir, fmt.Sprintf("level_%d", level), id+".sst")

	// Create directory if needed
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return nil, err
	}

	file, err := os.Create(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Write header
	header := SSTableHeader{
		Version:     1,
		NumEntries:  int64(len(memtable.data)),
		CreatedAt:   time.Now(),
		Level:       level,
		Compression: lsm.config.CompressionEnabled,
	}

	if err := lsm.writeSSTableHeader(file, header); err != nil {
		return nil, err
	}

	// Get sorted keys
	keys := make([]string, 0, len(memtable.data))
	for key := range memtable.data {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	// Write entries
	var minKey, maxKey string
	var size int64

	for i, key := range keys {
		entry := memtable.data[key]

		if i == 0 {
			minKey = key
		}
		if i == len(keys)-1 {
			maxKey = key
		}

		entrySize, err := lsm.writeSSTableEntry(file, key, entry)
		if err != nil {
			return nil, err
		}
		size += entrySize
	}

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}

	sstable := &SSTable{
		id:       id,
		filepath: filePath,
		size:     fileInfo.Size(),
		minKey:   minKey,
		maxKey:   maxKey,
		metadata: SSTableMetadata{
			NumEntries:      header.NumEntries,
			CreatedAt:       header.CreatedAt,
			Level:           level,
			CompressionType: "none", // Simplified
		},
	}

	lsm.stats.mutex.Lock()
	lsm.stats.DiskBytesWritten += fileInfo.Size()
	lsm.stats.mutex.Unlock()

	return sstable, nil
}

func (lsm *LSMStorage) compactLevel(level int) error {
	if level >= len(lsm.levels)-1 {
		return nil // Can't compact last level
	}

	lsm.stats.mutex.Lock()
	lsm.stats.Compactions++
	lsm.stats.mutex.Unlock()

	// Simplified compaction - just merge all tables in level
	// A full implementation would be more sophisticated

	return nil
}

func (lsm *LSMStorage) estimateVectorSize(vector Vector) int64 {
	size := int64(len(vector.ID) + len(vector.Values)*4) // 4 bytes per float32
	for k, v := range vector.Metadata {
		size += int64(len(k) + len(v))
	}
	return size
}

func (lsm *LSMStorage) keyInRange(key, minKey, maxKey string) bool {
	return key >= minKey && key <= maxKey
}

func (lsm *LSMStorage) getFromSSTable(table *SSTable, key string) (*VectorEntry, bool, error) {
	// Simplified implementation - would use bloom filter and binary search
	// in a real implementation

	file, err := os.Open(table.filepath)
	if err != nil {
		return nil, false, err
	}
	defer file.Close()

	lsm.stats.mutex.Lock()
	lsm.stats.DiskBytesRead += table.size
	lsm.stats.mutex.Unlock()

	// For now, just return not found
	return nil, false, nil
}

func (lsm *LSMStorage) loadExistingSSTables() error {
	// Load existing SSTables from disk
	// This is a simplified implementation
	return nil
}

// Helper types and functions

type SSTableHeader struct {
	Version     int32
	NumEntries  int64
	CreatedAt   time.Time
	Level       int
	Compression bool
}

func (lsm *LSMStorage) writeSSTableHeader(w io.Writer, header SSTableHeader) error {
	return binary.Write(w, binary.LittleEndian, header)
}

func (lsm *LSMStorage) writeSSTableEntry(w io.Writer, key string, entry *VectorEntry) (int64, error) {
	// Simplified entry writing
	keyBytes := []byte(key)

	// Write key length
	if err := binary.Write(w, binary.LittleEndian, int32(len(keyBytes))); err != nil {
		return 0, err
	}

	// Write key
	if _, err := w.Write(keyBytes); err != nil {
		return 0, err
	}

	// Write entry (simplified)
	// A real implementation would serialize the entire VectorEntry

	return int64(4 + len(keyBytes)), nil
}

// NewMemTable creates a new in-memory table
func NewMemTable() *MemTable {
	return &MemTable{
		data:      make(map[string]*VectorEntry),
		size:      0,
		createdAt: time.Now(),
		readOnly:  false,
	}
}

// Put adds an entry to the memtable
func (mt *MemTable) Put(key string, entry *VectorEntry) {
	mt.mutex.Lock()
	defer mt.mutex.Unlock()

	if existing, exists := mt.data[key]; exists {
		mt.size -= existing.Size
	}

	mt.data[key] = entry
	mt.size += entry.Size
}

// Get retrieves an entry from the memtable
func (mt *MemTable) Get(key string) (*VectorEntry, bool) {
	mt.mutex.RLock()
	defer mt.mutex.RUnlock()

	entry, exists := mt.data[key]
	return entry, exists
}

// WriteAheadLog simplified implementation
type WriteAheadLog struct {
	file  *os.File
	mutex sync.Mutex
}

func NewWriteAheadLog(filepath string, syncInterval time.Duration) (*WriteAheadLog, error) {
	file, err := os.OpenFile(filepath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}

	return &WriteAheadLog{file: file}, nil
}

func (wal *WriteAheadLog) WriteEntry(key string, vector Vector, deleted bool) error {
	wal.mutex.Lock()
	defer wal.mutex.Unlock()

	// Simplified WAL entry writing
	entry := fmt.Sprintf("%s:%v:%v\n", key, vector.ID, deleted)
	_, err := wal.file.WriteString(entry)
	return err
}

func (wal *WriteAheadLog) Close() error {
	return wal.file.Close()
}

func intPow(base, exp int) int {
	result := 1
	for exp > 0 {
		if exp%2 == 1 {
			result *= base
		}
		base *= base
		exp /= 2
	}
	return result
}
