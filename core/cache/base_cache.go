package cache

import (
	"container/list"
	"context"
	"sync"
	"time"
)

// BaseCache provides a thread-safe LRU cache implementation
type BaseCache struct {
	mu              sync.RWMutex
	capacity        int64
	maxMemory       int64
	currentSize     int64
	currentMemory   int64
	items           map[string]*list.Element
	lruList         *list.List
	stats           CacheStats
	evictionPolicy  EvictionPolicy
	cleanupInterval time.Duration
	stopCleanup     chan struct{}
	defaultTTL      time.Duration
}

// NewBaseCache creates a new base cache instance
func NewBaseCache(capacity, maxMemory int64, cleanupInterval time.Duration) *BaseCache {
	bc := &BaseCache{
		capacity:        capacity,
		maxMemory:       maxMemory,
		items:           make(map[string]*list.Element),
		lruList:         list.New(),
		cleanupInterval: cleanupInterval,
		stopCleanup:     make(chan struct{}),
		defaultTTL:      5 * time.Minute,
	}
	
	// Start cleanup goroutine
	go bc.cleanupLoop()
	
	return bc
}

// Get retrieves a value from the cache
func (bc *BaseCache) Get(ctx context.Context, key string) (interface{}, bool) {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	
	elem, exists := bc.items[key]
	if !exists {
		bc.stats.Misses++
		bc.updateHitRate()
		return nil, false
	}
	
	entry := elem.Value.(*CacheEntry)
	
	// Check if expired
	if entry.IsExpired() {
		bc.removeLocked(elem)
		bc.stats.Misses++
		bc.updateHitRate()
		return nil, false
	}
	
	// Update access statistics
	entry.Touch()
	bc.lruList.MoveToFront(elem)
	
	// Update cache statistics
	bc.stats.Hits++
	bc.updateHitRate()
	
	// Notify eviction policy if set
	if bc.evictionPolicy != nil {
		bc.evictionPolicy.OnAccess(key)
	}
	
	return entry.Value, true
}

// Set stores a value in the cache
func (bc *BaseCache) Set(ctx context.Context, key string, value interface{}, options ...CacheOption) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	
	// Apply options
	opts := &cacheOptions{
		TTL:      bc.defaultTTL,
		Priority: 0,
	}
	for _, opt := range options {
		opt(opts)
	}
	
	// Calculate size (simplified - in real implementation would use reflection or interface)
	size := int64(len(key)) + estimateSize(value)
	
	// Check if we need to evict items
	for (bc.currentSize >= bc.capacity || bc.currentMemory+size > bc.maxMemory) && bc.lruList.Len() > 0 {
		bc.evictLocked()
	}
	
	// Check if item already exists
	if elem, exists := bc.items[key]; exists {
		// Update existing item
		entry := elem.Value.(*CacheEntry)
		bc.currentMemory -= entry.Size
		
		entry.Value = value
		entry.Size = size
		entry.TTL = opts.TTL
		entry.Priority = opts.Priority
		entry.UserContext = opts.UserContext
		entry.Touch()
		
		bc.currentMemory += size
		bc.lruList.MoveToFront(elem)
	} else {
		// Add new item
		entry := &CacheEntry{
			Key:         key,
			Value:       value,
			Size:        size,
			CreatedAt:   time.Now(),
			LastAccess:  time.Now(),
			AccessCount: 1,
			TTL:         opts.TTL,
			Priority:    opts.Priority,
			UserContext: opts.UserContext,
			Metadata:    make(map[string]interface{}),
		}
		
		elem := bc.lruList.PushFront(entry)
		bc.items[key] = elem
		bc.currentSize++
		bc.currentMemory += size
	}
	
	return nil
}

// Delete removes a value from the cache
func (bc *BaseCache) Delete(ctx context.Context, key string) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	
	elem, exists := bc.items[key]
	if !exists {
		return nil
	}
	
	bc.removeLocked(elem)
	return nil
}

// Clear removes all values from the cache
func (bc *BaseCache) Clear(ctx context.Context) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	
	bc.items = make(map[string]*list.Element)
	bc.lruList = list.New()
	bc.currentSize = 0
	bc.currentMemory = 0
	bc.stats = CacheStats{LastReset: time.Now()}
	
	return nil
}

// Stats returns cache performance statistics
func (bc *BaseCache) Stats() CacheStats {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	
	stats := bc.stats
	stats.Size = bc.currentSize
	stats.MemoryUsage = bc.currentMemory
	
	return stats
}

// Size returns the current size of the cache
func (bc *BaseCache) Size() int64 {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	
	return bc.currentSize
}

// SetEvictionPolicy sets a custom eviction policy
func (bc *BaseCache) SetEvictionPolicy(policy EvictionPolicy) {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	
	bc.evictionPolicy = policy
}

// Close stops the cleanup goroutine
func (bc *BaseCache) Close() {
	close(bc.stopCleanup)
}

// evictLocked removes the least recently used item (must be called with lock held)
func (bc *BaseCache) evictLocked() {
	if bc.lruList.Len() == 0 {
		return
	}
	
	var victim *list.Element
	
	if bc.evictionPolicy != nil {
		// Use custom eviction policy
		candidates := make([]CachedItem, 0, bc.lruList.Len())
		for elem := bc.lruList.Back(); elem != nil; elem = elem.Prev() {
			entry := elem.Value.(*CacheEntry)
			candidates = append(candidates, CachedItem{
				Key:         entry.Key,
				Value:       entry.Value,
				Size:        entry.Size,
				AccessCount: entry.AccessCount,
				LastAccess:  entry.LastAccess,
				CreatedAt:   entry.CreatedAt,
				TTL:         entry.TTL,
				Priority:    entry.Priority,
				UserContext: entry.UserContext,
			})
		}
		
		selected := bc.evictionPolicy.SelectVictim(candidates)
		if elem, exists := bc.items[selected.Key]; exists {
			victim = elem
		}
	}
	
	// Default to LRU if no victim selected
	if victim == nil {
		victim = bc.lruList.Back()
	}
	
	bc.removeLocked(victim)
	bc.stats.Evictions++
}

// removeLocked removes an element from the cache (must be called with lock held)
func (bc *BaseCache) removeLocked(elem *list.Element) {
	entry := elem.Value.(*CacheEntry)
	
	bc.lruList.Remove(elem)
	delete(bc.items, entry.Key)
	bc.currentSize--
	bc.currentMemory -= entry.Size
	
	// Notify eviction policy
	if bc.evictionPolicy != nil {
		bc.evictionPolicy.OnEvict(CachedItem{
			Key:         entry.Key,
			Value:       entry.Value,
			Size:        entry.Size,
			AccessCount: entry.AccessCount,
			LastAccess:  entry.LastAccess,
			CreatedAt:   entry.CreatedAt,
			TTL:         entry.TTL,
			Priority:    entry.Priority,
			UserContext: entry.UserContext,
		})
	}
}

// cleanupLoop periodically removes expired items
func (bc *BaseCache) cleanupLoop() {
	ticker := time.NewTicker(bc.cleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			bc.cleanup()
		case <-bc.stopCleanup:
			return
		}
	}
}

// cleanup removes expired items
func (bc *BaseCache) cleanup() {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	
	var toRemove []*list.Element
	
	for elem := bc.lruList.Back(); elem != nil; elem = elem.Prev() {
		entry := elem.Value.(*CacheEntry)
		if entry.IsExpired() {
			toRemove = append(toRemove, elem)
		}
	}
	
	for _, elem := range toRemove {
		bc.removeLocked(elem)
	}
}

// updateHitRate updates the hit rate statistic
func (bc *BaseCache) updateHitRate() {
	total := bc.stats.Hits + bc.stats.Misses
	if total > 0 {
		bc.stats.HitRate = float64(bc.stats.Hits) / float64(total)
	}
}

// estimateSize provides a rough estimate of value size
func estimateSize(value interface{}) int64 {
	// Simplified size estimation
	// In production, would use more sophisticated methods
	switch v := value.(type) {
	case string:
		return int64(len(v))
	case []byte:
		return int64(len(v))
	case []float32:
		return int64(len(v) * 4)
	case []float64:
		return int64(len(v) * 8)
	default:
		// Default estimate
		return 64
	}
}