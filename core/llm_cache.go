package core

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// CacheEntry represents a cached item with metadata
type CacheEntry struct {
	Key           string
	Value         interface{}
	AccessCount   int64
	LastAccessed  time.Time
	CreatedAt     time.Time
	ExpiresAt     time.Time
	Size          int64
	AgentID       string
	ConversationID string
	Semantic      bool // Whether this is a semantic cache entry
}

// LLMCache provides context-aware caching for LLM workloads
type LLMCache struct {
	// Semantic cache for vector neighborhoods
	semanticCache    map[string]*CacheEntry
	semanticLRU      *list.List
	semanticElements map[string]*list.Element
	
	// Temporal cache for recent vectors
	temporalCache    map[string]*CacheEntry
	temporalLRU      *list.List
	temporalElements map[string]*list.Element
	
	// Agent-specific caches
	agentCaches map[string]*AgentCache
	
	// Cache configuration
	maxSemanticSize  int64
	maxTemporalSize  int64
	maxAgentCaches   int
	temporalTTL      time.Duration
	semanticTTL      time.Duration
	
	// Statistics
	stats cacheStatsInternal
	mutex sync.RWMutex
}

// AgentCache represents a per-agent LRU cache
type AgentCache struct {
	cache       map[string]*CacheEntry
	lru         *list.List
	elements    map[string]*list.Element
	maxSize     int64
	currentSize int64
	agentID     string
	mutex       sync.RWMutex
}

// CacheStats tracks cache performance metrics
type CacheStats struct {
	SemanticHits     int64
	SemanticMisses   int64
	TemporalHits     int64
	TemporalMisses   int64
	AgentHits        int64
	AgentMisses      int64
	Evictions        int64
	TotalQueries     int64
	AverageHitRatio  float64
}

// cacheStatsInternal contains the mutex for thread-safe access
type cacheStatsInternal struct {
	CacheStats
	mutex sync.RWMutex
}

// NewLLMCache creates a new LLM-aware cache system
func NewLLMCache(config CacheConfig) *LLMCache {
	return &LLMCache{
		semanticCache:    make(map[string]*CacheEntry),
		semanticLRU:      list.New(),
		semanticElements: make(map[string]*list.Element),
		temporalCache:    make(map[string]*CacheEntry),
		temporalLRU:      list.New(),
		temporalElements: make(map[string]*list.Element),
		agentCaches:      make(map[string]*AgentCache),
		maxSemanticSize:  config.MaxSemanticSize,
		maxTemporalSize:  config.MaxTemporalSize,
		maxAgentCaches:   config.MaxAgentCaches,
		temporalTTL:      config.TemporalTTL,
		semanticTTL:      config.SemanticTTL,
	}
}

// CacheConfig configures the LLM cache
type CacheConfig struct {
	MaxSemanticSize int64         // Max size for semantic cache (bytes)
	MaxTemporalSize int64         // Max size for temporal cache (bytes)
	MaxAgentCaches  int           // Max number of agent caches
	TemporalTTL     time.Duration // TTL for temporal cache entries
	SemanticTTL     time.Duration // TTL for semantic cache entries
	AgentCacheSize  int64         // Size per agent cache
}

// DefaultCacheConfig returns sensible defaults
func DefaultCacheConfig() CacheConfig {
	return CacheConfig{
		MaxSemanticSize: 100 * 1024 * 1024, // 100MB
		MaxTemporalSize: 50 * 1024 * 1024,  // 50MB
		MaxAgentCaches:  100,                // Support 100 concurrent agents
		TemporalTTL:     15 * time.Minute,   // Recent vectors stay for 15 minutes
		SemanticTTL:     2 * time.Hour,      // Semantic neighborhoods stay for 2 hours
		AgentCacheSize:  10 * 1024 * 1024,   // 10MB per agent
	}
}

// CacheSearchResults caches search results with context
func (lc *LLMCache) CacheSearchResults(query []float32, results []SearchResult, agentID, conversationID string) {
	key := lc.generateQueryKey(query)
	entry := &CacheEntry{
		Key:            key,
		Value:          results,
		AccessCount:    1,
		LastAccessed:   time.Now(),
		CreatedAt:      time.Now(),
		ExpiresAt:      time.Now().Add(lc.temporalTTL),
		Size:           lc.estimateResultsSize(results),
		AgentID:        agentID,
		ConversationID: conversationID,
		Semantic:       false,
	}
	
	lc.putTemporal(key, entry)
	
	// Also cache in agent-specific cache
	if agentID != "" {
		lc.putAgent(agentID, key, entry)
	}
}

// CacheSemanticNeighborhood caches a vector neighborhood for semantic search
func (lc *LLMCache) CacheSemanticNeighborhood(centerVector []float32, neighbors []SearchResult, radius float32) {
	key := lc.generateSemanticKey(centerVector, radius)
	entry := &CacheEntry{
		Key:          key,
		Value:        neighbors,
		AccessCount:  1,
		LastAccessed: time.Now(),
		CreatedAt:    time.Now(),
		ExpiresAt:    time.Now().Add(lc.semanticTTL),
		Size:         lc.estimateResultsSize(neighbors),
		Semantic:     true,
	}
	
	lc.putSemantic(key, entry)
}

// GetSearchResults retrieves cached search results
func (lc *LLMCache) GetSearchResults(query []float32, agentID string) ([]SearchResult, bool) {
	key := lc.generateQueryKey(query)
	
	// Try agent cache first (most specific)
	if agentID != "" {
		if results, found := lc.getAgent(agentID, key); found {
			lc.recordHit("agent")
			return results.([]SearchResult), true
		}
	}
	
	// Try temporal cache
	if results, found := lc.getTemporal(key); found {
		lc.recordHit("temporal")
		return results.([]SearchResult), true
	}
	
	// Try semantic cache (look for similar queries)
	if results, found := lc.getSemanticSimilar(query); found {
		lc.recordHit("semantic")
		return results.([]SearchResult), true
	}
	
	lc.recordMiss()
	return nil, false
}

// GetSemanticNeighborhood retrieves cached semantic neighborhood
func (lc *LLMCache) GetSemanticNeighborhood(centerVector []float32, radius float32) ([]SearchResult, bool) {
	key := lc.generateSemanticKey(centerVector, radius)
	
	if results, found := lc.getSemantic(key); found {
		lc.recordHit("semantic")
		return results.([]SearchResult), true
	}
	
	lc.recordMiss()
	return nil, false
}

// InvalidateAgent clears all cached data for a specific agent
func (lc *LLMCache) InvalidateAgent(agentID string) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	delete(lc.agentCaches, agentID)
}

// InvalidateConversation clears cached data for a specific conversation
func (lc *LLMCache) InvalidateConversation(conversationID string) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	// Remove from temporal cache
	for key, entry := range lc.temporalCache {
		if entry.ConversationID == conversationID {
			lc.removeFromTemporal(key)
		}
	}
	
	// Remove from agent caches
	for _, agentCache := range lc.agentCaches {
		agentCache.mutex.Lock()
		for key, entry := range agentCache.cache {
			if entry.ConversationID == conversationID {
				agentCache.removeEntry(key)
			}
		}
		agentCache.mutex.Unlock()
	}
}

// Cleanup removes expired entries
func (lc *LLMCache) Cleanup() {
	now := time.Now()
	
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	// Clean temporal cache
	for key, entry := range lc.temporalCache {
		if now.After(entry.ExpiresAt) {
			lc.removeFromTemporal(key)
			lc.stats.CacheStats.Evictions++
		}
	}
	
	// Clean semantic cache
	for key, entry := range lc.semanticCache {
		if now.After(entry.ExpiresAt) {
			lc.removeFromSemantic(key)
			lc.stats.CacheStats.Evictions++
		}
	}
	
	// Clean agent caches
	for _, agentCache := range lc.agentCaches {
		agentCache.cleanup(now)
	}
}

// GetStats returns cache performance statistics
func (lc *LLMCache) GetStats() CacheStats {
	lc.stats.mutex.RLock()
	defer lc.stats.mutex.RUnlock()
	
	stats := CacheStats{
		SemanticHits:   lc.stats.CacheStats.SemanticHits,
		SemanticMisses: lc.stats.CacheStats.SemanticMisses,
		TemporalHits:   lc.stats.CacheStats.TemporalHits,
		TemporalMisses: lc.stats.CacheStats.TemporalMisses,
		AgentHits:      lc.stats.CacheStats.AgentHits,
		AgentMisses:    lc.stats.CacheStats.AgentMisses,
		Evictions:      lc.stats.CacheStats.Evictions,
		TotalQueries:   lc.stats.CacheStats.TotalQueries,
	}
	
	if stats.TotalQueries > 0 {
		totalHits := stats.SemanticHits + stats.TemporalHits + stats.AgentHits
		stats.AverageHitRatio = float64(totalHits) / float64(stats.TotalQueries)
	}
	
	return stats
}

// Private methods

func (lc *LLMCache) generateQueryKey(query []float32) string {
	hasher := sha256.New()
	for _, val := range query {
		hasher.Write([]byte(fmt.Sprintf("%.6f", val)))
	}
	return hex.EncodeToString(hasher.Sum(nil))[:16] // Use first 16 chars
}

func (lc *LLMCache) generateSemanticKey(vector []float32, radius float32) string {
	hasher := sha256.New()
	hasher.Write([]byte(fmt.Sprintf("semantic_%.6f", radius)))
	for _, val := range vector {
		hasher.Write([]byte(fmt.Sprintf("%.6f", val)))
	}
	return hex.EncodeToString(hasher.Sum(nil))[:16]
}

func (lc *LLMCache) estimateResultsSize(results []SearchResult) int64 {
	// Rough estimation: each result is ~100 bytes + vector size
	size := int64(len(results)) * 100
	return size
}

func (lc *LLMCache) putTemporal(key string, entry *CacheEntry) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	// Remove existing entry if present
	if elem, exists := lc.temporalElements[key]; exists {
		lc.temporalLRU.Remove(elem)
		delete(lc.temporalCache, key)
	}
	
	// Add new entry
	elem := lc.temporalLRU.PushFront(key)
	lc.temporalElements[key] = elem
	lc.temporalCache[key] = entry
	
	// Evict if necessary
	lc.evictTemporalIfNeeded()
}

func (lc *LLMCache) getTemporal(key string) (interface{}, bool) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	entry, exists := lc.temporalCache[key]
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil, false
	}
	
	// Move to front
	if elem := lc.temporalElements[key]; elem != nil {
		lc.temporalLRU.MoveToFront(elem)
	}
	
	entry.AccessCount++
	entry.LastAccessed = time.Now()
	
	return entry.Value, true
}

func (lc *LLMCache) putSemantic(key string, entry *CacheEntry) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	if elem, exists := lc.semanticElements[key]; exists {
		lc.semanticLRU.Remove(elem)
		delete(lc.semanticCache, key)
	}
	
	elem := lc.semanticLRU.PushFront(key)
	lc.semanticElements[key] = elem
	lc.semanticCache[key] = entry
	
	lc.evictSemanticIfNeeded()
}

func (lc *LLMCache) getSemantic(key string) (interface{}, bool) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	entry, exists := lc.semanticCache[key]
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil, false
	}
	
	if elem := lc.semanticElements[key]; elem != nil {
		lc.semanticLRU.MoveToFront(elem)
	}
	
	entry.AccessCount++
	entry.LastAccessed = time.Now()
	
	return entry.Value, true
}

func (lc *LLMCache) getSemanticSimilar(query []float32) (interface{}, bool) {
	// This is a simplified implementation - in practice you'd use
	// locality-sensitive hashing or approximate nearest neighbor search
	// to find semantically similar cached queries
	
	lc.mutex.RLock()
	defer lc.mutex.RUnlock()
	
	// For now, just return false - semantic similarity matching
	// would require more sophisticated implementation
	return nil, false
}

func (lc *LLMCache) putAgent(agentID, key string, entry *CacheEntry) {
	lc.mutex.Lock()
	agentCache, exists := lc.agentCaches[agentID]
	if !exists {
		if len(lc.agentCaches) >= lc.maxAgentCaches {
			// Evict least recently used agent cache
			lc.evictOldestAgentCache()
		}
		agentCache = NewAgentCache(agentID, lc.maxTemporalSize/int64(lc.maxAgentCaches))
		lc.agentCaches[agentID] = agentCache
	}
	lc.mutex.Unlock()
	
	agentCache.Put(key, entry)
}

func (lc *LLMCache) getAgent(agentID, key string) (interface{}, bool) {
	lc.mutex.RLock()
	agentCache, exists := lc.agentCaches[agentID]
	lc.mutex.RUnlock()
	
	if !exists {
		return nil, false
	}
	
	return agentCache.Get(key)
}

func (lc *LLMCache) evictTemporalIfNeeded() {
	// Simplified size-based eviction
	for len(lc.temporalCache) > int(lc.maxTemporalSize/1000) { // Rough size check
		if elem := lc.temporalLRU.Back(); elem != nil {
			key := elem.Value.(string)
			lc.removeFromTemporal(key)
			lc.stats.CacheStats.Evictions++
		}
	}
}

func (lc *LLMCache) evictSemanticIfNeeded() {
	for len(lc.semanticCache) > int(lc.maxSemanticSize/1000) {
		if elem := lc.semanticLRU.Back(); elem != nil {
			key := elem.Value.(string)
			lc.removeFromSemantic(key)
			lc.stats.CacheStats.Evictions++
		}
	}
}

func (lc *LLMCache) evictOldestAgentCache() {
	// Find and remove the oldest agent cache
	var oldestAgent string
	oldestTime := time.Now()
	
	for agentID, cache := range lc.agentCaches {
		if cache.getOldestAccess().Before(oldestTime) {
			oldestTime = cache.getOldestAccess()
			oldestAgent = agentID
		}
	}
	
	if oldestAgent != "" {
		delete(lc.agentCaches, oldestAgent)
	}
}

func (lc *LLMCache) removeFromTemporal(key string) {
	if elem := lc.temporalElements[key]; elem != nil {
		lc.temporalLRU.Remove(elem)
		delete(lc.temporalElements, key)
	}
	delete(lc.temporalCache, key)
}

func (lc *LLMCache) removeFromSemantic(key string) {
	if elem := lc.semanticElements[key]; elem != nil {
		lc.semanticLRU.Remove(elem)
		delete(lc.semanticElements, key)
	}
	delete(lc.semanticCache, key)
}

func (lc *LLMCache) recordHit(cacheType string) {
	lc.stats.mutex.Lock()
	defer lc.stats.mutex.Unlock()
	
	switch cacheType {
	case "semantic":
		lc.stats.CacheStats.SemanticHits++
	case "temporal":
		lc.stats.CacheStats.TemporalHits++
	case "agent":
		lc.stats.CacheStats.AgentHits++
	}
	lc.stats.CacheStats.TotalQueries++
}

func (lc *LLMCache) recordMiss() {
	lc.stats.mutex.Lock()
	defer lc.stats.mutex.Unlock()
	
	lc.stats.CacheStats.SemanticMisses++
	lc.stats.CacheStats.TemporalMisses++
	lc.stats.CacheStats.AgentMisses++
	lc.stats.CacheStats.TotalQueries++
}

// AgentCache methods

// NewAgentCache creates a new agent-specific cache
func NewAgentCache(agentID string, maxSize int64) *AgentCache {
	return &AgentCache{
		cache:    make(map[string]*CacheEntry),
		lru:      list.New(),
		elements: make(map[string]*list.Element),
		maxSize:  maxSize,
		agentID:  agentID,
	}
}

// Put adds an entry to the agent cache
func (ac *AgentCache) Put(key string, entry *CacheEntry) {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()
	
	if elem, exists := ac.elements[key]; exists {
		ac.lru.Remove(elem)
		ac.currentSize -= ac.cache[key].Size
		delete(ac.cache, key)
	}
	
	elem := ac.lru.PushFront(key)
	ac.elements[key] = elem
	ac.cache[key] = entry
	ac.currentSize += entry.Size
	
	ac.evictIfNeeded()
}

// Get retrieves an entry from the agent cache
func (ac *AgentCache) Get(key string) (interface{}, bool) {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()
	
	entry, exists := ac.cache[key]
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil, false
	}
	
	if elem := ac.elements[key]; elem != nil {
		ac.lru.MoveToFront(elem)
	}
	
	entry.AccessCount++
	entry.LastAccessed = time.Now()
	
	return entry.Value, true
}

func (ac *AgentCache) removeEntry(key string) {
	if elem := ac.elements[key]; elem != nil {
		ac.lru.Remove(elem)
		delete(ac.elements, key)
	}
	if entry := ac.cache[key]; entry != nil {
		ac.currentSize -= entry.Size
	}
	delete(ac.cache, key)
}

func (ac *AgentCache) evictIfNeeded() {
	for ac.currentSize > ac.maxSize {
		if elem := ac.lru.Back(); elem != nil {
			key := elem.Value.(string)
			ac.removeEntry(key)
		} else {
			break
		}
	}
}

func (ac *AgentCache) cleanup(now time.Time) {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()
	
	for key, entry := range ac.cache {
		if now.After(entry.ExpiresAt) {
			ac.removeEntry(key)
		}
	}
}

func (ac *AgentCache) getOldestAccess() time.Time {
	ac.mutex.RLock()
	defer ac.mutex.RUnlock()
	
	oldest := time.Now()
	for _, entry := range ac.cache {
		if entry.LastAccessed.Before(oldest) {
			oldest = entry.LastAccessed
		}
	}
	return oldest
}