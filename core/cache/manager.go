package cache

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// MultiLevelCacheManager coordinates the multi-level cache system
type MultiLevelCacheManager struct {
	queryCache     *QueryResultCache
	vectorCache    *VectorCache
	indexCache     *IndexPartitionCache
	semanticCache  SemanticCache
	memoryBudgets  map[CacheLevel]int64
	warmupStrategy WarmupStrategy
	mu             sync.RWMutex
}

// WarmupStrategy defines how to warm up caches
type WarmupStrategy interface {
	WarmupQueryCache(ctx context.Context, cache *QueryResultCache) error
	WarmupVectorCache(ctx context.Context, cache *VectorCache) error
	WarmupIndexCache(ctx context.Context, cache *IndexPartitionCache) error
}

// CacheManagerConfig configures the cache manager
type CacheManagerConfig struct {
	// L1 Query Cache
	QueryCacheSize      int64
	QueryCacheMemory    int64
	EnableSemantic      bool
	SemanticClusterCount int
	
	// L2 Vector Cache
	VectorCacheSize    int64
	VectorCacheMemory  int64
	UserHotSetSize     int
	
	// L3 Index Cache
	IndexCacheSize     int64
	IndexCacheMemory   int64
	MaxPartitions      int
	PartitionSizeMB    int64
	
	// General
	CleanupInterval time.Duration
	DefaultTTL      time.Duration
}

// DefaultCacheManagerConfig returns default configuration
func DefaultCacheManagerConfig() CacheManagerConfig {
	return CacheManagerConfig{
		// L1 Query Cache - 512MB
		QueryCacheSize:       10000,
		QueryCacheMemory:     512 * 1024 * 1024,
		EnableSemantic:       true,
		SemanticClusterCount: 100,
		
		// L2 Vector Cache - 1GB
		VectorCacheSize:   50000,
		VectorCacheMemory: 1024 * 1024 * 1024,
		UserHotSetSize:    1000,
		
		// L3 Index Cache - 2GB
		IndexCacheSize:   100,
		IndexCacheMemory: 2048 * 1024 * 1024,
		MaxPartitions:    32,
		PartitionSizeMB:  64,
		
		// General
		CleanupInterval: 5 * time.Minute,
		DefaultTTL:      5 * time.Minute,
	}
}

// NewMultiLevelCacheManager creates a new cache manager
func NewMultiLevelCacheManager(config CacheManagerConfig) *MultiLevelCacheManager {
	// Create base cache configurations
	cacheConfig := CacheConfig{
		DefaultTTL:      config.DefaultTTL,
		CleanupInterval: config.CleanupInterval,
	}
	
	// Create L1 Query Cache
	queryCache := NewQueryResultCache(
		config.QueryCacheSize,
		config.QueryCacheMemory,
		cacheConfig,
	)
	
	// Create L2 Vector Cache
	vectorCache := NewVectorCache(
		config.VectorCacheSize,
		config.VectorCacheMemory,
		cacheConfig,
		config.UserHotSetSize,
	)
	
	// Create L3 Index Cache
	indexCache := NewIndexPartitionCache(
		config.IndexCacheSize,
		config.IndexCacheMemory,
		cacheConfig,
		config.MaxPartitions,
		config.PartitionSizeMB*1024*1024,
	)
	
	// Create semantic cache if enabled
	var semanticCache SemanticCache
	if config.EnableSemantic {
		semanticCache = NewSemanticCache(
			config.QueryCacheSize,
			config.QueryCacheMemory,
			cacheConfig,
			config.SemanticClusterCount,
		)
		queryCache.SetSemanticCache(semanticCache)
	}
	
	// Set memory budgets
	memoryBudgets := map[CacheLevel]int64{
		L1QueryCache:  config.QueryCacheMemory,
		L2VectorCache: config.VectorCacheMemory,
		L3IndexCache:  config.IndexCacheMemory,
	}
	
	return &MultiLevelCacheManager{
		queryCache:    queryCache,
		vectorCache:   vectorCache,
		indexCache:    indexCache,
		semanticCache: semanticCache,
		memoryBudgets: memoryBudgets,
	}
}

// Get attempts to retrieve from all cache levels
func (m *MultiLevelCacheManager) Get(ctx context.Context, key string) (interface{}, CacheLevel, bool) {
	// Try L1 Query Cache first
	if value, found := m.queryCache.Get(ctx, key); found {
		return value, L1QueryCache, true
	}
	
	// Try L2 Vector Cache
	if value, found := m.vectorCache.Get(ctx, key); found {
		// Optionally promote to L1
		m.promoteToL1(ctx, key, value)
		return value, L2VectorCache, true
	}
	
	// Try L3 Index Cache
	if value, found := m.indexCache.Get(ctx, key); found {
		return value, L3IndexCache, true
	}
	
	return nil, -1, false
}

// Set stores in the appropriate cache level
func (m *MultiLevelCacheManager) Set(ctx context.Context, key string, value interface{}, options ...CacheOption) error {
	// Determine appropriate cache level based on value type
	switch v := value.(type) {
	case *CachedQueryResult:
		return m.queryCache.Set(ctx, key, v, options...)
		
	case *CachedVector:
		return m.vectorCache.Set(ctx, key, v, options...)
		
	case IndexPartition:
		return m.indexCache.Set(ctx, key, v, options...)
		
	default:
		// Default to L1
		return m.queryCache.Set(ctx, key, value, options...)
	}
}

// GetQueryResult retrieves cached query results with semantic matching
func (m *MultiLevelCacheManager) GetQueryResult(ctx context.Context, key QueryCacheKey, semanticThreshold float64) ([]SearchResult, bool) {
	// Try exact match first
	if results, found := m.queryCache.GetQueryResult(ctx, key); found {
		return results, true
	}
	
	// Try semantic matching if enabled
	if m.semanticCache != nil && len(key.QueryVector) > 0 {
		similarResults, err := m.queryCache.GetSemanticallySimilar(
			ctx,
			key.QueryVector,
			semanticThreshold,
			key.UserID,
		)
		
		if err == nil && len(similarResults) > 0 {
			// Return the most similar result
			return similarResults[0].Results, true
		}
	}
	
	return nil, false
}

// SetQueryResult stores query results with semantic indexing
func (m *MultiLevelCacheManager) SetQueryResult(ctx context.Context, key QueryCacheKey, results []SearchResult, ttl time.Duration) error {
	// Store in query cache
	err := m.queryCache.SetQueryResult(ctx, key, results, ttl)
	if err != nil {
		return err
	}
	
	// Add to semantic cache if enabled
	if m.semanticCache != nil && len(key.QueryVector) > 0 {
		hash := key.Hash()
		embedding := &SimpleVector{values: key.QueryVector}
		
		cachedResult := CachedResult{
			QueryHash:   hash,
			Results:     []interface{}{results},
			Timestamp:   time.Now(),
			UserContext: key.UserID,
			Confidence:  1.0,
		}
		
		return m.semanticCache.AddWithEmbedding(ctx, hash, embedding, cachedResult)
	}
	
	return nil
}

// GetVector retrieves a cached vector
func (m *MultiLevelCacheManager) GetVector(ctx context.Context, key VectorCacheKey, userID string) (*CachedVector, bool) {
	return m.vectorCache.GetVector(ctx, key, userID)
}

// SetVector stores a vector in cache
func (m *MultiLevelCacheManager) SetVector(ctx context.Context, key VectorCacheKey, vector *CachedVector, computeCost float64, userID string) error {
	return m.vectorCache.SetVector(ctx, key, vector, computeCost, userID)
}

// GetPartition retrieves a cached index partition
func (m *MultiLevelCacheManager) GetPartition(ctx context.Context, key IndexCacheKey) (interface{}, bool) {
	return m.indexCache.GetPartition(ctx, key)
}

// SetPartition stores an index partition
func (m *MultiLevelCacheManager) SetPartition(ctx context.Context, key IndexCacheKey, partition interface{}, loadTime time.Duration, vectorCount int) error {
	return m.indexCache.SetPartition(ctx, key, partition, loadTime, vectorCount)
}

// WarmUp pre-loads frequently accessed items
func (m *MultiLevelCacheManager) WarmUp(ctx context.Context) error {
	if m.warmupStrategy == nil {
		return nil
	}
	
	// Warm up caches in parallel
	var wg sync.WaitGroup
	errors := make(chan error, 3)
	
	// Warm up query cache
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := m.warmupStrategy.WarmupQueryCache(ctx, m.queryCache); err != nil {
			errors <- fmt.Errorf("query cache warmup failed: %w", err)
		}
	}()
	
	// Warm up vector cache
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := m.warmupStrategy.WarmupVectorCache(ctx, m.vectorCache); err != nil {
			errors <- fmt.Errorf("vector cache warmup failed: %w", err)
		}
	}()
	
	// Warm up index cache
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := m.warmupStrategy.WarmupIndexCache(ctx, m.indexCache); err != nil {
			errors <- fmt.Errorf("index cache warmup failed: %w", err)
		}
	}()
	
	wg.Wait()
	close(errors)
	
	// Collect any errors
	for err := range errors {
		return err
	}
	
	return nil
}

// Optimize redistributes items between cache levels
func (m *MultiLevelCacheManager) Optimize(ctx context.Context) error {
	// Analyze cache statistics
	stats := m.GetStats()
	
	// Identify optimization opportunities
	// For example, if L1 has low hit rate but L2 has high hit rate,
	// we might want to promote more items from L2 to L1
	
	l1Stats := stats[L1QueryCache]
	l2Stats := stats[L2VectorCache]
	
	if l1Stats.HitRate < 0.3 && l2Stats.HitRate > 0.7 {
		// Promote hot vectors to query cache
		_ = m.vectorCache.GetUserHotVectors("", 100)
		// Implementation would promote these vectors
	}
	
	// Clean up cold partitions
	return m.indexCache.CleanupColdPartitions(ctx, 24*time.Hour)
}

// GetStats returns statistics for all cache levels
func (m *MultiLevelCacheManager) GetStats() map[CacheLevel]CacheStats {
	return map[CacheLevel]CacheStats{
		L1QueryCache:  m.queryCache.Stats(),
		L2VectorCache: m.vectorCache.Stats(),
		L3IndexCache:  m.indexCache.Stats(),
	}
}

// SetMemoryBudget sets memory limits for each level
func (m *MultiLevelCacheManager) SetMemoryBudget(budgets map[CacheLevel]int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.memoryBudgets = budgets
	
	// Apply budgets to individual caches
	// This would require adding SetMaxMemory methods to each cache
	
	return nil
}

// GetLevel returns a specific cache level
func (m *MultiLevelCacheManager) GetLevel(level CacheLevel) Cache {
	switch level {
	case L1QueryCache:
		return m.queryCache
	case L2VectorCache:
		return m.vectorCache
	case L3IndexCache:
		return m.indexCache
	default:
		return nil
	}
}

// SetEvictionPolicy sets the eviction policy for a cache level
func (m *MultiLevelCacheManager) SetEvictionPolicy(level CacheLevel, policy EvictionPolicy) error {
	cache := m.GetLevel(level)
	if cache == nil {
		return fmt.Errorf("invalid cache level: %v", level)
	}
	
	// Would need to add SetEvictionPolicy to Cache interface
	// For now, return nil
	return nil
}

// Promote moves an item up in the cache hierarchy
func (m *MultiLevelCacheManager) Promote(ctx context.Context, key string, fromLevel, toLevel CacheLevel) error {
	if fromLevel <= toLevel {
		return fmt.Errorf("can only promote to higher cache levels")
	}
	
	// Get value from source level
	fromCache := m.GetLevel(fromLevel)
	if fromCache == nil {
		return fmt.Errorf("invalid source cache level")
	}
	
	value, found := fromCache.Get(ctx, key)
	if !found {
		return fmt.Errorf("key not found in source cache")
	}
	
	// Set in destination level
	toCache := m.GetLevel(toLevel)
	if toCache == nil {
		return fmt.Errorf("invalid destination cache level")
	}
	
	return toCache.Set(ctx, key, value)
}

// Demote moves an item down in the cache hierarchy
func (m *MultiLevelCacheManager) Demote(ctx context.Context, key string, fromLevel, toLevel CacheLevel) error {
	if fromLevel >= toLevel {
		return fmt.Errorf("can only demote to lower cache levels")
	}
	
	return m.Promote(ctx, key, fromLevel, toLevel)
}

// promoteToL1 promotes a value to L1 cache
func (m *MultiLevelCacheManager) promoteToL1(ctx context.Context, key string, value interface{}) {
	// Promote with shorter TTL
	m.queryCache.Set(ctx, key, value, WithTTL(1*time.Minute))
}

// InvalidateUser invalidates all cache entries for a user
func (m *MultiLevelCacheManager) InvalidateUser(ctx context.Context, userID string) error {
	// Invalidate in all cache levels
	if err := m.queryCache.InvalidateUser(ctx, userID); err != nil {
		return err
	}
	
	// Vector cache doesn't have user-specific invalidation yet
	// Would need to add this method
	
	return nil
}

// InvalidateCollection invalidates all cache entries for a collection
func (m *MultiLevelCacheManager) InvalidateCollection(ctx context.Context, collection string) error {
	return m.queryCache.InvalidateCollection(ctx, collection)
}

// SetWarmupStrategy sets the warmup strategy
func (m *MultiLevelCacheManager) SetWarmupStrategy(strategy WarmupStrategy) {
	m.warmupStrategy = strategy
}

// Close shuts down all cache levels
func (m *MultiLevelCacheManager) Close() {
	m.queryCache.Close()
	m.vectorCache.Close()
	m.indexCache.Close()
	
	if m.semanticCache != nil {
		if sc, ok := m.semanticCache.(*SemanticCacheImpl); ok {
			sc.Close()
		}
	}
}

// IndexPartition represents a cached index partition
type IndexPartition interface {
	GetID() string
	GetVectorCount() int
	GetSizeBytes() int64
}