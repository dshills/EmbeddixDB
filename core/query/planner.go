package query

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// QueryPlan represents an optimized execution plan for a search query
type QueryPlan struct {
	ID             string
	QueryHash      string
	Collection     string
	IndexType      string
	EstimatedCost  float64
	OptimalK       int
	UseFastPath    bool
	ParallelDegree int
	CacheHint      bool
	CreatedAt      time.Time
	LastUsed       time.Time
	ExecutionCount int64
	AvgLatency     time.Duration
}

// QueryPlanner creates and caches optimized query execution plans
type QueryPlanner struct {
	mu              sync.RWMutex
	planCache       map[string]*QueryPlan
	maxCacheSize    int
	adaptiveMetrics *AdaptiveMetrics
	evictionPolicy  EvictionPolicy
}

// AdaptiveMetrics tracks performance metrics for adaptive optimization
type AdaptiveMetrics struct {
	mu                 sync.RWMutex
	collectionStats    map[string]*CollectionStats
	queryPatterns      map[string]*QueryPattern
	performanceHistory []PerformanceRecord
	maxHistorySize     int
}

// CollectionStats holds statistics about a collection
type CollectionStats struct {
	Name             string
	VectorCount      int
	Dimension        int
	IndexType        string
	AvgSearchLatency time.Duration
	UpdateFrequency  float64 // Updates per second
	LastUpdated      time.Time
}

// QueryPattern represents a pattern of queries
type QueryPattern struct {
	PatternID      string
	QueryHash      string
	Frequency      int64
	AvgK           int
	AvgResultCount int
	CacheHitRate   float64
	LastSeen       time.Time
}

// PerformanceRecord captures performance data for a query execution
type PerformanceRecord struct {
	QueryHash   string
	Collection  string
	Latency     time.Duration
	ResultCount int
	K           int
	Timestamp   time.Time
}

// EvictionPolicy defines how to evict plans from cache
type EvictionPolicy interface {
	ShouldEvict(plan *QueryPlan, cache map[string]*QueryPlan) bool
	SelectVictim(cache map[string]*QueryPlan) string
}

// LRUEvictionPolicy implements least recently used eviction
type LRUEvictionPolicy struct {
	maxAge time.Duration
}

// NewQueryPlanner creates a new query planner with adaptive optimization
func NewQueryPlanner(maxCacheSize int) *QueryPlanner {
	return &QueryPlanner{
		planCache:    make(map[string]*QueryPlan),
		maxCacheSize: maxCacheSize,
		adaptiveMetrics: &AdaptiveMetrics{
			collectionStats:    make(map[string]*CollectionStats),
			queryPatterns:      make(map[string]*QueryPattern),
			performanceHistory: make([]PerformanceRecord, 0, 10000),
			maxHistorySize:     10000,
		},
		evictionPolicy: &LRUEvictionPolicy{maxAge: 1 * time.Hour},
	}
}

// GetPlan retrieves or creates an optimized query plan
func (qp *QueryPlanner) GetPlan(collection string, req SearchRequest, stats *CollectionStats) (*QueryPlan, error) {
	queryHash := qp.hashQuery(collection, req)

	// Check cache first
	qp.mu.RLock()
	if plan, exists := qp.planCache[queryHash]; exists {
		plan.LastUsed = time.Now()
		plan.ExecutionCount++
		qp.mu.RUnlock()
		return plan, nil
	}
	qp.mu.RUnlock()

	// Create new plan
	plan := qp.createPlan(queryHash, collection, req, stats)

	// Cache the plan
	qp.mu.Lock()
	defer qp.mu.Unlock()

	// Check cache size and evict if necessary
	if len(qp.planCache) >= qp.maxCacheSize {
		victim := qp.evictionPolicy.SelectVictim(qp.planCache)
		delete(qp.planCache, victim)
	}

	qp.planCache[queryHash] = plan
	return plan, nil
}

// createPlan creates a new optimized query plan based on collection stats and query patterns
func (qp *QueryPlanner) createPlan(queryHash, collection string, req SearchRequest, stats *CollectionStats) *QueryPlan {
	plan := &QueryPlan{
		ID:             fmt.Sprintf("plan_%s_%d", queryHash[:8], time.Now().UnixNano()),
		QueryHash:      queryHash,
		Collection:     collection,
		IndexType:      stats.IndexType,
		OptimalK:       qp.calculateOptimalK(req.TopK, stats),
		CreatedAt:      time.Now(),
		LastUsed:       time.Now(),
		ExecutionCount: 0,
	}

	// Determine if we can use fast path
	plan.UseFastPath = qp.canUseFastPath(req, stats)

	// Calculate parallel degree based on collection size and query complexity
	plan.ParallelDegree = qp.calculateParallelDegree(stats)

	// Estimate query cost
	plan.EstimatedCost = qp.estimateQueryCost(req, stats)

	// Determine if results should be cached
	plan.CacheHint = qp.shouldCacheResults(queryHash, stats)

	return plan
}

// hashQuery creates a unique hash for a query
func (qp *QueryPlanner) hashQuery(collection string, req SearchRequest) string {
	h := sha256.New()
	h.Write([]byte(collection))
	h.Write([]byte(fmt.Sprintf("%v", req.Query)))
	h.Write([]byte(fmt.Sprintf("%d", req.TopK)))

	// Include filter in hash if present
	if req.Filter != nil {
		h.Write([]byte(fmt.Sprintf("%v", req.Filter)))
	}

	return hex.EncodeToString(h.Sum(nil))
}

// calculateOptimalK determines the optimal K value based on statistics
func (qp *QueryPlanner) calculateOptimalK(requestedK int, stats *CollectionStats) int {
	// For small collections, use requested K
	if stats.VectorCount < 1000 {
		return requestedK
	}

	// For frequently updated collections, use slightly larger K to account for changes
	if stats.UpdateFrequency > 10.0 { // More than 10 updates per second
		return int(float64(requestedK) * 1.2)
	}

	// For large collections with HNSW index, we might need to search more to ensure quality
	if stats.IndexType == "hnsw" && stats.VectorCount > 100000 {
		return int(float64(requestedK) * 1.1)
	}

	return requestedK
}

// canUseFastPath determines if we can use optimized fast path
func (qp *QueryPlanner) canUseFastPath(req SearchRequest, stats *CollectionStats) bool {
	// Fast path for small K on flat index
	if stats.IndexType == "flat" && req.TopK <= 10 && stats.VectorCount < 10000 {
		return true
	}

	// Fast path for no filter queries on HNSW
	if stats.IndexType == "hnsw" && req.Filter == nil {
		return true
	}

	return false
}

// calculateParallelDegree determines optimal parallelism
func (qp *QueryPlanner) calculateParallelDegree(stats *CollectionStats) int {
	// Base on collection size
	if stats.VectorCount < 10000 {
		return 1 // No parallelism for small collections
	} else if stats.VectorCount < 100000 {
		return 2
	} else if stats.VectorCount < 1000000 {
		return 4
	}
	return 8 // Max parallelism
}

// estimateQueryCost estimates the computational cost of a query
func (qp *QueryPlanner) estimateQueryCost(req SearchRequest, stats *CollectionStats) float64 {
	baseCost := float64(stats.VectorCount) * float64(stats.Dimension) * 0.001

	// Adjust for index type
	if stats.IndexType == "hnsw" {
		baseCost *= 0.1 // HNSW is much faster
	}

	// Adjust for filters
	if req.Filter != nil {
		baseCost *= 1.5 // Filters add overhead
	}

	// Adjust for K
	baseCost *= float64(req.TopK) * 0.1

	return baseCost
}

// shouldCacheResults determines if query results should be cached
func (qp *QueryPlanner) shouldCacheResults(queryHash string, stats *CollectionStats) bool {
	qp.adaptiveMetrics.mu.RLock()
	pattern, exists := qp.adaptiveMetrics.queryPatterns[queryHash]
	qp.adaptiveMetrics.mu.RUnlock()

	if !exists {
		return false // Don't cache first-time queries
	}

	// Cache frequently accessed queries
	if pattern.Frequency > 10 {
		return true
	}

	// Cache queries with high cache hit rate
	if pattern.CacheHitRate > 0.5 {
		return true
	}

	// Don't cache queries on frequently updated collections
	if stats.UpdateFrequency > 1.0 {
		return false
	}

	return false
}

// RecordExecution records the execution of a query plan
func (qp *QueryPlanner) RecordExecution(plan *QueryPlan, latency time.Duration, resultCount int) {
	qp.mu.Lock()
	if p, exists := qp.planCache[plan.QueryHash]; exists {
		// Update average latency
		totalLatency := p.AvgLatency * time.Duration(p.ExecutionCount)
		p.ExecutionCount++
		p.AvgLatency = (totalLatency + latency) / time.Duration(p.ExecutionCount)
		p.LastUsed = time.Now()
	}
	qp.mu.Unlock()

	// Record in adaptive metrics
	qp.adaptiveMetrics.RecordPerformance(PerformanceRecord{
		QueryHash:   plan.QueryHash,
		Collection:  plan.Collection,
		Latency:     latency,
		ResultCount: resultCount,
		K:           plan.OptimalK,
		Timestamp:   time.Now(),
	})
}

// UpdateCollectionStats updates statistics for a collection
func (qp *QueryPlanner) UpdateCollectionStats(stats *CollectionStats) {
	qp.adaptiveMetrics.mu.Lock()
	defer qp.adaptiveMetrics.mu.Unlock()

	qp.adaptiveMetrics.collectionStats[stats.Name] = stats
}

// GetCollectionStats retrieves statistics for a collection
func (qp *QueryPlanner) GetCollectionStats(collection string) (*CollectionStats, bool) {
	qp.adaptiveMetrics.mu.RLock()
	defer qp.adaptiveMetrics.mu.RUnlock()

	stats, exists := qp.adaptiveMetrics.collectionStats[collection]
	return stats, exists
}

// RecordPerformance records a performance metric
func (am *AdaptiveMetrics) RecordPerformance(record PerformanceRecord) {
	am.mu.Lock()
	defer am.mu.Unlock()

	// Add to history
	am.performanceHistory = append(am.performanceHistory, record)

	// Trim history if too large
	if len(am.performanceHistory) > am.maxHistorySize {
		am.performanceHistory = am.performanceHistory[1:]
	}

	// Update query pattern
	pattern, exists := am.queryPatterns[record.QueryHash]
	if !exists {
		pattern = &QueryPattern{
			PatternID: record.QueryHash,
			QueryHash: record.QueryHash,
		}
		am.queryPatterns[record.QueryHash] = pattern
	}

	pattern.Frequency++
	pattern.LastSeen = record.Timestamp
	pattern.AvgK = (pattern.AvgK*int(pattern.Frequency-1) + record.K) / int(pattern.Frequency)
	pattern.AvgResultCount = (pattern.AvgResultCount*int(pattern.Frequency-1) + record.ResultCount) / int(pattern.Frequency)
}

// ShouldEvict checks if a plan should be evicted
func (lru *LRUEvictionPolicy) ShouldEvict(plan *QueryPlan, cache map[string]*QueryPlan) bool {
	return time.Since(plan.LastUsed) > lru.maxAge
}

// SelectVictim selects a plan to evict from cache
func (lru *LRUEvictionPolicy) SelectVictim(cache map[string]*QueryPlan) string {
	var victim string
	var oldestTime time.Time

	for hash, plan := range cache {
		if victim == "" || plan.LastUsed.Before(oldestTime) {
			victim = hash
			oldestTime = plan.LastUsed
		}
	}

	return victim
}
