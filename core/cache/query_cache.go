package cache

import (
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// SearchResult represents a search result for caching (avoiding import cycle)
type SearchResult struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"`
	Vector   interface{}       `json:"vector,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// QueryResultCache implements L1 caching for query results
type QueryResultCache struct {
	*BaseCache
	semanticCache  SemanticCache
	ttlCalculator  TTLCalculator
	personalization bool
}

// TTLCalculator determines TTL based on collection characteristics
type TTLCalculator interface {
	CalculateTTL(collection string, updateFrequency time.Duration) time.Duration
}

// DefaultTTLCalculator provides default TTL calculation
type DefaultTTLCalculator struct {
	MinTTL time.Duration
	MaxTTL time.Duration
}

// CalculateTTL returns appropriate TTL for a collection
func (d *DefaultTTLCalculator) CalculateTTL(collection string, updateFrequency time.Duration) time.Duration {
	if updateFrequency == 0 {
		return d.MaxTTL
	}
	
	// TTL should be shorter than update frequency
	ttl := updateFrequency / 2
	
	if ttl < d.MinTTL {
		return d.MinTTL
	}
	if ttl > d.MaxTTL {
		return d.MaxTTL
	}
	
	return ttl
}

// NewQueryResultCache creates a new query result cache
func NewQueryResultCache(capacity, maxMemory int64, config CacheConfig) *QueryResultCache {
	baseCache := NewBaseCache(capacity, maxMemory, config.CleanupInterval)
	
	return &QueryResultCache{
		BaseCache: baseCache,
		ttlCalculator: &DefaultTTLCalculator{
			MinTTL: 30 * time.Second,
			MaxTTL: 5 * time.Minute,
		},
		personalization: true,
	}
}

// GetQueryResult retrieves cached query results
func (qc *QueryResultCache) GetQueryResult(ctx context.Context, key QueryCacheKey) ([]SearchResult, bool) {
	hash := key.Hash()
	
	value, found := qc.Get(ctx, hash)
	if !found {
		return nil, false
	}
	
	// Type assertion with safety check
	cachedResult, ok := value.(*CachedQueryResult)
	if !ok {
		return nil, false
	}
	
	// Check if personalization context matches
	if qc.personalization && cachedResult.UserContext != key.UserID {
		return nil, false
	}
	
	return cachedResult.Results, true
}

// SetQueryResult stores query results in cache
func (qc *QueryResultCache) SetQueryResult(ctx context.Context, key QueryCacheKey, results []SearchResult, ttl time.Duration) error {
	hash := key.Hash()
	
	cachedResult := &CachedQueryResult{
		QueryKey:    key,
		Results:     results,
		Timestamp:   time.Now(),
		UserContext: key.UserID,
		Confidence:  1.0, // Full confidence for exact matches
	}
	
	options := []CacheOption{
		WithTTL(ttl),
		WithUserContext(key.UserID),
	}
	
	return qc.Set(ctx, hash, cachedResult, options...)
}

// GetSemanticallySimilar finds cached results for similar queries
func (qc *QueryResultCache) GetSemanticallySimilar(ctx context.Context, queryVector []float32, threshold float64, userID string) ([]CachedQueryResult, error) {
	if qc.semanticCache == nil {
		return nil, fmt.Errorf("semantic cache not configured")
	}
	
	// Convert to Vector interface
	vec := &SimpleVector{values: queryVector}
	
	// Find similar cached queries
	cachedResults, err := qc.semanticCache.GetSimilar(ctx, vec, threshold)
	if err != nil {
		return nil, err
	}
	
	// Filter by user context if personalization is enabled
	if qc.personalization && userID != "" {
		filtered := make([]CachedResult, 0, len(cachedResults))
		for _, result := range cachedResults {
			if result.UserContext == userID {
				filtered = append(filtered, result)
			}
		}
		cachedResults = filtered
	}
	
	// Convert to QueryResult type
	queryResults := make([]CachedQueryResult, 0, len(cachedResults))
	for _, result := range cachedResults {
		if qr, ok := result.Results[0].(*CachedQueryResult); ok {
			queryResults = append(queryResults, *qr)
		}
	}
	
	return queryResults, nil
}

// SetSemanticCache configures the semantic cache
func (qc *QueryResultCache) SetSemanticCache(semanticCache SemanticCache) {
	qc.semanticCache = semanticCache
}

// EnablePersonalization enables/disables personalization
func (qc *QueryResultCache) EnablePersonalization(enabled bool) {
	qc.personalization = enabled
}

// InvalidateUser removes all cache entries for a specific user
func (qc *QueryResultCache) InvalidateUser(ctx context.Context, userID string) error {
	qc.mu.Lock()
	defer qc.mu.Unlock()
	
	var toRemove []*list.Element
	
	for elem := qc.lruList.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(*CacheEntry)
		if entry.UserContext == userID {
			toRemove = append(toRemove, elem)
		}
	}
	
	for _, elem := range toRemove {
		qc.removeLocked(elem)
	}
	
	return nil
}

// InvalidateCollection removes all cache entries for a collection
func (qc *QueryResultCache) InvalidateCollection(ctx context.Context, collection string) error {
	qc.mu.Lock()
	defer qc.mu.Unlock()
	
	var toRemove []*list.Element
	
	for elem := qc.lruList.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(*CacheEntry)
		if cachedResult, ok := entry.Value.(*CachedQueryResult); ok {
			if cachedResult.QueryKey.Collection == collection {
				toRemove = append(toRemove, elem)
			}
		}
	}
	
	for _, elem := range toRemove {
		qc.removeLocked(elem)
	}
	
	return nil
}

// CachedQueryResult represents a cached query result
type CachedQueryResult struct {
	QueryKey     QueryCacheKey
	Results      []SearchResult
	Timestamp    time.Time
	AccessCount  int64
	UserContext  string
	Confidence   float64
	SemanticHash string
}

// SimpleVector implements the Vector interface
type SimpleVector struct {
	values []float32
}

func (v *SimpleVector) Dimension() int {
	return len(v.values)
}

func (v *SimpleVector) Values() []float32 {
	return v.values
}

func (v *SimpleVector) Distance(other Vector) float32 {
	// Simple L2 distance for now
	otherValues := other.Values()
	if len(v.values) != len(otherValues) {
		return -1
	}
	
	var sum float32
	for i := range v.values {
		diff := v.values[i] - otherValues[i]
		sum += diff * diff
	}
	
	return sum // Return squared distance for efficiency
}

// MarshalJSON implements json.Marshaler
func (r *CachedQueryResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		QueryKey    QueryCacheKey       `json:"query_key"`
		Results     []SearchResult `json:"results"`
		Timestamp   time.Time           `json:"timestamp"`
		AccessCount int64               `json:"access_count"`
		UserContext string              `json:"user_context"`
		Confidence  float64             `json:"confidence"`
	}{
		QueryKey:    r.QueryKey,
		Results:     r.Results,
		Timestamp:   r.Timestamp,
		AccessCount: r.AccessCount,
		UserContext: r.UserContext,
		Confidence:  r.Confidence,
	})
}