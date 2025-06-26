package cache

import (
	"context"
	"time"
)

// CacheLevel represents different levels in the cache hierarchy
type CacheLevel int

const (
	// L1QueryCache caches complete query results
	L1QueryCache CacheLevel = iota
	// L2VectorCache caches frequently accessed vectors
	L2VectorCache
	// L3IndexCache caches hot index partitions
	L3IndexCache
)

// CacheStats provides cache performance statistics
type CacheStats struct {
	Hits        int64     `json:"hits"`
	Misses      int64     `json:"misses"`
	Evictions   int64     `json:"evictions"`
	Size        int64     `json:"size"`
	MemoryUsage int64     `json:"memory_usage"`
	HitRate     float64   `json:"hit_rate"`
	LastReset   time.Time `json:"last_reset"`
}

// CacheOption configures cache operations
type CacheOption func(*cacheOptions)

type cacheOptions struct {
	TTL           time.Duration
	Priority      int
	UserContext   string
	SemanticGroup string
}

// WithTTL sets the time-to-live for cached items
func WithTTL(ttl time.Duration) CacheOption {
	return func(opts *cacheOptions) {
		opts.TTL = ttl
	}
}

// WithPriority sets the priority for cache retention
func WithPriority(priority int) CacheOption {
	return func(opts *cacheOptions) {
		opts.Priority = priority
	}
}

// WithUserContext associates cache entry with a user
func WithUserContext(userID string) CacheOption {
	return func(opts *cacheOptions) {
		opts.UserContext = userID
	}
}

// WithSemanticGroup associates cache entry with a semantic group
func WithSemanticGroup(group string) CacheOption {
	return func(opts *cacheOptions) {
		opts.SemanticGroup = group
	}
}

// Cache defines the basic cache interface
type Cache interface {
	// Get retrieves a value from the cache
	Get(ctx context.Context, key string) (interface{}, bool)
	
	// Set stores a value in the cache with optional configuration
	Set(ctx context.Context, key string, value interface{}, options ...CacheOption) error
	
	// Delete removes a value from the cache
	Delete(ctx context.Context, key string) error
	
	// Clear removes all values from the cache
	Clear(ctx context.Context) error
	
	// Stats returns cache performance statistics
	Stats() CacheStats
	
	// Size returns the current size of the cache
	Size() int64
}

// EvictionPolicy defines how items are removed from cache
type EvictionPolicy interface {
	// ShouldEvict determines if an item should be evicted
	ShouldEvict(item CachedItem) bool
	
	// SelectVictim chooses which item to evict
	SelectVictim(items []CachedItem) CachedItem
	
	// OnAccess updates access statistics for an item
	OnAccess(key string)
	
	// OnEvict handles cleanup when an item is evicted
	OnEvict(item CachedItem)
}

// CachedItem represents an item stored in cache
type CachedItem struct {
	Key          string
	Value        interface{}
	Size         int64
	AccessCount  int64
	LastAccess   time.Time
	CreatedAt    time.Time
	TTL          time.Duration
	Priority     int
	UserContext  string
	ComputeCost  float64 // Cost to recompute this item
}

// MultiLevelCache manages multiple cache levels
type MultiLevelCache interface {
	Cache
	
	// GetLevel returns a specific cache level
	GetLevel(level CacheLevel) Cache
	
	// SetEvictionPolicy sets the eviction policy for a cache level
	SetEvictionPolicy(level CacheLevel, policy EvictionPolicy) error
	
	// Promote moves an item up in the cache hierarchy
	Promote(ctx context.Context, key string, fromLevel, toLevel CacheLevel) error
	
	// Demote moves an item down in the cache hierarchy
	Demote(ctx context.Context, key string, fromLevel, toLevel CacheLevel) error
}

// Vector represents a vector for semantic caching
type Vector interface {
	// Dimension returns the vector dimension
	Dimension() int
	
	// Values returns the vector values
	Values() []float32
	
	// Distance calculates distance to another vector
	Distance(other Vector) float32
}

// CachedResult represents a cached query result
type CachedResult struct {
	QueryHash    string
	Results      []interface{}
	Timestamp    time.Time
	AccessCount  int64
	UserContext  string
	Confidence   float64
	SemanticHash string
}

// SemanticCache provides semantic similarity-based caching
type SemanticCache interface {
	Cache
	
	// GetSimilar finds cached results similar to the query
	GetSimilar(ctx context.Context, query Vector, threshold float64) ([]CachedResult, error)
	
	// AddWithEmbedding adds a result with its query embedding
	AddWithEmbedding(ctx context.Context, key string, embedding Vector, result CachedResult) error
	
	// UpdateClusters updates the semantic clustering
	UpdateClusters(ctx context.Context) error
	
	// GetClusterStats returns clustering statistics
	GetClusterStats() ClusterStats
}

// ClusterStats provides semantic clustering statistics
type ClusterStats struct {
	ClusterCount    int     `json:"cluster_count"`
	AverageSize     float64 `json:"average_size"`
	MinSize         int     `json:"min_size"`
	MaxSize         int     `json:"max_size"`
	Compactness     float64 `json:"compactness"`
	SeparationScore float64 `json:"separation_score"`
}

// CacheManager coordinates the multi-level cache system
type CacheManager interface {
	// Get attempts to retrieve from all cache levels
	Get(ctx context.Context, key string) (interface{}, CacheLevel, bool)
	
	// Set stores in the appropriate cache level
	Set(ctx context.Context, key string, value interface{}, options ...CacheOption) error
	
	// WarmUp pre-loads frequently accessed items
	WarmUp(ctx context.Context) error
	
	// Optimize redistributes items between cache levels
	Optimize(ctx context.Context) error
	
	// GetStats returns statistics for all cache levels
	GetStats() map[CacheLevel]CacheStats
	
	// SetMemoryBudget sets memory limits for each level
	SetMemoryBudget(budgets map[CacheLevel]int64) error
}