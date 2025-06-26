package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"time"
)

// QueryCacheKey represents a cache key for query results
type QueryCacheKey struct {
	Collection     string
	QueryVector    []float32
	TopK           int
	Filters        map[string]string
	DistanceMetric string
	UserID         string
	IncludeVectors bool
}

// Hash generates a unique hash for the query cache key
func (k QueryCacheKey) Hash() string {
	h := sha256.New()
	
	// Write collection name
	h.Write([]byte(k.Collection))
	h.Write([]byte("|"))
	
	// Write query vector (first 8 values for efficiency)
	vectorLen := len(k.QueryVector)
	if vectorLen > 8 {
		vectorLen = 8
	}
	for i := 0; i < vectorLen; i++ {
		h.Write([]byte(fmt.Sprintf("%.4f", k.QueryVector[i])))
	}
	h.Write([]byte(fmt.Sprintf("|len:%d|", len(k.QueryVector))))
	
	// Write TopK
	h.Write([]byte(fmt.Sprintf("k:%d|", k.TopK)))
	
	// Write sorted filters
	if len(k.Filters) > 0 {
		var filterKeys []string
		for key := range k.Filters {
			filterKeys = append(filterKeys, key)
		}
		sort.Strings(filterKeys)
		
		for _, key := range filterKeys {
			h.Write([]byte(fmt.Sprintf("%s:%s|", key, k.Filters[key])))
		}
	}
	
	// Write distance metric
	h.Write([]byte(k.DistanceMetric))
	h.Write([]byte("|"))
	
	// Write user ID for personalization
	if k.UserID != "" {
		h.Write([]byte("user:" + k.UserID))
		h.Write([]byte("|"))
	}
	
	// Write include vectors flag
	h.Write([]byte(fmt.Sprintf("vec:%t", k.IncludeVectors)))
	
	return hex.EncodeToString(h.Sum(nil))
}

// VectorCacheKey represents a cache key for vectors
type VectorCacheKey struct {
	Collection string
	VectorID   string
}

// Hash generates a unique hash for the vector cache key
func (k VectorCacheKey) Hash() string {
	return fmt.Sprintf("%s:%s", k.Collection, k.VectorID)
}

// IndexCacheKey represents a cache key for index partitions
type IndexCacheKey struct {
	Collection    string
	IndexType     string
	PartitionID   int
	PartitionHash string
}

// Hash generates a unique hash for the index cache key
func (k IndexCacheKey) Hash() string {
	return fmt.Sprintf("%s:%s:p%d:%s", k.Collection, k.IndexType, k.PartitionID, k.PartitionHash)
}

// CacheEntry represents a generic cache entry
type CacheEntry struct {
	Key         string
	Value       interface{}
	Size        int64
	CreatedAt   time.Time
	LastAccess  time.Time
	AccessCount int64
	TTL         time.Duration
	Priority    int
	UserContext string
	Metadata    map[string]interface{}
}

// IsExpired checks if the cache entry has expired
func (e *CacheEntry) IsExpired() bool {
	if e.TTL == 0 {
		return false
	}
	return time.Since(e.CreatedAt) > e.TTL
}

// Touch updates the last access time and increments access count
func (e *CacheEntry) Touch() {
	e.LastAccess = time.Now()
	e.AccessCount++
}

// Age returns the age of the cache entry
func (e *CacheEntry) Age() time.Duration {
	return time.Since(e.CreatedAt)
}

// IdleTime returns the time since last access
func (e *CacheEntry) IdleTime() time.Duration {
	return time.Since(e.LastAccess)
}

// CacheConfig represents cache configuration
type CacheConfig struct {
	MaxSize         int64         `json:"max_size"`
	MaxMemory       int64         `json:"max_memory"`
	DefaultTTL      time.Duration `json:"default_ttl"`
	EvictionPolicy  string        `json:"eviction_policy"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// MemoryStats tracks memory usage
type MemoryStats struct {
	Used      int64   `json:"used"`
	Available int64   `json:"available"`
	Limit     int64   `json:"limit"`
	Pressure  float64 `json:"pressure"` // 0.0 to 1.0
}

// CacheEventType represents types of cache events
type CacheEventType string

const (
	CacheEventHit       CacheEventType = "hit"
	CacheEventMiss      CacheEventType = "miss"
	CacheEventSet       CacheEventType = "set"
	CacheEventEvict     CacheEventType = "evict"
	CacheEventExpire    CacheEventType = "expire"
	CacheEventClear     CacheEventType = "clear"
	CacheEventPromote   CacheEventType = "promote"
	CacheEventDemote    CacheEventType = "demote"
)

// CacheEvent represents a cache event for monitoring
type CacheEvent struct {
	Type      CacheEventType
	Level     CacheLevel
	Key       string
	Size      int64
	Timestamp time.Time
	UserID    string
	Metadata  map[string]interface{}
}

// SemanticCluster represents a cluster of semantically similar queries
type SemanticCluster struct {
	ID          string
	Centroid    []float32
	Members     []string // Cache keys
	Radius      float32
	CreatedAt   time.Time
	LastUpdated time.Time
	HitCount    int64
	MissCount   int64
}

// Contains checks if a vector is within the cluster radius
func (c *SemanticCluster) Contains(vector []float32, distanceFunc func(a, b []float32) float32) bool {
	distance := distanceFunc(c.Centroid, vector)
	return distance <= c.Radius
}

// HitRate returns the cluster's cache hit rate
func (c *SemanticCluster) HitRate() float64 {
	total := c.HitCount + c.MissCount
	if total == 0 {
		return 0
	}
	return float64(c.HitCount) / float64(total)
}

// VectorMetadata stores metadata about cached vectors
type VectorMetadata struct {
	VectorID        string
	Collection      string
	Dimension       int
	ComputeCost     float64 // Milliseconds to compute/fetch
	AccessFrequency float64 // Exponentially weighted access rate
	UserAssociation []string
	LastModified    time.Time
}

// IndexPartitionMetadata stores metadata about cached index partitions
type IndexPartitionMetadata struct {
	PartitionID   int
	Collection    string
	IndexType     string
	VectorCount   int
	SizeBytes     int64
	AccessPattern []time.Time // Recent access times
	Temperature   float64     // Hot/cold score
}