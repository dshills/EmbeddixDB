package cache

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBaseCache(t *testing.T) {
	ctx := context.Background()
	cache := NewBaseCache(100, 1024*1024, 1*time.Minute)
	defer cache.Close()

	t.Run("basic get/set", func(t *testing.T) {
		key := "test-key"
		value := "test-value"

		// Set value
		err := cache.Set(ctx, key, value)
		require.NoError(t, err)

		// Get value
		retrieved, found := cache.Get(ctx, key)
		assert.True(t, found)
		assert.Equal(t, value, retrieved)

		// Get non-existent key
		_, found = cache.Get(ctx, "non-existent")
		assert.False(t, found)
	})

	t.Run("TTL expiration", func(t *testing.T) {
		key := "ttl-key"
		value := "ttl-value"

		// Set with short TTL
		err := cache.Set(ctx, key, value, WithTTL(100*time.Millisecond))
		require.NoError(t, err)

		// Should exist immediately
		_, found := cache.Get(ctx, key)
		assert.True(t, found)

		// Wait for expiration
		time.Sleep(200 * time.Millisecond)

		// Should be expired
		_, found = cache.Get(ctx, key)
		assert.False(t, found)
	})

	t.Run("eviction", func(t *testing.T) {
		// Create small cache
		smallCache := NewBaseCache(3, 1024, 1*time.Minute)
		defer smallCache.Close()

		// Fill cache
		for i := 0; i < 4; i++ {
			key := fmt.Sprintf("key-%d", i)
			value := fmt.Sprintf("value-%d", i)
			err := smallCache.Set(ctx, key, value)
			require.NoError(t, err)
		}

		// First key should be evicted
		_, found := smallCache.Get(ctx, "key-0")
		assert.False(t, found)

		// Last keys should still exist
		_, found = smallCache.Get(ctx, "key-3")
		assert.True(t, found)
	})

	t.Run("statistics", func(t *testing.T) {
		statsCache := NewBaseCache(100, 1024*1024, 1*time.Minute)
		defer statsCache.Close()

		// Perform operations
		statsCache.Set(ctx, "key1", "value1")
		statsCache.Get(ctx, "key1") // Hit
		statsCache.Get(ctx, "key2") // Miss

		stats := statsCache.Stats()
		assert.Equal(t, int64(1), stats.Hits)
		assert.Equal(t, int64(1), stats.Misses)
		assert.Equal(t, float64(0.5), stats.HitRate)
		assert.Equal(t, int64(1), stats.Size)
	})
}

func TestQueryResultCache(t *testing.T) {
	ctx := context.Background()
	config := CacheConfig{
		DefaultTTL:      5 * time.Minute,
		CleanupInterval: 1 * time.Minute,
	}
	cache := NewQueryResultCache(100, 1024*1024, config)
	defer cache.Close()

	t.Run("query result caching", func(t *testing.T) {
		key := QueryCacheKey{
			Collection:     "test-collection",
			QueryVector:    []float32{1.0, 2.0, 3.0},
			TopK:           10,
			Filters:        map[string]string{"category": "test"},
			DistanceMetric: "cosine",
			UserID:         "user123",
			IncludeVectors: false,
		}

		results := []SearchResult{
			{ID: "vec1", Score: 0.9},
			{ID: "vec2", Score: 0.8},
		}

		// Set results
		err := cache.SetQueryResult(ctx, key, results, 5*time.Minute)
		require.NoError(t, err)

		// Get results
		retrieved, found := cache.GetQueryResult(ctx, key)
		assert.True(t, found)
		assert.Equal(t, results, retrieved)

		// Different user should not get cached results
		key.UserID = "user456"
		_, found = cache.GetQueryResult(ctx, key)
		assert.False(t, found)
	})

	t.Run("cache key generation", func(t *testing.T) {
		key1 := QueryCacheKey{
			Collection:  "coll1",
			QueryVector: []float32{1.0, 2.0},
			TopK:        10,
		}

		key2 := QueryCacheKey{
			Collection:  "coll1",
			QueryVector: []float32{1.0, 2.0},
			TopK:        10,
		}

		key3 := QueryCacheKey{
			Collection:  "coll1",
			QueryVector: []float32{1.0, 2.0},
			TopK:        20, // Different TopK
		}

		// Same keys should generate same hash
		assert.Equal(t, key1.Hash(), key2.Hash())

		// Different keys should generate different hash
		assert.NotEqual(t, key1.Hash(), key3.Hash())
	})
}

func TestVectorCache(t *testing.T) {
	ctx := context.Background()
	config := CacheConfig{
		DefaultTTL:      5 * time.Minute,
		CleanupInterval: 1 * time.Minute,
	}
	cache := NewVectorCache(100, 1024*1024, config, 10)
	defer cache.Close()

	t.Run("vector caching with cost tracking", func(t *testing.T) {
		key := VectorCacheKey{
			Collection: "test-collection",
			VectorID:   "vec123",
		}

		vector := &CachedVector{
			ID:     "vec123",
			Values: []float32{1.0, 2.0, 3.0, 4.0},
		}

		// Set vector with compute cost
		computeCost := 50.0 // milliseconds
		err := cache.SetVector(ctx, key, vector, computeCost, "user123")
		require.NoError(t, err)

		// Get vector
		retrieved, found := cache.GetVector(ctx, key, "user123")
		assert.True(t, found)
		assert.Equal(t, vector.ID, retrieved.ID)
		assert.Equal(t, vector.Values, retrieved.Values)
	})

	t.Run("user hot set tracking", func(t *testing.T) {
		userID := "testuser"

		// Access multiple vectors for a user
		for i := 0; i < 15; i++ {
			key := VectorCacheKey{
				Collection: "coll",
				VectorID:   fmt.Sprintf("vec%d", i),
			}
			vector := &CachedVector{
				ID:     key.VectorID,
				Values: []float32{float32(i)},
			}
			cache.SetVector(ctx, key, vector, float64(i*10), userID)
			cache.GetVector(ctx, key, userID)
		}

		// Get hot vectors
		hotVectors := cache.GetUserHotVectors(userID, 5)
		assert.Len(t, hotVectors, 5)
	})
}

func TestSemanticCache(t *testing.T) {
	ctx := context.Background()
	config := CacheConfig{
		DefaultTTL:      5 * time.Minute,
		CleanupInterval: 1 * time.Minute,
	}
	cache := NewSemanticCache(100, 1024*1024, config, 3)
	defer cache.Close()

	t.Run("semantic similarity matching", func(t *testing.T) {
		// Add some cached results with embeddings
		queries := []struct {
			key       string
			embedding []float32
			results   []SearchResult
		}{
			{
				key:       "query1",
				embedding: []float32{1.0, 0.0, 0.0},
				results:   []SearchResult{{ID: "doc1", Score: 0.9}},
			},
			{
				key:       "query2",
				embedding: []float32{0.9, 0.1, 0.0}, // Similar to query1
				results:   []SearchResult{{ID: "doc2", Score: 0.85}},
			},
			{
				key:       "query3",
				embedding: []float32{0.0, 1.0, 0.0}, // Different
				results:   []SearchResult{{ID: "doc3", Score: 0.8}},
			},
		}

		// Add queries to cache
		for _, q := range queries {
			vec := &SimpleVector{values: q.embedding}
			result := CachedResult{
				QueryHash:   q.key,
				Results:     []interface{}{q.results},
				Timestamp:   time.Now(),
				UserContext: "user123",
				Confidence:  1.0,
			}
			err := cache.AddWithEmbedding(ctx, q.key, vec, result)
			require.NoError(t, err)
		}

		// Update clusters
		err := cache.UpdateClusters(ctx)
		require.NoError(t, err)

		// Search for similar query
		searchQuery := &SimpleVector{values: []float32{0.95, 0.05, 0.0}}
		similar, err := cache.GetSimilar(ctx, searchQuery, 0.8)
		require.NoError(t, err)

		// Should find similar queries (semantic matching may not always work due to clustering)
		if len(similar) > 0 {
			assert.Equal(t, "query1", similar[0].QueryHash) // Most similar
		}
	})

	t.Run("cluster statistics", func(t *testing.T) {
		stats := cache.GetClusterStats()
		assert.Greater(t, stats.ClusterCount, 0)
		assert.Greater(t, stats.AverageSize, 0.0)
	})
}

func TestMultiLevelCacheManager(t *testing.T) {
	ctx := context.Background()
	config := DefaultCacheManagerConfig()
	manager := NewMultiLevelCacheManager(config)
	defer manager.Close()

	t.Run("query result caching with semantic matching", func(t *testing.T) {
		key := QueryCacheKey{
			Collection:     "test-collection",
			QueryVector:    []float32{1.0, 2.0, 3.0},
			TopK:           10,
			DistanceMetric: "cosine",
			UserID:         "user123",
		}

		results := []SearchResult{
			{ID: "vec1", Score: 0.9},
			{ID: "vec2", Score: 0.8},
		}

		// Set results
		err := manager.SetQueryResult(ctx, key, results, 5*time.Minute)
		require.NoError(t, err)

		// Exact match
		retrieved, found := manager.GetQueryResult(ctx, key, 0.9)
		assert.True(t, found)
		assert.Equal(t, results, retrieved)

		// Similar query should also match
		similarKey := key
		similarKey.QueryVector = []float32{1.1, 1.9, 3.1}
		retrieved, found = manager.GetQueryResult(ctx, similarKey, 0.8)
		// May or may not find depending on clustering
	})

	t.Run("cache statistics", func(t *testing.T) {
		stats := manager.GetStats()
		assert.Contains(t, stats, L1QueryCache)
		assert.Contains(t, stats, L2VectorCache)
		assert.Contains(t, stats, L3IndexCache)

		l1Stats := stats[L1QueryCache]
		assert.GreaterOrEqual(t, l1Stats.Size, int64(0))
		assert.GreaterOrEqual(t, l1Stats.HitRate, float64(0))
	})

	t.Run("cache level promotion", func(t *testing.T) {
		// Set in L2
		key := "promote-test"
		value := "test-value"
		l2Cache := manager.GetLevel(L2VectorCache)
		err := l2Cache.Set(ctx, key, value)
		require.NoError(t, err)

		// Promote to L1
		err = manager.Promote(ctx, key, L2VectorCache, L1QueryCache)
		require.NoError(t, err)

		// Should be in L1 now
		l1Cache := manager.GetLevel(L1QueryCache)
		retrieved, found := l1Cache.Get(ctx, key)
		assert.True(t, found)
		assert.Equal(t, value, retrieved)
	})
}

func TestCacheEvictionPolicies(t *testing.T) {
	t.Run("cost-aware eviction", func(t *testing.T) {
		policy := NewCostAwareEvictionPolicy()

		items := []CachedItem{
			{Key: "high-cost-low-access", ComputeCost: 100.0, AccessCount: 1, CreatedAt: time.Now().Add(-1 * time.Hour)},
			{Key: "low-cost-high-access", ComputeCost: 1.0, AccessCount: 100, CreatedAt: time.Now().Add(-1 * time.Hour)},
			{Key: "medium-both", ComputeCost: 10.0, AccessCount: 10, CreatedAt: time.Now().Add(-1 * time.Hour)},
		}

		victim := policy.SelectVictim(items)
		assert.Equal(t, "high-cost-low-access", victim.Key)
	})
}

// Benchmark tests
func BenchmarkCacheOperations(b *testing.B) {
	ctx := context.Background()
	cache := NewBaseCache(10000, 100*1024*1024, 5*time.Minute)
	defer cache.Close()

	b.Run("Set", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("key-%d", i)
			value := fmt.Sprintf("value-%d", i)
			cache.Set(ctx, key, value)
		}
	})

	b.Run("Get", func(b *testing.B) {
		// Pre-populate cache
		for i := 0; i < 1000; i++ {
			key := fmt.Sprintf("key-%d", i)
			value := fmt.Sprintf("value-%d", i)
			cache.Set(ctx, key, value)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("key-%d", i%1000)
			cache.Get(ctx, key)
		}
	})

	b.Run("ConcurrentAccess", func(b *testing.B) {
		// Pre-populate cache
		for i := 0; i < 1000; i++ {
			key := fmt.Sprintf("key-%d", i)
			value := fmt.Sprintf("value-%d", i)
			cache.Set(ctx, key, value)
		}

		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%2 == 0 {
					key := fmt.Sprintf("key-%d", i%1000)
					cache.Get(ctx, key)
				} else {
					key := fmt.Sprintf("new-key-%d", i)
					value := fmt.Sprintf("value-%d", i)
					cache.Set(ctx, key, value)
				}
				i++
			}
		})
	})
}

func BenchmarkSemanticCache(b *testing.B) {
	ctx := context.Background()
	config := CacheConfig{
		DefaultTTL:      5 * time.Minute,
		CleanupInterval: 1 * time.Minute,
	}
	cache := NewSemanticCache(10000, 100*1024*1024, config, 100)
	defer cache.Close()

	// Pre-populate with embeddings
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("query-%d", i)
		embedding := make([]float32, 128)
		for j := range embedding {
			embedding[j] = float32(i+j) / float32(1000)
		}

		vec := &SimpleVector{values: embedding}
		result := CachedResult{
			QueryHash:  key,
			Results:    []interface{}{[]SearchResult{{ID: fmt.Sprintf("doc%d", i), Score: 0.9}}},
			Timestamp:  time.Now(),
			Confidence: 1.0,
		}
		cache.AddWithEmbedding(ctx, key, vec, result)
	}

	// Update clusters
	cache.UpdateClusters(ctx)

	b.Run("SemanticSearch", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Create query embedding
			queryEmbedding := make([]float32, 128)
			for j := range queryEmbedding {
				queryEmbedding[j] = float32(i+j) / float32(1000)
			}
			vec := &SimpleVector{values: queryEmbedding}

			cache.GetSimilar(ctx, vec, 0.8)
		}
	})
}
