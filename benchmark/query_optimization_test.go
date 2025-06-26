package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/performance"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

// BenchmarkQueryOptimizations tests various query optimization strategies
func BenchmarkQueryOptimizations(b *testing.B) {
	vectorCounts := []int{10000, 50000, 100000}
	topKValues := []int{5, 10, 50, 100}

	for _, count := range vectorCounts {
		for _, k := range topKValues {
			b.Run(fmt.Sprintf("Vectors_%d_TopK_%d", count, k), func(b *testing.B) {
				benchmarkOptimizedSearch(b, count, k)
			})
		}
	}
}

// benchmarkOptimizedSearch compares standard vs optimized search
func benchmarkOptimizedSearch(b *testing.B, vectorCount, topK int) {
	// Create test data
	dimension := 768
	vectors := generateTestVectors(vectorCount, dimension)

	// Create standard vector store
	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	standardStore := core.NewVectorStore(memStore, indexFactory)

	// Create optimized vector store
	config := core.DefaultOptimizationConfig()
	config.EnableParallelExecution = true
	config.EnableProgressiveSearch = true
	config.EnableQueryPlanCaching = true
	optimizedStore := core.NewOptimizedVectorStore(standardStore, config)

	// Set up profiler
	profilingConfig := performance.DefaultProfilingConfig()
	profilingConfig.Enabled = true
	profiler := performance.NewProfiler(profilingConfig)
	optimizedStore.SetProfiler(profiler)

	// Create collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "bench_collection",
		Dimension: dimension,
		Distance:  "cosine",
		IndexType: "hnsw",
	}

	if err := standardStore.CreateCollection(ctx, collection); err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	// Add vectors in batches
	b.Logf("Adding %d vectors...", vectorCount)
	batchSize := 1000
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		if err := standardStore.AddVectorsBatch(ctx, collection.Name, vectors[i:end]); err != nil {
			b.Fatalf("Failed to add vectors: %v", err)
		}
	}

	// Generate query vectors
	queryVectors := generateTestVectors(100, dimension)

	b.Run("Standard", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := queryVectors[i%len(queryVectors)]
			req := core.SearchRequest{
				Query: query.Values,
				TopK:  topK,
			}

			_, err := standardStore.Search(ctx, collection.Name, req)
			if err != nil {
				b.Fatalf("Search failed: %v", err)
			}
		}
	})

	b.Run("Optimized", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := queryVectors[i%len(queryVectors)]
			req := core.SearchRequest{
				Query: query.Values,
				TopK:  topK,
			}

			_, err := optimizedStore.OptimizedSearch(ctx, collection.Name, req)
			if err != nil {
				b.Fatalf("Optimized search failed: %v", err)
			}
		}
	})

	// Report optimization metrics
	metrics := optimizedStore.GetQueryMetrics()
	b.Logf("Parallel queries: %d/%d (%.2f%%)",
		metrics.ParallelQueries,
		metrics.QueriesExecuted,
		float64(metrics.ParallelQueries)/float64(metrics.QueriesExecuted)*100)
}

// BenchmarkParallelExecution tests parallel query execution
func BenchmarkParallelExecution(b *testing.B) {
	parallelDegrees := []int{1, 2, 4, 8}

	for _, degree := range parallelDegrees {
		b.Run(fmt.Sprintf("Parallel_%d", degree), func(b *testing.B) {
			benchmarkParallelDegree(b, degree)
		})
	}
}

// benchmarkParallelDegree tests specific parallel degree
func benchmarkParallelDegree(b *testing.B, parallelDegree int) {
	// Setup
	vectorCount := 100000
	dimension := 768
	topK := 50

	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	standardStore := core.NewVectorStore(memStore, indexFactory)

	config := core.DefaultOptimizationConfig()
	config.ParallelWorkers = parallelDegree
	config.EnableParallelExecution = true
	optimizedStore := core.NewOptimizedVectorStore(standardStore, config)

	// Create and populate collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      fmt.Sprintf("parallel_bench_%d", parallelDegree),
		Dimension: dimension,
		Distance:  "cosine",
		IndexType: "hnsw",
	}

	if err := standardStore.CreateCollection(ctx, collection); err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	vectors := generateTestVectors(vectorCount, dimension)
	if err := standardStore.AddVectorsBatch(ctx, collection.Name, vectors); err != nil {
		b.Fatalf("Failed to add vectors: %v", err)
	}

	// Run benchmark
	queryVector := generateTestVectors(1, dimension)[0]
	req := core.SearchRequest{
		Query: queryVector.Values,
		TopK:  topK,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := optimizedStore.OptimizedSearch(ctx, collection.Name, req)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

// BenchmarkProgressiveSearch tests progressive search performance
func BenchmarkProgressiveSearch(b *testing.B) {
	vectorCount := 100000
	dimension := 768

	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	standardStore := core.NewVectorStore(memStore, indexFactory)

	config := core.DefaultOptimizationConfig()
	config.EnableProgressiveSearch = true
	optimizedStore := core.NewOptimizedVectorStore(standardStore, config)

	// Setup collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "progressive_bench",
		Dimension: dimension,
		Distance:  "cosine",
		IndexType: "hnsw",
	}

	if err := standardStore.CreateCollection(ctx, collection); err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	vectors := generateTestVectors(vectorCount, dimension)
	if err := standardStore.AddVectorsBatch(ctx, collection.Name, vectors); err != nil {
		b.Fatalf("Failed to add vectors: %v", err)
	}

	// Test different result sizes
	topKValues := []int{10, 50, 100, 500}

	for _, k := range topKValues {
		b.Run(fmt.Sprintf("TopK_%d", k), func(b *testing.B) {
			queryVector := generateTestVectors(1, dimension)[0]
			req := core.SearchRequest{
				Query: queryVector.Values,
				TopK:  k,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := optimizedStore.OptimizedSearch(ctx, collection.Name, req)
				if err != nil {
					b.Fatalf("Progressive search failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkConcurrentQueries tests concurrent query performance
func BenchmarkConcurrentQueries(b *testing.B) {
	concurrencyLevels := []int{1, 10, 50, 100}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrent_%d", concurrency), func(b *testing.B) {
			benchmarkConcurrentQueries(b, concurrency)
		})
	}
}

// benchmarkConcurrentQueries tests specific concurrency level
func benchmarkConcurrentQueries(b *testing.B, concurrency int) {
	// Setup
	vectorCount := 50000
	dimension := 768
	topK := 10

	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	standardStore := core.NewVectorStore(memStore, indexFactory)

	config := core.DefaultOptimizationConfig()
	config.MaxConcurrentQueries = concurrency
	optimizedStore := core.NewOptimizedVectorStore(standardStore, config)

	// Create collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "concurrent_bench",
		Dimension: dimension,
		Distance:  "cosine",
		IndexType: "hnsw",
	}

	if err := standardStore.CreateCollection(ctx, collection); err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	vectors := generateTestVectors(vectorCount, dimension)
	if err := standardStore.AddVectorsBatch(ctx, collection.Name, vectors); err != nil {
		b.Fatalf("Failed to add vectors: %v", err)
	}

	// Generate query vectors
	queryVectors := generateTestVectors(concurrency, dimension)

	b.ResetTimer()

	// Run concurrent queries
	var wg sync.WaitGroup
	errors := make(chan error, concurrency)

	for i := 0; i < b.N; i++ {
		for j := 0; j < concurrency; j++ {
			wg.Add(1)
			go func(queryIdx int) {
				defer wg.Done()

				req := core.SearchRequest{
					Query: queryVectors[queryIdx].Values,
					TopK:  topK,
				}

				_, err := optimizedStore.OptimizedSearch(ctx, collection.Name, req)
				if err != nil {
					errors <- err
				}
			}(j % len(queryVectors))
		}

		wg.Wait()

		// Check for errors
		select {
		case err := <-errors:
			b.Fatalf("Concurrent query failed: %v", err)
		default:
		}
	}
}

// BenchmarkQueryPlanCaching tests query plan cache effectiveness
func BenchmarkQueryPlanCaching(b *testing.B) {
	vectorCount := 50000
	dimension := 768
	topK := 20

	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	standardStore := core.NewVectorStore(memStore, indexFactory)

	// Test with and without caching
	configs := []struct {
		name    string
		caching bool
	}{
		{"WithoutCache", false},
		{"WithCache", true},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			config := core.DefaultOptimizationConfig()
			config.EnableQueryPlanCaching = cfg.caching
			optimizedStore := core.NewOptimizedVectorStore(standardStore, config)

			ctx := context.Background()
			collection := core.Collection{
				Name:      fmt.Sprintf("cache_bench_%s", cfg.name),
				Dimension: dimension,
				Distance:  "cosine",
				IndexType: "hnsw",
			}

			if err := standardStore.CreateCollection(ctx, collection); err != nil {
				b.Fatalf("Failed to create collection: %v", err)
			}

			vectors := generateTestVectors(vectorCount, dimension)
			if err := standardStore.AddVectorsBatch(ctx, collection.Name, vectors); err != nil {
				b.Fatalf("Failed to add vectors: %v", err)
			}

			// Use a small set of query patterns to test cache effectiveness
			queryPatterns := generateTestVectors(10, dimension)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Repeatedly use same query patterns
				query := queryPatterns[i%len(queryPatterns)]
				req := core.SearchRequest{
					Query: query.Values,
					TopK:  topK,
				}

				_, err := optimizedStore.OptimizedSearch(ctx, collection.Name, req)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}

// generateTestVectors creates test vectors with random values
func generateTestVectors(count, dimension int) []core.Vector {
	vectors := make([]core.Vector, count)
	rand.Seed(42) // Fixed seed for reproducibility

	for i := 0; i < count; i++ {
		values := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			values[j] = rand.Float32()
		}

		vectors[i] = core.Vector{
			ID:     fmt.Sprintf("vec_%d", i),
			Values: values,
			Metadata: map[string]string{
				"index": fmt.Sprintf("%d", i),
			},
		}
	}

	return vectors
}

// BenchmarkMemoryEfficiency tests memory usage under different loads
func BenchmarkMemoryEfficiency(b *testing.B) {
	memoryLimits := []int64{
		100 * 1024 * 1024,  // 100MB
		500 * 1024 * 1024,  // 500MB
		1024 * 1024 * 1024, // 1GB
	}

	for _, limit := range memoryLimits {
		b.Run(fmt.Sprintf("Memory_%dMB", limit/(1024*1024)), func(b *testing.B) {
			benchmarkMemoryLimit(b, limit)
		})
	}
}

// benchmarkMemoryLimit tests performance under memory constraints
func benchmarkMemoryLimit(b *testing.B, memoryLimit int64) {
	vectorCount := 50000
	dimension := 768

	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	standardStore := core.NewVectorStore(memStore, indexFactory)

	config := core.DefaultOptimizationConfig()
	config.MaxMemoryBytes = memoryLimit
	optimizedStore := core.NewOptimizedVectorStore(standardStore, config)

	ctx := context.Background()
	collection := core.Collection{
		Name:      "memory_bench",
		Dimension: dimension,
		Distance:  "cosine",
		IndexType: "hnsw",
	}

	if err := standardStore.CreateCollection(ctx, collection); err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	vectors := generateTestVectors(vectorCount, dimension)
	if err := standardStore.AddVectorsBatch(ctx, collection.Name, vectors); err != nil {
		b.Fatalf("Failed to add vectors: %v", err)
	}

	// Run queries with varying memory requirements
	topKValues := []int{10, 50, 100}

	for _, k := range topKValues {
		b.Run(fmt.Sprintf("TopK_%d", k), func(b *testing.B) {
			queryVector := generateTestVectors(1, dimension)[0]
			req := core.SearchRequest{
				Query: queryVector.Values,
				TopK:  k,
			}

			b.ResetTimer()
			successCount := 0
			throttledCount := 0

			for i := 0; i < b.N; i++ {
				_, err := optimizedStore.OptimizedSearch(ctx, collection.Name, req)
				if err != nil {
					if err.Error() == "rate limit exceeded" || err.Error() == "insufficient memory" {
						throttledCount++
					} else {
						b.Fatalf("Search failed: %v", err)
					}
				} else {
					successCount++
				}
			}

			b.Logf("Success rate: %.2f%% (throttled: %d)",
				float64(successCount)/float64(b.N)*100, throttledCount)
		})
	}
}
