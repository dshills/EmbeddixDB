package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// BenchmarkConfig contains configuration for benchmarks
type BenchmarkConfig struct {
	VectorDimension int
	NumVectors      int
	NumQueries      int
	TopK            int
	Parallelism     int
	BatchSize       int
}

// BenchmarkResult contains timing results
type BenchmarkResult struct {
	Operation      string
	TotalTime      time.Duration
	OperationCount int
	AvgLatency     time.Duration
	MinLatency     time.Duration
	MaxLatency     time.Duration
	P50Latency     time.Duration
	P95Latency     time.Duration
	P99Latency     time.Duration
	Throughput     float64 // operations per second
}

// Benchmark runs performance benchmarks on a vector store
type Benchmark struct {
	store  core.VectorStore
	config BenchmarkConfig
}

// NewBenchmark creates a new benchmark runner
func NewBenchmark(store core.VectorStore, config BenchmarkConfig) *Benchmark {
	return &Benchmark{
		store:  store,
		config: config,
	}
}

// RunAll runs all benchmarks
func (b *Benchmark) RunAll(ctx context.Context, collectionName string) ([]BenchmarkResult, error) {
	var results []BenchmarkResult
	
	// Get collection info for recreation
	collections, _ := b.store.ListCollections(ctx)
	var collection core.Collection
	for _, col := range collections {
		if col.Name == collectionName {
			collection = col
			break
		}
	}
	
	// Generate test vectors
	vectors := b.generateVectors(b.config.NumVectors)
	
	// Benchmark individual inserts
	fmt.Printf("Running individual insert benchmark (%d vectors)...\n", b.config.NumVectors)
	insertResult, err := b.benchmarkInserts(ctx, collectionName, vectors)
	if err != nil {
		return nil, err
	}
	results = append(results, insertResult)
	
	// Clean and recreate collection for batch test
	b.store.DeleteCollection(ctx, collectionName)
	b.store.CreateCollection(ctx, collection)
	
	// Benchmark batch inserts
	fmt.Printf("Running batch insert benchmark (batch size: %d)...\n", b.config.BatchSize)
	batchResult, err := b.benchmarkBatchInserts(ctx, collectionName, vectors)
	if err != nil {
		return nil, err
	}
	results = append(results, batchResult)
	
	// Benchmark searches
	fmt.Printf("Running search benchmark (%d queries)...\n", b.config.NumQueries)
	searchResult, err := b.benchmarkSearches(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	results = append(results, searchResult)
	
	// Benchmark concurrent searches
	fmt.Printf("Running concurrent search benchmark (parallelism: %d)...\n", b.config.Parallelism)
	concurrentResult, err := b.benchmarkConcurrentSearches(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	results = append(results, concurrentResult)
	
	// Benchmark gets
	fmt.Printf("Running get vector benchmark...\n")
	getResult, err := b.benchmarkGets(ctx, collectionName, vectors[:min(1000, len(vectors))])
	if err != nil {
		return nil, err
	}
	results = append(results, getResult)
	
	// Benchmark updates
	fmt.Printf("Running update benchmark...\n")
	updateResult, err := b.benchmarkUpdates(ctx, collectionName, vectors[:min(100, len(vectors))])
	if err != nil {
		return nil, err
	}
	results = append(results, updateResult)
	
	// Benchmark range searches
	fmt.Printf("Running range search benchmark...\n")
	rangeResult, err := b.benchmarkRangeSearch(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	results = append(results, rangeResult)
	
	return results, nil
}

// benchmarkInserts measures individual insert performance
func (b *Benchmark) benchmarkInserts(ctx context.Context, collection string, vectors []core.Vector) (BenchmarkResult, error) {
	latencies := make([]time.Duration, 0, len(vectors))
	
	start := time.Now()
	for _, vec := range vectors {
		opStart := time.Now()
		if err := b.store.AddVector(ctx, collection, vec); err != nil {
			return BenchmarkResult{}, fmt.Errorf("insert failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
	}
	totalTime := time.Since(start)
	
	return b.calculateResult("Individual Insert", totalTime, len(vectors), latencies), nil
}

// benchmarkBatchInserts measures batch insert performance
func (b *Benchmark) benchmarkBatchInserts(ctx context.Context, collection string, vectors []core.Vector) (BenchmarkResult, error) {
	latencies := make([]time.Duration, 0, len(vectors)/b.config.BatchSize+1)
	batches := 0
	
	start := time.Now()
	for i := 0; i < len(vectors); i += b.config.BatchSize {
		end := min(i+b.config.BatchSize, len(vectors))
		batch := vectors[i:end]
		
		opStart := time.Now()
		if err := b.store.AddVectorsBatch(ctx, collection, batch); err != nil {
			return BenchmarkResult{}, fmt.Errorf("batch insert failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
		batches++
	}
	totalTime := time.Since(start)
	
	result := b.calculateResult("Batch Insert", totalTime, batches, latencies)
	result.OperationCount = len(vectors) // Report total vectors inserted
	result.Throughput = float64(len(vectors)) / totalTime.Seconds()
	
	return result, nil
}

// benchmarkSearches measures search performance
func (b *Benchmark) benchmarkSearches(ctx context.Context, collection string) (BenchmarkResult, error) {
	queries := b.generateVectors(b.config.NumQueries)
	latencies := make([]time.Duration, 0, len(queries))
	
	start := time.Now()
	for _, query := range queries {
		searchReq := core.SearchRequest{
			Query:          query.Values,
			TopK:           b.config.TopK,
			IncludeVectors: false,
		}
		
		opStart := time.Now()
		_, err := b.store.Search(ctx, collection, searchReq)
		if err != nil {
			return BenchmarkResult{}, fmt.Errorf("search failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
	}
	totalTime := time.Since(start)
	
	return b.calculateResult("Search", totalTime, len(queries), latencies), nil
}

// benchmarkConcurrentSearches measures concurrent search performance
func (b *Benchmark) benchmarkConcurrentSearches(ctx context.Context, collection string) (BenchmarkResult, error) {
	queries := b.generateVectors(b.config.NumQueries)
	latencies := make([]time.Duration, len(queries))
	var mu sync.Mutex
	var wg sync.WaitGroup
	
	// Create a channel to limit concurrency
	sem := make(chan struct{}, b.config.Parallelism)
	
	start := time.Now()
	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q core.Vector) {
			defer wg.Done()
			
			sem <- struct{}{} // Acquire
			defer func() { <-sem }() // Release
			
			searchReq := core.SearchRequest{
				Query:          q.Values,
				TopK:           b.config.TopK,
				IncludeVectors: false,
			}
			
			opStart := time.Now()
			_, err := b.store.Search(ctx, collection, searchReq)
			if err == nil {
				mu.Lock()
				latencies[idx] = time.Since(opStart)
				mu.Unlock()
			}
		}(i, query)
	}
	
	wg.Wait()
	totalTime := time.Since(start)
	
	// Filter out zero latencies (from errors)
	validLatencies := make([]time.Duration, 0, len(latencies))
	for _, lat := range latencies {
		if lat > 0 {
			validLatencies = append(validLatencies, lat)
		}
	}
	
	return b.calculateResult("Concurrent Search", totalTime, len(validLatencies), validLatencies), nil
}

// benchmarkGets measures get performance
func (b *Benchmark) benchmarkGets(ctx context.Context, collection string, vectors []core.Vector) (BenchmarkResult, error) {
	latencies := make([]time.Duration, 0, len(vectors))
	
	start := time.Now()
	for _, vec := range vectors {
		opStart := time.Now()
		_, err := b.store.GetVector(ctx, collection, vec.ID)
		if err != nil {
			return BenchmarkResult{}, fmt.Errorf("get failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
	}
	totalTime := time.Since(start)
	
	return b.calculateResult("Get Vector", totalTime, len(vectors), latencies), nil
}

// benchmarkUpdates measures update performance
func (b *Benchmark) benchmarkUpdates(ctx context.Context, collection string, vectors []core.Vector) (BenchmarkResult, error) {
	latencies := make([]time.Duration, 0, len(vectors))
	
	// Modify vectors slightly for update
	for i := range vectors {
		for j := range vectors[i].Values {
			vectors[i].Values[j] *= 1.1
		}
		vectors[i].Metadata["updated"] = "true"
	}
	
	start := time.Now()
	for _, vec := range vectors {
		opStart := time.Now()
		if err := b.store.UpdateVector(ctx, collection, vec); err != nil {
			return BenchmarkResult{}, fmt.Errorf("update failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
	}
	totalTime := time.Since(start)
	
	return b.calculateResult("Update Vector", totalTime, len(vectors), latencies), nil
}

// benchmarkRangeSearch measures range search performance
func (b *Benchmark) benchmarkRangeSearch(ctx context.Context, collection string) (BenchmarkResult, error) {
	queries := b.generateVectors(b.config.NumQueries)
	latencies := make([]time.Duration, 0, len(queries))
	totalFound := 0
	
	// Use a moderate radius for benchmarking
	radius := float32(0.5)
	
	start := time.Now()
	for _, query := range queries {
		searchReq := core.RangeSearchRequest{
			Query:          query.Values,
			Radius:         radius,
			IncludeVectors: false,
		}
		
		opStart := time.Now()
		result, err := b.store.RangeSearch(ctx, collection, searchReq)
		if err != nil {
			return BenchmarkResult{}, fmt.Errorf("range search failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
		totalFound += result.Count
	}
	totalTime := time.Since(start)
	
	result := b.calculateResult("Range Search (r=0.5)", totalTime, len(queries), latencies)
	
	// Log average results found
	avgFound := float64(totalFound) / float64(len(queries))
	fmt.Printf("   Average vectors found per query: %.2f\n", avgFound)
	
	return result, nil
}

// generateVectors creates random test vectors
func (b *Benchmark) generateVectors(count int) []core.Vector {
	vectors := make([]core.Vector, count)
	for i := 0; i < count; i++ {
		values := make([]float32, b.config.VectorDimension)
		for j := 0; j < b.config.VectorDimension; j++ {
			values[j] = rand.Float32()
		}
		
		vectors[i] = core.Vector{
			ID:     fmt.Sprintf("vec_%d", i),
			Values: values,
			Metadata: map[string]string{
				"index": fmt.Sprintf("%d", i),
				"type":  "benchmark",
			},
		}
	}
	return vectors
}

// calculateResult computes statistics from latencies
func (b *Benchmark) calculateResult(operation string, totalTime time.Duration, count int, latencies []time.Duration) BenchmarkResult {
	if len(latencies) == 0 {
		return BenchmarkResult{Operation: operation}
	}
	
	// Sort latencies for percentile calculation
	sortedLatencies := make([]time.Duration, len(latencies))
	copy(sortedLatencies, latencies)
	sortDurations(sortedLatencies)
	
	// Calculate statistics
	var sum time.Duration
	min := sortedLatencies[0]
	max := sortedLatencies[len(sortedLatencies)-1]
	
	for _, lat := range latencies {
		sum += lat
	}
	
	avg := sum / time.Duration(len(latencies))
	p50 := sortedLatencies[len(sortedLatencies)*50/100]
	p95 := sortedLatencies[len(sortedLatencies)*95/100]
	p99 := sortedLatencies[len(sortedLatencies)*99/100]
	
	throughput := float64(count) / totalTime.Seconds()
	
	return BenchmarkResult{
		Operation:      operation,
		TotalTime:      totalTime,
		OperationCount: count,
		AvgLatency:     avg,
		MinLatency:     min,
		MaxLatency:     max,
		P50Latency:     p50,
		P95Latency:     p95,
		P99Latency:     p99,
		Throughput:     throughput,
	}
}

// sortDurations sorts a slice of durations
func sortDurations(durations []time.Duration) {
	for i := 0; i < len(durations); i++ {
		for j := i + 1; j < len(durations); j++ {
			if durations[i] > durations[j] {
				durations[i], durations[j] = durations[j], durations[i]
			}
		}
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// PrintResults prints benchmark results in a formatted table
func PrintResults(results []BenchmarkResult) {
	fmt.Println("\n=== Benchmark Results ===")
	fmt.Printf("%-20s %10s %10s %10s %10s %10s %10s %12s\n", 
		"Operation", "Count", "Avg", "Min", "Max", "P95", "P99", "Throughput")
	fmt.Println(strings.Repeat("-", 102))
	
	for _, r := range results {
		fmt.Printf("%-20s %10d %10s %10s %10s %10s %10s %12.2f/s\n",
			r.Operation,
			r.OperationCount,
			formatDuration(r.AvgLatency),
			formatDuration(r.MinLatency),
			formatDuration(r.MaxLatency),
			formatDuration(r.P95Latency),
			formatDuration(r.P99Latency),
			r.Throughput,
		)
	}
}

// formatDuration formats a duration for display
func formatDuration(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%dns", d.Nanoseconds())
	} else if d < time.Millisecond {
		return fmt.Sprintf("%.1fµs", float64(d.Nanoseconds())/1000)
	} else if d < time.Second {
		return fmt.Sprintf("%.1fms", float64(d.Nanoseconds())/1000000)
	}
	return fmt.Sprintf("%.2fs", d.Seconds())
}