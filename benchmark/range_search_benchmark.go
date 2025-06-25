package benchmark

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// RangeSearchBenchmark performs range search benchmarks
type RangeSearchBenchmark struct {
	benchmark *Benchmark
}

// NewRangeSearchBenchmark creates a new range search benchmark
func NewRangeSearchBenchmark(b *Benchmark) *RangeSearchBenchmark {
	return &RangeSearchBenchmark{
		benchmark: b,
	}
}

// RunRangeSearchBenchmarks runs various range search benchmarks
func (r *RangeSearchBenchmark) RunRangeSearchBenchmarks(ctx context.Context, collectionName string) ([]BenchmarkResult, error) {
	var results []BenchmarkResult
	
	// Test different radius values
	radiusTests := []struct {
		name   string
		radius float32
	}{
		{"Small Radius (0.1)", 0.1},
		{"Medium Radius (0.5)", 0.5},
		{"Large Radius (1.0)", 1.0},
		{"Very Large Radius (2.0)", 2.0},
	}
	
	for _, rt := range radiusTests {
		fmt.Printf("Running range search benchmark: %s\n", rt.name)
		result, err := r.benchmarkRangeSearch(ctx, collectionName, rt.radius, 0)
		if err != nil {
			return nil, err
		}
		result.Operation = fmt.Sprintf("Range Search - %s", rt.name)
		results = append(results, result)
	}
	
	// Test with different limits
	limitTests := []struct {
		name  string
		limit int
	}{
		{"No Limit", 0},
		{"Limit 10", 10},
		{"Limit 100", 100},
		{"Limit 1000", 1000},
	}
	
	for _, lt := range limitTests {
		fmt.Printf("Running range search benchmark with limit: %s\n", lt.name)
		result, err := r.benchmarkRangeSearch(ctx, collectionName, 0.5, lt.limit)
		if err != nil {
			return nil, err
		}
		result.Operation = fmt.Sprintf("Range Search - %s", lt.name)
		results = append(results, result)
	}
	
	// Test with filters
	fmt.Println("Running range search benchmark with metadata filter")
	result, err := r.benchmarkRangeSearchWithFilter(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	results = append(results, result)
	
	return results, nil
}

// benchmarkRangeSearch measures range search performance
func (r *RangeSearchBenchmark) benchmarkRangeSearch(ctx context.Context, collection string, radius float32, limit int) (BenchmarkResult, error) {
	queries := r.benchmark.generateVectors(r.benchmark.config.NumQueries)
	latencies := make([]time.Duration, 0, len(queries))
	totalFound := 0
	
	start := time.Now()
	for _, query := range queries {
		searchReq := core.RangeSearchRequest{
			Query:          query.Values,
			Radius:         radius,
			IncludeVectors: false,
			Limit:          limit,
		}
		
		opStart := time.Now()
		result, err := r.benchmark.store.RangeSearch(ctx, collection, searchReq)
		if err != nil {
			return BenchmarkResult{}, fmt.Errorf("range search failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
		totalFound += result.Count
	}
	totalTime := time.Since(start)
	
	result := r.benchmark.calculateResult("Range Search", totalTime, len(queries), latencies)
	
	// Add average results found as additional info
	avgFound := float64(totalFound) / float64(len(queries))
	fmt.Printf("   Average vectors found per query: %.2f\n", avgFound)
	
	return result, nil
}

// benchmarkRangeSearchWithFilter measures filtered range search performance
func (r *RangeSearchBenchmark) benchmarkRangeSearchWithFilter(ctx context.Context, collection string) (BenchmarkResult, error) {
	queries := r.benchmark.generateVectors(r.benchmark.config.NumQueries)
	latencies := make([]time.Duration, 0, len(queries))
	
	// Use a common filter
	filter := map[string]string{"type": "benchmark"}
	
	start := time.Now()
	for _, query := range queries {
		searchReq := core.RangeSearchRequest{
			Query:          query.Values,
			Radius:         0.5,
			Filter:         filter,
			IncludeVectors: false,
		}
		
		opStart := time.Now()
		_, err := r.benchmark.store.RangeSearch(ctx, collection, searchReq)
		if err != nil {
			return BenchmarkResult{}, fmt.Errorf("filtered range search failed: %w", err)
		}
		latencies = append(latencies, time.Since(opStart))
	}
	totalTime := time.Since(start)
	
	return r.benchmark.calculateResult("Range Search with Filter", totalTime, len(queries), latencies), nil
}

// CompareSearchMethods compares KNN search vs Range search performance
func (r *RangeSearchBenchmark) CompareSearchMethods(ctx context.Context, collectionName string) error {
	fmt.Println("\n=== Search Method Comparison ===")
	
	queries := r.benchmark.generateVectors(100) // Use fewer queries for comparison
	
	// Benchmark KNN search
	fmt.Println("Running KNN search benchmark...")
	knnLatencies := make([]time.Duration, 0, len(queries))
	knnStart := time.Now()
	
	for _, query := range queries {
		searchReq := core.SearchRequest{
			Query:          query.Values,
			TopK:           10,
			IncludeVectors: false,
		}
		
		opStart := time.Now()
		_, err := r.benchmark.store.Search(ctx, collectionName, searchReq)
		if err != nil {
			return fmt.Errorf("KNN search failed: %w", err)
		}
		knnLatencies = append(knnLatencies, time.Since(opStart))
	}
	knnTotalTime := time.Since(knnStart)
	
	// Benchmark Range search
	fmt.Println("Running Range search benchmark...")
	rangeLatencies := make([]time.Duration, 0, len(queries))
	rangeStart := time.Now()
	
	for _, query := range queries {
		searchReq := core.RangeSearchRequest{
			Query:          query.Values,
			Radius:         0.5,
			IncludeVectors: false,
		}
		
		opStart := time.Now()
		_, err := r.benchmark.store.RangeSearch(ctx, collectionName, searchReq)
		if err != nil {
			return fmt.Errorf("Range search failed: %w", err)
		}
		rangeLatencies = append(rangeLatencies, time.Since(opStart))
	}
	rangeTotalTime := time.Since(rangeStart)
	
	// Calculate and display results
	knnResult := r.benchmark.calculateResult("KNN Search (k=10)", knnTotalTime, len(queries), knnLatencies)
	rangeResult := r.benchmark.calculateResult("Range Search (r=0.5)", rangeTotalTime, len(queries), rangeLatencies)
	
	fmt.Println("\nComparison Results:")
	fmt.Printf("%-20s %10s %10s %10s %12s\n", "Method", "Avg", "P95", "P99", "Throughput")
	fmt.Println(strings.Repeat("-", 64))
	
	fmt.Printf("%-20s %10s %10s %10s %12.2f/s\n",
		knnResult.Operation,
		formatDuration(knnResult.AvgLatency),
		formatDuration(knnResult.P95Latency),
		formatDuration(knnResult.P99Latency),
		knnResult.Throughput,
	)
	
	fmt.Printf("%-20s %10s %10s %10s %12.2f/s\n",
		rangeResult.Operation,
		formatDuration(rangeResult.AvgLatency),
		formatDuration(rangeResult.P95Latency),
		formatDuration(rangeResult.P99Latency),
		rangeResult.Throughput,
	)
	
	// Performance comparison
	speedup := float64(knnResult.AvgLatency) / float64(rangeResult.AvgLatency)
	if speedup > 1 {
		fmt.Printf("\nRange search is %.2fx slower than KNN search\n", speedup)
	} else {
		fmt.Printf("\nRange search is %.2fx faster than KNN search\n", 1/speedup)
	}
	
	return nil
}

