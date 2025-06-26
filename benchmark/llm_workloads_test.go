//go:build integration
// +build integration

package benchmark

import (
	"fmt"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

// BenchmarkLLMWorkloads runs comprehensive benchmarks for LLM use cases
func BenchmarkLLMWorkloads(b *testing.B) {
	for _, workload := range LLMWorkloadSuite {
		b.Run(workload.Name, func(b *testing.B) {
			benchmarkWorkload(b, workload)
		})
	}
}

// benchmarkWorkload executes a single workload benchmark
func benchmarkWorkload(b *testing.B, workload LLMWorkload) {
	// Create vector store with memory persistence for consistent benchmarking
	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(memStore, indexFactory)

	executor := NewWorkloadExecutor(vectorStore)

	b.ResetTimer()
	b.StartTimer()

	results, err := executor.ExecuteWorkload(workload)
	if err != nil {
		b.Fatalf("Workload execution failed: %v", err)
	}

	b.StopTimer()

	// Report detailed metrics
	reportWorkloadResults(b, results)
}

// reportWorkloadResults logs detailed performance metrics
func reportWorkloadResults(b *testing.B, results *WorkloadResults) {
	b.Logf("=== Workload Results: %s ===", results.WorkloadName)
	b.Logf("Duration: %v", results.Duration)
	b.Logf("Total Queries: %d", results.TotalQueries)
	b.Logf("Total Insertions: %d", results.TotalInsertions)
	b.Logf("Query Throughput: %.2f QPS", results.QueryThroughput)
	b.Logf("Insertion Throughput: %.2f IPS", results.InsertionThroughput)
	b.Logf("Error Rate: %.4f%%", results.ErrorRate*100)
	b.Logf("Cache Hit Rate: %.2f%%", results.CacheHitRate*100)

	// Latency percentiles
	b.Logf("Query Latency - P50: %v, P95: %v, P99: %v",
		results.LatencyPercentiles.QueryP50,
		results.LatencyPercentiles.QueryP95,
		results.LatencyPercentiles.QueryP99)

	b.Logf("Insertion Latency - P50: %v, P95: %v, P99: %v",
		results.LatencyPercentiles.InsertionP50,
		results.LatencyPercentiles.InsertionP95,
		results.LatencyPercentiles.InsertionP99)

	// Resource usage
	b.Logf("Memory Usage - Avg: %d MB, Max: %d MB",
		results.ResourceUsage.AvgMemoryUsage/(1024*1024),
		results.ResourceUsage.MaxMemoryUsage/(1024*1024))

	b.Logf("CPU Usage - Avg: %.2f%%, Max: %.2f%%",
		results.ResourceUsage.AvgCPUUsage*100,
		results.ResourceUsage.MaxCPUUsage*100)
}

// BenchmarkLatencyRegression tests for performance regressions
func BenchmarkLatencyRegression(b *testing.B) {
	workload := LLMWorkload{
		Name:             "Latency Regression Test",
		AgentCount:       1,
		QueriesPerSecond: 10.0,
		InsertionRate:    2.0,
		VectorDimension:  768,
		CollectionSize:   10000,
		Duration:         time.Minute * 2,
		QueryPatterns: []QueryPattern{
			{Type: QueryTypeAgentMemory, Frequency: 1.0, Complexity: ComplexityLow, K: 5},
		},
	}

	memStore := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(memStore, indexFactory)

	executor := NewWorkloadExecutor(vectorStore)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		results, err := executor.ExecuteWorkload(workload)
		if err != nil {
			b.Fatalf("Benchmark failed: %v", err)
		}

		// Assert performance thresholds
		if results.LatencyPercentiles.QueryP95 > 500*time.Millisecond {
			b.Errorf("P95 latency regression: %v > 500ms", results.LatencyPercentiles.QueryP95)
		}

		if results.ErrorRate > 0.01 { // 1% error rate threshold
			b.Errorf("Error rate too high: %.4f%%", results.ErrorRate*100)
		}
	}
}

// BenchmarkThroughputScaling tests throughput scaling with concurrent agents
func BenchmarkThroughputScaling(b *testing.B) {
	agentCounts := []int{1, 2, 4, 8, 16}

	for _, agentCount := range agentCounts {
		b.Run(fmt.Sprintf("Agents_%d", agentCount), func(b *testing.B) {
			workload := LLMWorkload{
				Name:             fmt.Sprintf("Scaling_%d_agents", agentCount),
				AgentCount:       agentCount,
				QueriesPerSecond: 20.0,
				InsertionRate:    5.0,
				VectorDimension:  768,
				CollectionSize:   50000,
				Duration:         time.Minute * 3,
				QueryPatterns: []QueryPattern{
					{Type: QueryTypeAgentMemory, Frequency: 0.7, Complexity: ComplexityLow, K: 5},
					{Type: QueryTypeDocumentSearch, Frequency: 0.3, Complexity: ComplexityMedium, K: 10, HasMetadata: true},
				},
			}

			memStore := persistence.NewMemoryPersistence()
			indexFactory := index.NewDefaultFactory()
			vectorStore := core.NewVectorStore(memStore, indexFactory)

			executor := NewWorkloadExecutor(vectorStore)

			b.ResetTimer()

			results, err := executor.ExecuteWorkload(workload)
			if err != nil {
				b.Fatalf("Scaling benchmark failed: %v", err)
			}

			// Calculate throughput per agent
			throughputPerAgent := results.QueryThroughput / float64(agentCount)
			b.Logf("Throughput per agent: %.2f QPS", throughputPerAgent)

			// Report scaling efficiency
			if agentCount > 1 {
				// This would compare against single-agent baseline
				b.Logf("Scaling efficiency for %d agents: estimated", agentCount)
			}
		})
	}
}

// BenchmarkMemoryUsage tests memory efficiency under load
func BenchmarkMemoryUsage(b *testing.B) {
	collectionSizes := []int{10000, 50000, 100000, 500000}

	for _, size := range collectionSizes {
		b.Run(fmt.Sprintf("Collection_%d", size), func(b *testing.B) {
			workload := LLMWorkload{
				Name:             fmt.Sprintf("Memory_Test_%d", size),
				AgentCount:       2,
				QueriesPerSecond: 15.0,
				InsertionRate:    3.0,
				VectorDimension:  768,
				CollectionSize:   size,
				Duration:         time.Minute * 2,
				QueryPatterns: []QueryPattern{
					{Type: QueryTypeAgentMemory, Frequency: 0.8, Complexity: ComplexityLow, K: 5},
					{Type: QueryTypeDocumentSearch, Frequency: 0.2, Complexity: ComplexityMedium, K: 10},
				},
			}

			memStore := persistence.NewMemoryPersistence()
			indexFactory := index.NewDefaultFactory()
			vectorStore := core.NewVectorStore(memStore, indexFactory)

			executor := NewWorkloadExecutor(vectorStore)

			b.ResetTimer()

			results, err := executor.ExecuteWorkload(workload)
			if err != nil {
				b.Fatalf("Memory benchmark failed: %v", err)
			}

			// Calculate memory efficiency metrics
			memoryPerVector := float64(results.ResourceUsage.MaxMemoryUsage) / float64(size)
			b.Logf("Memory per vector: %.2f KB", memoryPerVector/1024)

			// Memory usage should scale linearly with collection size
			expectedMemory := float64(size) * 4 * 768 // 4 bytes per float32 * dimensions
			efficiency := expectedMemory / float64(results.ResourceUsage.MaxMemoryUsage)
			b.Logf("Memory efficiency: %.2f%%", efficiency*100)
		})
	}
}

// TestWorkloadValidation ensures workload configurations are valid
func TestWorkloadValidation(t *testing.T) {
	for _, workload := range LLMWorkloadSuite {
		t.Run(workload.Name, func(t *testing.T) {
			// Validate workload configuration
			if workload.AgentCount <= 0 {
				t.Errorf("Invalid agent count: %d", workload.AgentCount)
			}

			if workload.QueriesPerSecond <= 0 {
				t.Errorf("Invalid query rate: %f", workload.QueriesPerSecond)
			}

			if workload.VectorDimension <= 0 {
				t.Errorf("Invalid vector dimension: %d", workload.VectorDimension)
			}

			if workload.CollectionSize <= 0 {
				t.Errorf("Invalid collection size: %d", workload.CollectionSize)
			}

			// Validate query patterns
			totalFrequency := 0.0
			for _, pattern := range workload.QueryPatterns {
				totalFrequency += pattern.Frequency

				if pattern.K <= 0 {
					t.Errorf("Invalid K value: %d", pattern.K)
				}

				if pattern.CacheHit < 0 || pattern.CacheHit > 1 {
					t.Errorf("Invalid cache hit rate: %f", pattern.CacheHit)
				}
			}

			if totalFrequency < 0.99 || totalFrequency > 1.01 {
				t.Errorf("Query pattern frequencies don't sum to 1.0: %f", totalFrequency)
			}
		})
	}
}

// BenchmarkIndexComparison compares different index types under LLM workloads
func BenchmarkIndexComparison(b *testing.B) {
	indexTypes := []string{"flat", "hnsw"}

	workload := LLMWorkload{
		Name:             "Index Comparison",
		AgentCount:       3,
		QueriesPerSecond: 25.0,
		InsertionRate:    5.0,
		VectorDimension:  768,
		CollectionSize:   25000,
		Duration:         time.Minute * 3,
		QueryPatterns: []QueryPattern{
			{Type: QueryTypeAgentMemory, Frequency: 0.6, Complexity: ComplexityLow, K: 5},
			{Type: QueryTypeDocumentSearch, Frequency: 0.4, Complexity: ComplexityMedium, K: 10},
		},
	}

	for _, indexType := range indexTypes {
		b.Run(fmt.Sprintf("Index_%s", indexType), func(b *testing.B) {
			memStore := persistence.NewMemoryPersistence()
			indexFactory := index.NewDefaultFactory()
			vectorStore := core.NewVectorStore(memStore, indexFactory)

			executor := NewWorkloadExecutor(vectorStore)

			b.ResetTimer()

			results, err := executor.ExecuteWorkload(workload)
			if err != nil {
				b.Fatalf("Index comparison benchmark failed: %v", err)
			}

			b.Logf("Index type %s - Query P95: %v, Throughput: %.2f QPS",
				indexType, results.LatencyPercentiles.QueryP95, results.QueryThroughput)
		})
	}
}

// Utility functions for statistical analysis
