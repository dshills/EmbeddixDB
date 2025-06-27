package benchmark

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
)

// HierarchicalBenchmark benchmarks hierarchical HNSW performance
type HierarchicalBenchmark struct {
	dimension      int
	numVectors     int
	numQueries     int
	k              int
	index          core.Index
	vectors        []core.Vector
	queries        [][]float32
	distanceMetric core.DistanceMetric
}

// NewHierarchicalBenchmark creates a new hierarchical benchmark
func NewHierarchicalBenchmark(dimension, numVectors, numQueries, k int, metric core.DistanceMetric) *HierarchicalBenchmark {
	return &HierarchicalBenchmark{
		dimension:      dimension,
		numVectors:     numVectors,
		numQueries:     numQueries,
		k:              k,
		distanceMetric: metric,
	}
}

// Setup initializes the benchmark data
func (hb *HierarchicalBenchmark) Setup() error {
	// Create hierarchical index with optimized config
	config := index.DefaultHierarchicalConfig(hb.dimension)
	config.NumFineClusters = int(math.Sqrt(float64(hb.numVectors)))
	config.MaxVectorsPerCluster = hb.numVectors/config.NumFineClusters + 100
	config.EnableIncrementalUpdates = true
	config.EnableQualityMonitoring = true

	hb.index = index.NewHierarchicalHNSW(hb.dimension, hb.distanceMetric, config)

	// Generate vectors
	hb.vectors = hb.generateVectors()

	// Generate queries
	hb.queries = hb.generateQueries()

	return nil
}

// RunInsertBenchmark benchmarks vector insertion
func (hb *HierarchicalBenchmark) RunInsertBenchmark() BenchmarkResult {
	startTime := time.Now()

	errors := []string{}
	for i, vec := range hb.vectors {
		if err := hb.index.Add(vec); err != nil {
			errors = append(errors, fmt.Sprintf("Failed to add vector %d: %v", i, err))
		}
	}

	duration := time.Since(startTime)

	result := BenchmarkResult{
		Operation:      "HierarchicalHNSW Insert",
		TotalTime:      duration,
		OperationCount: hb.numVectors,
		AvgLatency:     duration / time.Duration(hb.numVectors),
		Throughput:     float64(hb.numVectors) / duration.Seconds(),
	}

	// Log memory usage if needed
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory used: %.2f MB\n", float64(m.Alloc)/1024/1024)

	return result
}

// RunSearchBenchmark benchmarks search performance
func (hb *HierarchicalBenchmark) RunSearchBenchmark() BenchmarkResult {
	// Warm up
	for i := 0; i < 10 && i < len(hb.queries); i++ {
		hb.index.Search(hb.queries[i], hb.k, nil)
	}

	// Measure search time
	startTime := time.Now()
	totalResults := 0
	errors := []string{}

	for _, query := range hb.queries {
		results, err := hb.index.Search(query, hb.k, nil)
		if err != nil {
			errors = append(errors, fmt.Sprintf("Search failed: %v", err))
		} else {
			totalResults += len(results)
		}
	}

	duration := time.Since(startTime)

	result := BenchmarkResult{
		Operation:      "HierarchicalHNSW Search",
		TotalTime:      duration,
		OperationCount: hb.numQueries,
		AvgLatency:     duration / time.Duration(hb.numQueries),
		Throughput:     float64(hb.numQueries) / duration.Seconds(),
	}

	// Log errors if any
	if len(errors) > 0 {
		fmt.Printf("Search errors: %v\n", errors)
	}

	return result
}

// RunConcurrentSearchBenchmark benchmarks concurrent search performance
func (hb *HierarchicalBenchmark) RunConcurrentSearchBenchmark(numThreads int) BenchmarkResult {

	queriesPerThread := hb.numQueries / numThreads
	var wg sync.WaitGroup
	var mu sync.Mutex
	totalLatency := 0.0
	totalResults := 0
	errors := []string{}

	startTime := time.Now()

	for t := 0; t < numThreads; t++ {
		wg.Add(1)
		go func(threadID int) {
			defer wg.Done()

			startIdx := threadID * queriesPerThread
			endIdx := startIdx + queriesPerThread
			if threadID == numThreads-1 {
				endIdx = hb.numQueries
			}

			threadLatency := 0.0
			threadResults := 0

			for i := startIdx; i < endIdx; i++ {
				queryStart := time.Now()
				results, err := hb.index.Search(hb.queries[i], hb.k, nil)
				queryLatency := time.Since(queryStart).Seconds() * 1000

				if err != nil {
					mu.Lock()
					errors = append(errors, fmt.Sprintf("Thread %d search failed: %v", threadID, err))
					mu.Unlock()
				} else {
					threadLatency += queryLatency
					threadResults += len(results)
				}
			}

			mu.Lock()
			totalLatency += threadLatency
			totalResults += threadResults
			mu.Unlock()
		}(t)
	}

	wg.Wait()

	duration := time.Since(startTime)

	result := BenchmarkResult{
		Operation:      fmt.Sprintf("HierarchicalHNSW Concurrent Search (%d threads)", numThreads),
		TotalTime:      duration,
		OperationCount: hb.numQueries,
		AvgLatency:     time.Duration(totalLatency/float64(hb.numQueries)) * time.Millisecond,
		Throughput:     float64(totalResults) / duration.Seconds(),
	}

	return result
}

// RunRecallBenchmark measures search recall quality
func (hb *HierarchicalBenchmark) RunRecallBenchmark(groundTruth [][]int) BenchmarkResult {

	startTime := time.Now()
	totalRecall := 0.0
	errors := []string{}

	for i, query := range hb.queries[:len(groundTruth)] {
		results, err := hb.index.Search(query, hb.k, nil)
		if err != nil {
			errors = append(errors, fmt.Sprintf("Search %d failed: %v", i, err))
			continue
		}

		// Calculate recall
		found := 0
		resultMap := make(map[string]bool)
		for _, r := range results {
			resultMap[r.ID] = true
		}

		for _, truthIdx := range groundTruth[i][:hb.k] {
			if truthIdx < len(hb.vectors) {
				if resultMap[hb.vectors[truthIdx].ID] {
					found++
				}
			}
		}

		recall := float64(found) / float64(hb.k)
		totalRecall += recall
	}

	duration := time.Since(startTime)
	avgRecall := totalRecall / float64(len(groundTruth))

	// Log recall info
	fmt.Printf("Average recall@%d: %.3f\n", hb.k, avgRecall)
	if len(errors) > 0 {
		fmt.Printf("Recall errors: %v\n", errors)
	}

	return BenchmarkResult{
		Operation:      "HierarchicalHNSW Recall",
		TotalTime:      duration,
		OperationCount: len(groundTruth),
		AvgLatency:     duration / time.Duration(len(groundTruth)),
		Throughput:     float64(len(groundTruth)) / duration.Seconds(),
	}
}

// CompareWithStandardHNSW compares performance with standard HNSW
func (hb *HierarchicalBenchmark) CompareWithStandardHNSW() ComparisonResult {
	comparison := ComparisonResult{
		Name: "Hierarchical vs Standard HNSW",
	}

	// Create standard HNSW
	standardConfig := index.DefaultHNSWConfig()
	standardIndex := index.NewHNSWIndex(hb.dimension, hb.distanceMetric, standardConfig)

	// Insert vectors into standard index
	startTime := time.Now()
	for _, vec := range hb.vectors {
		standardIndex.Add(vec)
	}
	standardInsertTime := time.Since(startTime)

	// Compare search performance
	hierarchicalSearchTime := 0.0
	standardSearchTime := 0.0

	for i := 0; i < 100 && i < len(hb.queries); i++ {
		// Hierarchical search
		start := time.Now()
		hb.index.Search(hb.queries[i], hb.k, nil)
		hierarchicalSearchTime += time.Since(start).Seconds()

		// Standard search
		start = time.Now()
		standardIndex.Search(hb.queries[i], hb.k, nil)
		standardSearchTime += time.Since(start).Seconds()
	}

	comparison.HierarchicalInsertTime = hb.RunInsertBenchmark().TotalTime
	comparison.StandardInsertTime = standardInsertTime
	comparison.HierarchicalSearchTime = hierarchicalSearchTime
	comparison.StandardSearchTime = standardSearchTime
	comparison.SpeedupFactor = standardSearchTime / hierarchicalSearchTime

	// Memory comparison
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	comparison.HierarchicalMemoryMB = float64(m.Alloc) / 1024 / 1024

	return comparison
}

// Helper methods

func (hb *HierarchicalBenchmark) generateVectors() []core.Vector {
	vectors := make([]core.Vector, hb.numVectors)

	// Generate mixed distribution: clustered + random
	numClusters := int(math.Sqrt(float64(hb.numVectors)))
	vectorsPerCluster := hb.numVectors / numClusters

	idx := 0
	for c := 0; c < numClusters; c++ {
		// Generate cluster center
		center := make([]float32, hb.dimension)
		for i := range center {
			center[i] = rand.Float32()*2 - 1
		}

		// Generate vectors around center
		for v := 0; v < vectorsPerCluster && idx < hb.numVectors; v++ {
			values := make([]float32, hb.dimension)
			for i := range values {
				values[i] = center[i] + (rand.Float32()*0.4 - 0.2) // ±0.2 noise
			}

			vectors[idx] = core.Vector{
				ID:     fmt.Sprintf("vec_%d", idx),
				Values: values,
			}
			idx++
		}
	}

	// Fill remaining with random vectors
	for ; idx < hb.numVectors; idx++ {
		values := make([]float32, hb.dimension)
		for i := range values {
			values[i] = rand.Float32()*2 - 1
		}

		vectors[idx] = core.Vector{
			ID:     fmt.Sprintf("vec_%d", idx),
			Values: values,
		}
	}

	return vectors
}

func (hb *HierarchicalBenchmark) generateQueries() [][]float32 {
	queries := make([][]float32, hb.numQueries)

	for i := range queries {
		query := make([]float32, hb.dimension)

		// Mix of queries: some near existing vectors, some random
		if i < hb.numQueries/2 && i < len(hb.vectors) {
			// Query near existing vector
			baseVector := hb.vectors[rand.Intn(len(hb.vectors))].Values
			for j := range query {
				query[j] = baseVector[j] + (rand.Float32()*0.2 - 0.1)
			}
		} else {
			// Random query
			for j := range query {
				query[j] = rand.Float32()*2 - 1
			}
		}

		queries[i] = query
	}

	return queries
}

// BenchmarkResult contains benchmark results
// HierarchicalBenchmarkResult extends the base benchmark result for hierarchical testing
type HierarchicalBenchmarkResult struct {
	BenchmarkResult
	NumClusters     int
	ClusterSizes    []int
	ClusterQuality  float64
	RoutingOverhead float64
}

// ComparisonResult contains comparison results
type ComparisonResult struct {
	Name                   string
	HierarchicalInsertTime time.Duration
	StandardInsertTime     time.Duration
	HierarchicalSearchTime float64
	StandardSearchTime     float64
	SpeedupFactor          float64
	HierarchicalMemoryMB   float64
	StandardMemoryMB       float64
}
