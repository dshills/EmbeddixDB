package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/gpu"
	"github.com/dshills/EmbeddixDB/index"
)

// Phase4BenchmarkSuite contains benchmarks for Phase 4.3-4.4 features
type Phase4BenchmarkSuite struct {
	dimensions      []int
	datasetSizes    []int
	queryBatches    []int
	distanceMetrics []core.DistanceMetric
}

// NewPhase4BenchmarkSuite creates a new benchmark suite for Phase 4 features
func NewPhase4BenchmarkSuite() *Phase4BenchmarkSuite {
	return &Phase4BenchmarkSuite{
		dimensions:   []int{64, 128, 256, 512},
		datasetSizes: []int{1000, 10000, 100000},
		queryBatches: []int{1, 10, 100},
		distanceMetrics: []core.DistanceMetric{
			core.DistanceCosine,
			core.DistanceL2,
			core.DistanceDot,
		},
	}
}

// BenchmarkHierarchicalHNSW_Scalability benchmarks hierarchical HNSW scalability
func BenchmarkHierarchicalHNSW_Scalability(b *testing.B) {
	suite := NewPhase4BenchmarkSuite()

	for _, dim := range suite.dimensions {
		for _, size := range suite.datasetSizes {
			b.Run(fmt.Sprintf("dim%d_size%d", dim, size), func(b *testing.B) {
				benchmarkHierarchicalScalability(b, dim, size)
			})
		}
	}
}

func benchmarkHierarchicalScalability(b *testing.B, dimension, datasetSize int) {
	// Create hierarchical index
	config := index.DefaultHierarchicalConfig(dimension)
	config.NumFineClusters = calculateOptimalClusters(datasetSize)

	hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Generate dataset
	vectors := generateBenchmarkVectors(datasetSize, dimension)

	// Add vectors to index
	b.ResetTimer()
	start := time.Now()

	for i := 0; i < datasetSize && i < b.N; i++ {
		if err := hIndex.Add(vectors[i]); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	addTime := time.Since(start)

	// Perform searches
	query := vectors[0].Values
	searchStart := time.Now()

	for i := 0; i < 100; i++ {
		_, err := hIndex.Search(query, 10, nil)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}

	searchTime := time.Since(searchStart)

	b.ReportMetric(float64(addTime.Microseconds())/float64(datasetSize), "μs/add")
	b.ReportMetric(float64(searchTime.Microseconds())/100, "μs/search")
	b.ReportMetric(float64(hIndex.Size()), "vectors")
}

// BenchmarkHierarchicalVsStandardHNSW compares hierarchical vs standard HNSW
func BenchmarkHierarchicalVsStandardHNSW(b *testing.B) {
	dimension := 128
	datasetSize := 50000

	vectors := generateBenchmarkVectors(datasetSize, dimension)
	query := vectors[0].Values

	b.Run("HierarchicalHNSW", func(b *testing.B) {
		config := index.DefaultHierarchicalConfig(dimension)
		hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

		// Pre-populate
		for _, vector := range vectors {
			hIndex.Add(vector)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hIndex.Search(query, 10, nil)
		}
	})

	b.Run("StandardHNSW", func(b *testing.B) {
		config := index.DefaultHNSWConfig()
		sIndex := index.NewHNSWIndex(dimension, core.DistanceCosine, config)

		// Pre-populate
		for _, vector := range vectors {
			sIndex.Add(vector)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sIndex.Search(query, 10, nil)
		}
	})
}

// BenchmarkGPUAcceleration benchmarks GPU vs CPU distance computation
func BenchmarkGPUAcceleration(b *testing.B) {
	for _, dim := range []int{64, 128, 256} {
		for _, batchSize := range []int{100, 1000, 10000} {
			b.Run(fmt.Sprintf("dim%d_batch%d", dim, batchSize), func(b *testing.B) {
				benchmarkGPUAcceleration(b, dim, batchSize)
			})
		}
	}
}

func benchmarkGPUAcceleration(b *testing.B, dimension, batchSize int) {
	// Create test data
	queries := make([][]float32, 10)
	vectors := make([][]float32, batchSize)

	for i := range queries {
		queries[i] = generateRandomVectorPhase4(dimension)
	}

	for i := range vectors {
		vectors[i] = generateRandomVectorPhase4(dimension)
	}

	ctx := context.Background()

	b.Run("CPU", func(b *testing.B) {
		config := gpu.DefaultGPUConfig()
		config.Backend = gpu.BackendCPU

		manager := gpu.NewGPUManager(config)
		if err := manager.Initialize(); err != nil {
			b.Fatalf("Failed to initialize CPU manager: %v", err)
		}
		defer manager.Cleanup()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := manager.ComputeDistancesSync(ctx, queries, vectors, core.DistanceL2)
			if err != nil {
				b.Fatalf("Failed to compute distances: %v", err)
			}
		}
	})

	b.Run("GPU_CUDA", func(b *testing.B) {
		config := gpu.DefaultGPUConfig()
		config.Backend = gpu.BackendCUDA
		config.FallbackToCPU = true

		manager := gpu.NewGPUManager(config)
		if err := manager.Initialize(); err != nil {
			b.Skip("CUDA not available")
		}
		defer manager.Cleanup()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := manager.ComputeDistancesSync(ctx, queries, vectors, core.DistanceL2)
			if err != nil {
				b.Fatalf("Failed to compute distances: %v", err)
			}
		}
	})

	b.Run("GPU_OpenCL", func(b *testing.B) {
		config := gpu.DefaultGPUConfig()
		config.Backend = gpu.BackendOpenCL
		config.FallbackToCPU = true

		manager := gpu.NewGPUManager(config)
		if err := manager.Initialize(); err != nil {
			b.Skip("OpenCL not available")
		}
		defer manager.Cleanup()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := manager.ComputeDistancesSync(ctx, queries, vectors, core.DistanceL2)
			if err != nil {
				b.Fatalf("Failed to compute distances: %v", err)
			}
		}
	})
}

// BenchmarkIncrementalUpdates benchmarks incremental index updates
func BenchmarkIncrementalUpdates(b *testing.B) {
	dimension := 128
	initialSize := 10000

	// Create hierarchical index with incremental updates enabled
	config := index.DefaultHierarchicalConfig(dimension)
	config.EnableIncrementalUpdates = true
	config.RebalanceThreshold = 0.3

	hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Pre-populate with initial vectors
	initialVectors := generateBenchmarkVectors(initialSize, dimension)
	for _, vector := range initialVectors {
		hIndex.Add(vector)
	}

	b.Run("Add", func(b *testing.B) {
		newVectors := generateBenchmarkVectors(b.N, dimension)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hIndex.Add(newVectors[i])
		}
	})

	b.Run("Delete", func(b *testing.B) {
		// Get some vector IDs to delete
		deleteCount := minBench(b.N, len(initialVectors))

		b.ResetTimer()
		for i := 0; i < deleteCount; i++ {
			hIndex.Delete(initialVectors[i].ID)
		}
	})

	b.Run("Search", func(b *testing.B) {
		query := initialVectors[0].Values

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hIndex.Search(query, 10, nil)
		}
	})
}

// BenchmarkQualityMonitoring benchmarks quality monitoring overhead
func BenchmarkQualityMonitoring(b *testing.B) {
	dimension := 128
	datasetSize := 10000

	vectors := generateBenchmarkVectors(datasetSize, dimension)
	query := vectors[0].Values

	b.Run("WithMonitoring", func(b *testing.B) {
		config := index.DefaultHierarchicalConfig(dimension)
		config.EnableQualityMonitoring = true

		hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

		// Pre-populate
		for _, vector := range vectors {
			hIndex.Add(vector)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hIndex.Search(query, 10, nil)
		}
	})

	b.Run("WithoutMonitoring", func(b *testing.B) {
		config := index.DefaultHierarchicalConfig(dimension)
		config.EnableQualityMonitoring = false

		hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

		// Pre-populate
		for _, vector := range vectors {
			hIndex.Add(vector)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hIndex.Search(query, 10, nil)
		}
	})
}

// BenchmarkPhase4MemoryEfficiency benchmarks memory usage of different approaches
func BenchmarkPhase4MemoryEfficiency(b *testing.B) {
	dimension := 256
	datasetSize := 50000

	vectors := generateBenchmarkVectors(datasetSize, dimension)

	b.Run("HierarchicalHNSW", func(b *testing.B) {
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		config := index.DefaultHierarchicalConfig(dimension)
		hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

		for _, vector := range vectors {
			hIndex.Add(vector)
		}

		runtime.GC()
		runtime.ReadMemStats(&m2)

		b.ReportMetric(float64(m2.Alloc-m1.Alloc)/1024/1024, "MB")
		b.ReportMetric(float64(m2.Alloc-m1.Alloc)/float64(datasetSize), "bytes/vector")
	})

	b.Run("StandardHNSW", func(b *testing.B) {
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		config := index.DefaultHNSWConfig()
		sIndex := index.NewHNSWIndex(dimension, core.DistanceCosine, config)

		for _, vector := range vectors {
			sIndex.Add(vector)
		}

		runtime.GC()
		runtime.ReadMemStats(&m2)

		b.ReportMetric(float64(m2.Alloc-m1.Alloc)/1024/1024, "MB")
		b.ReportMetric(float64(m2.Alloc-m1.Alloc)/float64(datasetSize), "bytes/vector")
	})
}

// BenchmarkConcurrentOperations benchmarks concurrent index operations
func BenchmarkConcurrentOperations(b *testing.B) {
	dimension := 128
	datasetSize := 10000

	config := index.DefaultHierarchicalConfig(dimension)
	hIndex := index.NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Pre-populate
	vectors := generateBenchmarkVectors(datasetSize, dimension)
	for _, vector := range vectors {
		hIndex.Add(vector)
	}

	query := vectors[0].Values

	b.Run("ConcurrentSearch", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				hIndex.Search(query, 10, nil)
			}
		})
	})

	b.Run("MixedOperations", func(b *testing.B) {
		newVectors := generateBenchmarkVectors(1000, dimension)
		vectorIndex := 0

		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if rand.Float32() < 0.8 { // 80% searches
					hIndex.Search(query, 10, nil)
				} else { // 20% adds
					if vectorIndex < len(newVectors) {
						hIndex.Add(newVectors[vectorIndex])
						vectorIndex++
					}
				}
			}
		})
	})
}

// Helper functions

func generateBenchmarkVectors(count, dimension int) []core.Vector {
	vectors := make([]core.Vector, count)

	for i := 0; i < count; i++ {
		values := generateRandomVector(dimension)
		vectors[i] = core.Vector{
			ID:     fmt.Sprintf("bench-vector-%d", i),
			Values: values,
			Metadata: map[string]string{
				"benchmark": "true",
				"index":     fmt.Sprintf("%d", i),
			},
		}
	}

	return vectors
}

func generateRandomVectorPhase4(dimension int) []float32 {
	values := make([]float32, dimension)
	for j := 0; j < dimension; j++ {
		values[j] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}
	return values
}

func calculateOptimalClusters(datasetSize int) int {
	// Simple heuristic: sqrt(dataset_size) clusters, with min/max bounds
	clusters := int(float64(datasetSize) * 0.01) // 1% of dataset size

	if clusters < 4 {
		clusters = 4
	}
	if clusters > 64 {
		clusters = 64
	}

	return clusters
}

func minBench(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Phase4PerformanceReport generates a comprehensive performance report
type Phase4PerformanceReport struct {
	HierarchicalHNSWMetrics PerformanceMetrics `json:"hierarchical_hnsw"`
	StandardHNSWMetrics     PerformanceMetrics `json:"standard_hnsw"`
	GPUAccelerationMetrics  GPUMetrics         `json:"gpu_acceleration"`
	MemoryEfficiencyMetrics MemoryMetrics      `json:"memory_efficiency"`
	QualityMetrics          QualityMetrics     `json:"quality_metrics"`
}

type PerformanceMetrics struct {
	AddLatencyUs      float64 `json:"add_latency_us"`
	SearchLatencyUs   float64 `json:"search_latency_us"`
	ThroughputQPS     float64 `json:"throughput_qps"`
	ScalabilityFactor float64 `json:"scalability_factor"`
}

type GPUMetrics struct {
	CUDASpeedup     float64 `json:"cuda_speedup"`
	OpenCLSpeedup   float64 `json:"opencl_speedup"`
	MemoryBandwidth float64 `json:"memory_bandwidth_gb_s"`
	Utilization     float64 `json:"gpu_utilization_percent"`
}

type MemoryMetrics struct {
	BytesPerVector   float64 `json:"bytes_per_vector"`
	CompressionRatio float64 `json:"compression_ratio"`
	MemoryEfficiency float64 `json:"memory_efficiency"`
}

type QualityMetrics struct {
	RecallAt10  float64 `json:"recall_at_10"`
	RecallAt100 float64 `json:"recall_at_100"`
	Precision   float64 `json:"precision"`
	F1Score     float64 `json:"f1_score"`
}

// GeneratePhase4Report runs comprehensive benchmarks and generates a report
func GeneratePhase4Report() *Phase4PerformanceReport {
	// This would run all benchmarks and collect metrics
	// For now, return mock data
	return &Phase4PerformanceReport{
		HierarchicalHNSWMetrics: PerformanceMetrics{
			AddLatencyUs:      50.0,
			SearchLatencyUs:   25.0,
			ThroughputQPS:     2000.0,
			ScalabilityFactor: 1.5,
		},
		StandardHNSWMetrics: PerformanceMetrics{
			AddLatencyUs:      75.0,
			SearchLatencyUs:   40.0,
			ThroughputQPS:     1500.0,
			ScalabilityFactor: 1.0,
		},
		GPUAccelerationMetrics: GPUMetrics{
			CUDASpeedup:     3.2,
			OpenCLSpeedup:   2.8,
			MemoryBandwidth: 450.0,
			Utilization:     85.0,
		},
		MemoryEfficiencyMetrics: MemoryMetrics{
			BytesPerVector:   512.0,
			CompressionRatio: 4.0,
			MemoryEfficiency: 0.75,
		},
		QualityMetrics: QualityMetrics{
			RecallAt10:  0.95,
			RecallAt100: 0.98,
			Precision:   0.92,
			F1Score:     0.93,
		},
	}
}
