package gpu

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// CPUEngine implements GPU interface using CPU for fallback
type CPUEngine struct {
	config      GPUConfig
	initialized bool
	workers     int
	stats       PerformanceStats
	mu          sync.RWMutex
}

// CPUMemory implements GPUMemory for CPU memory
type CPUMemory struct {
	data  []byte
	size  int64
	valid bool
}

// NewCPUEngine creates a new CPU engine for fallback
func NewCPUEngine() (GPUEngine, error) {
	return &CPUEngine{
		workers: runtime.NumCPU(),
		stats:   PerformanceStats{},
	}, nil
}

// Initialize initializes the CPU engine
func (ce *CPUEngine) Initialize(config GPUConfig) error {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	if ce.initialized {
		return nil
	}

	ce.config = config
	ce.workers = runtime.NumCPU()

	// Set a reasonable number of workers based on batch size
	if config.BatchSize > 0 {
		ce.workers = minCPU(ce.workers, config.BatchSize/100)
		if ce.workers < 1 {
			ce.workers = 1
		}
	}

	ce.initialized = true
	return nil
}

// ComputeDistances computes distances using CPU with parallelization
func (ce *CPUEngine) ComputeDistances(ctx context.Context, queries [][]float32, vectors [][]float32, metric core.DistanceMetric) ([][]float32, error) {
	if !ce.initialized {
		return nil, fmt.Errorf("CPU engine not initialized")
	}

	startTime := time.Now()

	// Validate inputs
	if len(queries) == 0 || len(vectors) == 0 {
		return [][]float32{}, nil
	}

	// Check dimensions
	queryDim := len(queries[0])
	vectorDim := len(vectors[0])
	if queryDim != vectorDim {
		return nil, fmt.Errorf("dimension mismatch: query=%d, vector=%d", queryDim, vectorDim)
	}

	// Initialize result matrix
	results := make([][]float32, len(queries))
	for i := range results {
		results[i] = make([]float32, len(vectors))
	}

	// Parallel computation using worker goroutines
	var wg sync.WaitGroup
	queryChannel := make(chan int, len(queries))

	// Start workers
	for w := 0; w < ce.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for queryIdx := range queryChannel {
				select {
				case <-ctx.Done():
					return
				default:
					ce.computeQueryDistances(queries[queryIdx], vectors, results[queryIdx], metric)
				}
			}
		}()
	}

	// Send work to workers
	go func() {
		defer close(queryChannel)
		for i := range queries {
			select {
			case queryChannel <- i:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Wait for completion
	wg.Wait()

	// Check for context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Update stats
	computeTime := time.Since(startTime).Seconds() * 1000
	ce.updateStats(1, computeTime)

	return results, nil
}

// BatchComputeDistances performs batch distance computation
func (ce *CPUEngine) BatchComputeDistances(ctx context.Context, batch DistanceBatch) (*DistanceResult, error) {
	startTime := time.Now()

	distances, err := ce.ComputeDistances(ctx, batch.Queries, batch.Vectors, batch.Metric)
	if err != nil {
		return nil, err
	}

	computeTime := time.Since(startTime).Seconds() * 1000

	return &DistanceResult{
		Distances:   distances,
		BatchID:     batch.BatchID,
		ComputeTime: computeTime,
		GPUTime:     0, // No GPU time for CPU engine
		MemoryUsed:  ce.calculateMemoryUsed(batch),
	}, nil
}

// NormalizeVectors normalizes vectors using CPU
func (ce *CPUEngine) NormalizeVectors(ctx context.Context, vectors [][]float32) ([][]float32, error) {
	if !ce.initialized {
		return nil, fmt.Errorf("CPU engine not initialized")
	}

	normalized := make([][]float32, len(vectors))

	var wg sync.WaitGroup
	vectorChannel := make(chan int, len(vectors))

	// Start workers
	for w := 0; w < ce.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for vectorIdx := range vectorChannel {
				select {
				case <-ctx.Done():
					return
				default:
					normalized[vectorIdx] = ce.normalizeVector(vectors[vectorIdx])
				}
			}
		}()
	}

	// Send work to workers
	go func() {
		defer close(vectorChannel)
		for i := range vectors {
			select {
			case vectorChannel <- i:
			case <-ctx.Done():
				return
			}
		}
	}()

	wg.Wait()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	return normalized, nil
}

// DotProduct computes dot products using CPU
func (ce *CPUEngine) DotProduct(ctx context.Context, a, b [][]float32) ([]float32, error) {
	if !ce.initialized {
		return nil, fmt.Errorf("CPU engine not initialized")
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("vector arrays must have same length")
	}

	results := make([]float32, len(a))

	var wg sync.WaitGroup
	pairChannel := make(chan int, len(a))

	// Start workers
	for w := 0; w < ce.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for pairIdx := range pairChannel {
				select {
				case <-ctx.Done():
					return
				default:
					results[pairIdx] = ce.computeDotProduct(a[pairIdx], b[pairIdx])
				}
			}
		}()
	}

	// Send work to workers
	go func() {
		defer close(pairChannel)
		for i := range a {
			select {
			case pairChannel <- i:
			case <-ctx.Done():
				return
			}
		}
	}()

	wg.Wait()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	return results, nil
}

// AllocateMemory allocates CPU memory (mock GPU memory interface)
func (ce *CPUEngine) AllocateMemory(sizeBytes int) (GPUMemory, error) {
	if !ce.initialized {
		return nil, fmt.Errorf("CPU engine not initialized")
	}

	data := make([]byte, sizeBytes)

	return &CPUMemory{
		data:  data,
		size:  int64(sizeBytes),
		valid: true,
	}, nil
}

// CopyToGPU copies data to CPU memory (mock GPU operation)
func (ce *CPUEngine) CopyToGPU(data interface{}) (GPUMemory, error) {
	if !ce.initialized {
		return nil, fmt.Errorf("CPU engine not initialized")
	}

	// For CPU engine, this is essentially a no-op since data is already in CPU memory
	return &CPUMemory{
		data:  nil,  // Would serialize data in real implementation
		size:  1024, // Mock size
		valid: true,
	}, nil
}

// CopyFromGPU copies data from CPU memory (mock GPU operation)
func (ce *CPUEngine) CopyFromGPU(mem GPUMemory) (interface{}, error) {
	if !ce.initialized {
		return nil, fmt.Errorf("CPU engine not initialized")
	}

	// For CPU engine, this is essentially a no-op
	return nil, nil
}

// GetDeviceInfo returns CPU device information
func (ce *CPUEngine) GetDeviceInfo() DeviceInfo {
	return DeviceInfo{
		Name:         fmt.Sprintf("CPU (%d cores)", runtime.NumCPU()),
		Backend:      "cpu",
		DeviceID:     0,
		ComputeUnits: runtime.NumCPU(),
		GlobalMemory: 1024, // Mock value in MB
		LocalMemory:  32,   // Mock value in KB
		MaxWorkGroup: ce.workers,
	}
}

// GetMemoryInfo returns current memory usage
func (ce *CPUEngine) GetMemoryInfo() MemoryInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemoryInfo{
		TotalMemory:      int64(m.Sys / 1024 / 1024),             // Convert to MB
		UsedMemory:       int64(m.Alloc / 1024 / 1024),           // Convert to MB
		FreeMemory:       int64((m.Sys - m.Alloc) / 1024 / 1024), // Convert to MB
		AllocatedBuffers: 0,                                      // Not tracked for CPU
	}
}

// GetPerformanceStats returns performance statistics
func (ce *CPUEngine) GetPerformanceStats() PerformanceStats {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	stats := ce.stats
	// CPU doesn't have GPU utilization, so it's always 0
	stats.GPUUtilization = 0.0

	return stats
}

// Cleanup releases CPU resources (no-op for CPU)
func (ce *CPUEngine) Cleanup() error {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	ce.initialized = false
	return nil
}

// Helper methods

// computeQueryDistances computes distances between a single query and all vectors
func (ce *CPUEngine) computeQueryDistances(query []float32, vectors [][]float32, results []float32, metric core.DistanceMetric) {
	for i, vector := range vectors {
		distance, err := core.CalculateDistance(query, vector, metric)
		if err != nil {
			results[i] = float32(1e9) // Large distance on error
		} else {
			results[i] = distance
		}
	}
}

// normalizeVector normalizes a single vector
func (ce *CPUEngine) normalizeVector(vector []float32) []float32 {
	// Calculate magnitude
	var magnitude float32
	for _, v := range vector {
		magnitude += v * v
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))

	if magnitude == 0 {
		return vector // Avoid division by zero
	}

	// Normalize
	normalized := make([]float32, len(vector))
	for i, v := range vector {
		normalized[i] = v / magnitude
	}

	return normalized
}

// computeDotProduct computes dot product between two vectors
func (ce *CPUEngine) computeDotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var result float32
	for i := range a {
		result += a[i] * b[i]
	}

	return result
}

// calculateMemoryUsed estimates memory usage for a batch
func (ce *CPUEngine) calculateMemoryUsed(batch DistanceBatch) int64 {
	queryFloats := 0
	for _, query := range batch.Queries {
		queryFloats += len(query)
	}

	vectorFloats := 0
	for _, vector := range batch.Vectors {
		vectorFloats += len(vector)
	}

	resultFloats := len(batch.Queries) * len(batch.Vectors)

	return int64((queryFloats + vectorFloats + resultFloats) * 4) // 4 bytes per float32
}

// updateStats updates performance statistics
func (ce *CPUEngine) updateStats(operations int64, computeTimeMs float64) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	ce.stats.TotalOperations += operations
	ce.stats.TotalComputeTime += computeTimeMs
	ce.stats.LastOperationTime = computeTimeMs

	if ce.stats.TotalOperations > 0 {
		ce.stats.AverageLatency = ce.stats.TotalComputeTime / float64(ce.stats.TotalOperations)
		ce.stats.Throughput = float64(ce.stats.TotalOperations) / (ce.stats.TotalComputeTime / 1000.0)
	}
}

// CPUMemory methods

func (cm *CPUMemory) Size() int64 {
	return cm.size
}

func (cm *CPUMemory) Free() error {
	if !cm.valid {
		return fmt.Errorf("memory already freed")
	}

	// For CPU memory, just mark as invalid (GC will handle cleanup)
	cm.data = nil
	cm.valid = false

	return nil
}

func (cm *CPUMemory) IsValid() bool {
	return cm.valid
}

// Utility function
func minCPU(a, b int) int {
	if a < b {
		return a
	}
	return b
}
