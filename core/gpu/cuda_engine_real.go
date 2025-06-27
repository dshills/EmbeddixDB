//go:build cuda
// +build cuda

package gpu

import (
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/dshills/EmbeddixDB/core"
)

// RealCUDAEngine implements GPU acceleration using actual CUDA
type RealCUDAEngine struct {
	config        GPUConfig
	kernelManager *CUDAKernelManager
	memoryManager *CUDAMemoryManager
	streamManager *CUDAStreamManager
	stats         PerformanceStats
	initialized   bool
	mu            sync.RWMutex
}

// CUDAMemoryManager manages GPU memory allocations
type CUDAMemoryManager struct {
	allocations    map[string]*CUDAAllocation
	totalAllocated int64
	maxMemory      int64
	mu             sync.Mutex
}

// CUDAAllocation represents a GPU memory allocation
type CUDAAllocation struct {
	ptr      unsafe.Pointer
	size     int64
	inUse    bool
	lastUsed time.Time
}

// CUDAStreamManager manages CUDA streams for concurrent operations
type CUDAStreamManager struct {
	streams    []*CUDAStreamHandle
	nextStream int
	mu         sync.Mutex
}

// CUDAStreamHandle represents a CUDA stream
type CUDAStreamHandle struct {
	id       int
	inUse    bool
	lastUsed time.Time
}

// NewRealCUDAEngine creates a new CUDA engine with actual GPU support
func NewRealCUDAEngine(config GPUConfig) (*RealCUDAEngine, error) {
	kernelManager, err := NewCUDAKernelManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create kernel manager: %w", err)
	}

	deviceInfo, err := kernelManager.GetDeviceInfo()
	if err != nil {
		kernelManager.Cleanup()
		return nil, fmt.Errorf("failed to get device info: %w", err)
	}

	engine := &RealCUDAEngine{
		config:        config,
		kernelManager: kernelManager,
		memoryManager: &CUDAMemoryManager{
			allocations: make(map[string]*CUDAAllocation),
			maxMemory:   deviceInfo.TotalMemory,
		},
		streamManager: &CUDAStreamManager{
			streams: make([]*CUDAStreamHandle, config.NumStreams),
		},
		stats: PerformanceStats{
			DeviceName: deviceInfo.Name,
		},
	}

	// Initialize streams
	for i := 0; i < config.NumStreams; i++ {
		engine.streamManager.streams[i] = &CUDAStreamHandle{
			id: i,
		}
	}

	engine.initialized = true
	return engine, nil
}

// Initialize initializes the CUDA engine
func (e *RealCUDAEngine) Initialize() error {
	if e.initialized {
		return nil
	}
	return fmt.Errorf("engine already created initialized")
}

// ComputeDistances computes distances between query and vectors on GPU
func (e *RealCUDAEngine) ComputeDistances(
	ctx context.Context,
	query []float32,
	vectors []float32,
	distanceType core.DistanceMetric,
) ([]float32, error) {
	if !e.initialized {
		return nil, ErrGPUNotInitialized
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	startTime := time.Now()
	dimension := len(query)
	numVectors := len(vectors) / dimension

	if numVectors == 0 {
		return []float32{}, nil
	}

	// Allocate GPU memory
	queryMem, err := e.allocateAndCopy("query", query)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate query memory: %w", err)
	}
	defer e.freeMemory("query")

	vectorsMem, err := e.allocateAndCopy("vectors", vectors)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate vectors memory: %w", err)
	}
	defer e.freeMemory("vectors")

	resultSize := numVectors * 4 // float32 = 4 bytes
	resultsMem, err := e.allocateMemory("results", int64(resultSize))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate results memory: %w", err)
	}
	defer e.freeMemory("results")

	// Launch appropriate kernel
	switch distanceType {
	case core.CosineDistance:
		err = e.kernelManager.LaunchCosineDistanceKernel(
			queryMem, vectorsMem, resultsMem, dimension, numVectors)
	case core.L2Distance:
		err = e.kernelManager.LaunchL2DistanceKernel(
			queryMem, vectorsMem, resultsMem, dimension, numVectors)
	case core.DotProduct:
		err = e.kernelManager.LaunchDotProductKernel(
			queryMem, vectorsMem, resultsMem, dimension, numVectors)
	default:
		return nil, fmt.Errorf("unsupported distance type: %v", distanceType)
	}

	if err != nil {
		return nil, fmt.Errorf("kernel launch failed: %w", err)
	}

	// Copy results back
	results := make([]float32, numVectors)
	if err := e.kernelManager.CopyFromDevice(results, resultsMem); err != nil {
		return nil, fmt.Errorf("failed to copy results: %w", err)
	}

	// Update statistics
	e.stats.OperationsCompleted++
	e.stats.TotalComputeTime += time.Since(startTime)
	e.stats.VectorsProcessed += int64(numVectors)

	return results, nil
}

// NormalizeVectors normalizes vectors in-place on GPU
func (e *RealCUDAEngine) NormalizeVectors(ctx context.Context, vectors []float32, dimension int) error {
	if !e.initialized {
		return ErrGPUNotInitialized
	}

	// TODO: Implement normalization kernel
	return fmt.Errorf("normalization kernel not yet implemented")
}

// DotProduct computes dot products between queries and vectors on GPU
func (e *RealCUDAEngine) DotProduct(
	ctx context.Context,
	queries []float32,
	vectors []float32,
	dimension int,
) ([]float32, error) {
	if !e.initialized {
		return nil, ErrGPUNotInitialized
	}

	numQueries := len(queries) / dimension
	results := make([]float32, 0, numQueries*len(vectors)/dimension)

	// Process each query
	for i := 0; i < numQueries; i++ {
		query := queries[i*dimension : (i+1)*dimension]
		distances, err := e.ComputeDistances(ctx, query, vectors, core.DotProduct)
		if err != nil {
			return nil, err
		}
		results = append(results, distances...)
	}

	return results, nil
}

// allocateAndCopy allocates GPU memory and copies data
func (e *RealCUDAEngine) allocateAndCopy(name string, data []float32) (unsafe.Pointer, error) {
	size := int64(len(data) * 4) // float32 = 4 bytes
	ptr, err := e.allocateMemory(name, size)
	if err != nil {
		return nil, err
	}

	if err := e.kernelManager.CopyToDevice(ptr, data); err != nil {
		e.freeMemory(name)
		return nil, err
	}

	return ptr, nil
}

// allocateMemory allocates GPU memory
func (e *RealCUDAEngine) allocateMemory(name string, size int64) (unsafe.Pointer, error) {
	e.memoryManager.mu.Lock()
	defer e.memoryManager.mu.Unlock()

	// Check if allocation already exists
	if alloc, exists := e.memoryManager.allocations[name]; exists {
		if alloc.size >= size {
			alloc.inUse = true
			alloc.lastUsed = time.Now()
			return alloc.ptr, nil
		}
		// Free old allocation
		e.kernelManager.FreeMemory(alloc.ptr)
		delete(e.memoryManager.allocations, name)
		e.memoryManager.totalAllocated -= alloc.size
	}

	// Check memory limit
	if e.memoryManager.totalAllocated+size > e.memoryManager.maxMemory {
		// Try to free unused allocations
		e.freeUnusedMemory()

		if e.memoryManager.totalAllocated+size > e.memoryManager.maxMemory {
			return nil, fmt.Errorf("out of GPU memory")
		}
	}

	// Allocate new memory
	ptr, err := e.kernelManager.AllocateMemory(int(size))
	if err != nil {
		return nil, err
	}

	e.memoryManager.allocations[name] = &CUDAAllocation{
		ptr:      ptr,
		size:     size,
		inUse:    true,
		lastUsed: time.Now(),
	}
	e.memoryManager.totalAllocated += size

	return ptr, nil
}

// freeMemory marks memory as not in use
func (e *RealCUDAEngine) freeMemory(name string) {
	e.memoryManager.mu.Lock()
	defer e.memoryManager.mu.Unlock()

	if alloc, exists := e.memoryManager.allocations[name]; exists {
		alloc.inUse = false
		alloc.lastUsed = time.Now()
	}
}

// freeUnusedMemory frees memory allocations not in use
func (e *RealCUDAEngine) freeUnusedMemory() {
	cutoff := time.Now().Add(-time.Minute)

	for name, alloc := range e.memoryManager.allocations {
		if !alloc.inUse && alloc.lastUsed.Before(cutoff) {
			e.kernelManager.FreeMemory(alloc.ptr)
			delete(e.memoryManager.allocations, name)
			e.memoryManager.totalAllocated -= alloc.size
		}
	}
}

// GetMemoryInfo returns current GPU memory information
func (e *RealCUDAEngine) GetMemoryInfo() (GPUMemory, error) {
	if !e.initialized {
		return nil, ErrGPUNotInitialized
	}

	e.memoryManager.mu.Lock()
	defer e.memoryManager.mu.Unlock()

	return &CUDAMemoryInfo{
		total:     e.memoryManager.maxMemory,
		allocated: e.memoryManager.totalAllocated,
		free:      e.memoryManager.maxMemory - e.memoryManager.totalAllocated,
	}, nil
}

// CUDAMemoryInfo implements GPUMemory interface
type CUDAMemoryInfo struct {
	total     int64
	allocated int64
	free      int64
}

func (m *CUDAMemoryInfo) TotalMemory() int64     { return m.total }
func (m *CUDAMemoryInfo) AllocatedMemory() int64 { return m.allocated }
func (m *CUDAMemoryInfo) FreeMemory() int64      { return m.free }
func (m *CUDAMemoryInfo) Allocate(size int) error {
	return fmt.Errorf("direct allocation not supported")
}
func (m *CUDAMemoryInfo) Free() error {
	return fmt.Errorf("direct free not supported")
}
func (m *CUDAMemoryInfo) CopyToDevice(data []float32) error {
	return fmt.Errorf("direct copy not supported")
}
func (m *CUDAMemoryInfo) CopyFromDevice(data []float32) error {
	return fmt.Errorf("direct copy not supported")
}

// GetStats returns performance statistics
func (e *RealCUDAEngine) GetStats() PerformanceStats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	stats := e.stats
	if stats.OperationsCompleted > 0 {
		stats.AverageComputeTime = stats.TotalComputeTime / time.Duration(stats.OperationsCompleted)
		stats.Throughput = float64(stats.VectorsProcessed) / stats.TotalComputeTime.Seconds()
	}

	// Add memory stats
	e.memoryManager.mu.Lock()
	stats.MemoryUsed = e.memoryManager.totalAllocated
	stats.MemoryTotal = e.memoryManager.maxMemory
	e.memoryManager.mu.Unlock()

	return stats
}

// Cleanup releases GPU resources
func (e *RealCUDAEngine) Cleanup() error {
	if !e.initialized {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Free all memory allocations
	e.memoryManager.mu.Lock()
	for name, alloc := range e.memoryManager.allocations {
		e.kernelManager.FreeMemory(alloc.ptr)
		delete(e.memoryManager.allocations, name)
	}
	e.memoryManager.totalAllocated = 0
	e.memoryManager.mu.Unlock()

	// Cleanup kernel manager
	if err := e.kernelManager.Cleanup(); err != nil {
		return err
	}

	e.initialized = false
	return nil
}

// IsAvailable checks if CUDA is available
func (e *RealCUDAEngine) IsAvailable() bool {
	return e.initialized
}

// GetDeviceInfo returns information about the GPU device
func (e *RealCUDAEngine) GetDeviceInfo() (DeviceInfo, error) {
	if !e.initialized {
		return DeviceInfo{}, ErrGPUNotInitialized
	}

	return e.kernelManager.GetDeviceInfo()
}
