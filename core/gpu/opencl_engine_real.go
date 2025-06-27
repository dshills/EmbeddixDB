// +build opencl

package gpu

import (
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/dshills/EmbeddixDB/core"
)

// RealOpenCLEngine implements GPU acceleration using actual OpenCL
type RealOpenCLEngine struct {
	config         GPUConfig
	kernelManager  *OpenCLKernelManager
	memoryManager  *OpenCLMemoryManager
	stats          PerformanceStats
	initialized    bool
	mu             sync.RWMutex
}

// OpenCLMemoryManager manages GPU memory allocations
type OpenCLMemoryManager struct {
	allocations    map[string]*OpenCLAllocation
	totalAllocated int64
	maxMemory      int64
	mu             sync.Mutex
}

// OpenCLAllocation represents a GPU memory allocation
type OpenCLAllocation struct {
	buffer    unsafe.Pointer
	size      int64
	readOnly  bool
	inUse     bool
	lastUsed  time.Time
}

// NewRealOpenCLEngine creates a new OpenCL engine with actual GPU support
func NewRealOpenCLEngine(config GPUConfig) (*RealOpenCLEngine, error) {
	kernelManager, err := NewOpenCLKernelManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create kernel manager: %w", err)
	}

	deviceInfo, err := kernelManager.GetDeviceInfo()
	if err != nil {
		kernelManager.Cleanup()
		return nil, fmt.Errorf("failed to get device info: %w", err)
	}

	engine := &RealOpenCLEngine{
		config:        config,
		kernelManager: kernelManager,
		memoryManager: &OpenCLMemoryManager{
			allocations: make(map[string]*OpenCLAllocation),
			maxMemory:   deviceInfo.TotalMemory,
		},
		stats: PerformanceStats{
			DeviceName: deviceInfo.Name,
		},
	}

	engine.initialized = true
	return engine, nil
}

// Initialize initializes the OpenCL engine
func (e *RealOpenCLEngine) Initialize() error {
	if e.initialized {
		return nil
	}
	return fmt.Errorf("engine already created initialized")
}

// ComputeDistances computes distances between query and vectors on GPU
func (e *RealOpenCLEngine) ComputeDistances(
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
	queryBuffer, err := e.allocateAndWrite("query", query, true)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate query buffer: %w", err)
	}
	defer e.freeBuffer("query")

	vectorsBuffer, err := e.allocateAndWrite("vectors", vectors, true)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate vectors buffer: %w", err)
	}
	defer e.freeBuffer("vectors")

	resultSize := numVectors * 4 // float32 = 4 bytes
	resultsBuffer, err := e.allocateBuffer("results", int64(resultSize), false)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate results buffer: %w", err)
	}
	defer e.freeBuffer("results")

	// Launch appropriate kernel
	switch distanceType {
	case core.CosineDistance:
		err = e.kernelManager.LaunchCosineDistanceKernel(
			queryBuffer, vectorsBuffer, resultsBuffer, dimension, numVectors)
	case core.L2Distance:
		err = e.kernelManager.LaunchL2DistanceKernel(
			queryBuffer, vectorsBuffer, resultsBuffer, dimension, numVectors)
	case core.DotProduct:
		err = e.kernelManager.LaunchDotProductKernel(
			queryBuffer, vectorsBuffer, resultsBuffer, dimension, numVectors)
	default:
		return nil, fmt.Errorf("unsupported distance type: %v", distanceType)
	}

	if err != nil {
		return nil, fmt.Errorf("kernel launch failed: %w", err)
	}

	// Read results back
	results := make([]float32, numVectors)
	if err := e.kernelManager.ReadBuffer(resultsBuffer, results); err != nil {
		return nil, fmt.Errorf("failed to read results: %w", err)
	}

	// Update statistics
	e.stats.OperationsCompleted++
	e.stats.TotalComputeTime += time.Since(startTime)
	e.stats.VectorsProcessed += int64(numVectors)

	return results, nil
}

// NormalizeVectors normalizes vectors in-place on GPU
func (e *RealOpenCLEngine) NormalizeVectors(ctx context.Context, vectors []float32, dimension int) error {
	if !e.initialized {
		return ErrGPUNotInitialized
	}

	// TODO: Implement normalization kernel
	return fmt.Errorf("normalization kernel not yet implemented")
}

// DotProduct computes dot products between queries and vectors on GPU
func (e *RealOpenCLEngine) DotProduct(
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

// allocateAndWrite allocates GPU buffer and writes data
func (e *RealOpenCLEngine) allocateAndWrite(name string, data []float32, readOnly bool) (unsafe.Pointer, error) {
	size := int64(len(data) * 4) // float32 = 4 bytes
	buffer, err := e.allocateBuffer(name, size, readOnly)
	if err != nil {
		return nil, err
	}

	if err := e.kernelManager.WriteBuffer(buffer, data); err != nil {
		e.freeBuffer(name)
		return nil, err
	}

	return buffer, nil
}

// allocateBuffer allocates GPU buffer
func (e *RealOpenCLEngine) allocateBuffer(name string, size int64, readOnly bool) (unsafe.Pointer, error) {
	e.memoryManager.mu.Lock()
	defer e.memoryManager.mu.Unlock()

	// Check if allocation already exists
	if alloc, exists := e.memoryManager.allocations[name]; exists {
		if alloc.size >= size && alloc.readOnly == readOnly {
			alloc.inUse = true
			alloc.lastUsed = time.Now()
			return alloc.buffer, nil
		}
		// Free old allocation
		e.kernelManager.FreeBuffer(alloc.buffer)
		delete(e.memoryManager.allocations, name)
		e.memoryManager.totalAllocated -= alloc.size
	}

	// Check memory limit
	if e.memoryManager.totalAllocated+size > e.memoryManager.maxMemory {
		// Try to free unused allocations
		e.freeUnusedBuffers()
		
		if e.memoryManager.totalAllocated+size > e.memoryManager.maxMemory {
			return nil, fmt.Errorf("out of GPU memory")
		}
	}

	// Allocate new buffer
	buffer, err := e.kernelManager.AllocateBuffer(int(size), readOnly)
	if err != nil {
		return nil, err
	}

	e.memoryManager.allocations[name] = &OpenCLAllocation{
		buffer:   buffer,
		size:     size,
		readOnly: readOnly,
		inUse:    true,
		lastUsed: time.Now(),
	}
	e.memoryManager.totalAllocated += size

	return buffer, nil
}

// freeBuffer marks buffer as not in use
func (e *RealOpenCLEngine) freeBuffer(name string) {
	e.memoryManager.mu.Lock()
	defer e.memoryManager.mu.Unlock()

	if alloc, exists := e.memoryManager.allocations[name]; exists {
		alloc.inUse = false
		alloc.lastUsed = time.Now()
	}
}

// freeUnusedBuffers frees buffer allocations not in use
func (e *RealOpenCLEngine) freeUnusedBuffers() {
	cutoff := time.Now().Add(-time.Minute)
	
	for name, alloc := range e.memoryManager.allocations {
		if !alloc.inUse && alloc.lastUsed.Before(cutoff) {
			e.kernelManager.FreeBuffer(alloc.buffer)
			delete(e.memoryManager.allocations, name)
			e.memoryManager.totalAllocated -= alloc.size
		}
	}
}

// GetMemoryInfo returns current GPU memory information
func (e *RealOpenCLEngine) GetMemoryInfo() (GPUMemory, error) {
	if !e.initialized {
		return nil, ErrGPUNotInitialized
	}

	e.memoryManager.mu.Lock()
	defer e.memoryManager.mu.Unlock()

	return &OpenCLMemoryInfo{
		total:     e.memoryManager.maxMemory,
		allocated: e.memoryManager.totalAllocated,
		free:      e.memoryManager.maxMemory - e.memoryManager.totalAllocated,
	}, nil
}

// OpenCLMemoryInfo implements GPUMemory interface
type OpenCLMemoryInfo struct {
	total     int64
	allocated int64
	free      int64
}

func (m *OpenCLMemoryInfo) TotalMemory() int64     { return m.total }
func (m *OpenCLMemoryInfo) AllocatedMemory() int64 { return m.allocated }
func (m *OpenCLMemoryInfo) FreeMemory() int64      { return m.free }
func (m *OpenCLMemoryInfo) Allocate(size int) error {
	return fmt.Errorf("direct allocation not supported")
}
func (m *OpenCLMemoryInfo) Free() error {
	return fmt.Errorf("direct free not supported")
}
func (m *OpenCLMemoryInfo) CopyToDevice(data []float32) error {
	return fmt.Errorf("direct copy not supported")
}
func (m *OpenCLMemoryInfo) CopyFromDevice(data []float32) error {
	return fmt.Errorf("direct copy not supported")
}

// GetStats returns performance statistics
func (e *RealOpenCLEngine) GetStats() PerformanceStats {
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
func (e *RealOpenCLEngine) Cleanup() error {
	if !e.initialized {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Free all buffer allocations
	e.memoryManager.mu.Lock()
	for name, alloc := range e.memoryManager.allocations {
		e.kernelManager.FreeBuffer(alloc.buffer)
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

// IsAvailable checks if OpenCL is available
func (e *RealOpenCLEngine) IsAvailable() bool {
	return e.initialized
}

// GetDeviceInfo returns information about the GPU device
func (e *RealOpenCLEngine) GetDeviceInfo() (DeviceInfo, error) {
	if !e.initialized {
		return DeviceInfo{}, ErrGPUNotInitialized
	}
	
	return e.kernelManager.GetDeviceInfo()
}