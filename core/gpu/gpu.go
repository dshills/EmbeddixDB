package gpu

import (
	"context"
	"fmt"
	"sync"

	"github.com/dshills/EmbeddixDB/core"
)

// GPUBackend represents the type of GPU acceleration backend
type GPUBackend string

const (
	BackendCUDA   GPUBackend = "cuda"
	BackendOpenCL GPUBackend = "opencl"
	BackendCPU    GPUBackend = "cpu" // Fallback
)

// GPUConfig configures GPU acceleration settings
type GPUConfig struct {
	Backend          GPUBackend `json:"backend"`            // GPU backend to use
	DeviceID         int        `json:"device_id"`          // GPU device ID
	EnableAutoDetect bool       `json:"enable_auto_detect"` // Auto-detect best backend
	BatchSize        int        `json:"batch_size"`         // Batch size for GPU operations
	MemoryLimitMB    int        `json:"memory_limit_mb"`    // GPU memory limit in MB
	EnableProfiling  bool       `json:"enable_profiling"`   // Enable performance profiling
	FallbackToCPU    bool       `json:"fallback_to_cpu"`    // Fallback to CPU if GPU fails
}

// DefaultGPUConfig returns sensible defaults for GPU acceleration
func DefaultGPUConfig() GPUConfig {
	return GPUConfig{
		Backend:          BackendCUDA,
		DeviceID:         0,
		EnableAutoDetect: true,
		BatchSize:        1024,
		MemoryLimitMB:    2048, // 2GB default
		EnableProfiling:  false,
		FallbackToCPU:    true,
	}
}

// GPUEngine interface for GPU-accelerated operations
type GPUEngine interface {
	// Initialize the GPU engine
	Initialize(config GPUConfig) error

	// Cleanup resources
	Cleanup() error

	// Distance computation operations
	ComputeDistances(ctx context.Context, queries [][]float32, vectors [][]float32, metric core.DistanceMetric) ([][]float32, error)
	BatchComputeDistances(ctx context.Context, batch DistanceBatch) (*DistanceResult, error)

	// Vector operations
	NormalizeVectors(ctx context.Context, vectors [][]float32) ([][]float32, error)
	DotProduct(ctx context.Context, a, b [][]float32) ([]float32, error)

	// Memory management
	AllocateMemory(sizeBytes int) (GPUMemory, error)
	CopyToGPU(data interface{}) (GPUMemory, error)
	CopyFromGPU(mem GPUMemory) (interface{}, error)

	// Information
	GetDeviceInfo() DeviceInfo
	GetMemoryInfo() MemoryInfo

	// Performance monitoring
	GetPerformanceStats() PerformanceStats
}

// DistanceBatch represents a batch of distance computations
type DistanceBatch struct {
	Queries  [][]float32         `json:"queries"`
	Vectors  [][]float32         `json:"vectors"`
	Metric   core.DistanceMetric `json:"metric"`
	BatchID  string              `json:"batch_id"`
	Priority int                 `json:"priority"`
}

// DistanceResult contains the results of batch distance computation
type DistanceResult struct {
	Distances   [][]float32 `json:"distances"`
	BatchID     string      `json:"batch_id"`
	ComputeTime float64     `json:"compute_time_ms"`
	GPUTime     float64     `json:"gpu_time_ms"`
	MemoryUsed  int64       `json:"memory_used_bytes"`
}

// GPUMemory represents allocated GPU memory
type GPUMemory interface {
	Size() int64
	Free() error
	IsValid() bool
}

// DeviceInfo contains GPU device information
type DeviceInfo struct {
	Name            string `json:"name"`
	Backend         string `json:"backend"`
	DeviceID        int    `json:"device_id"`
	ComputeUnits    int    `json:"compute_units"`
	MaxWorkGroup    int    `json:"max_work_group"`
	GlobalMemory    int64  `json:"global_memory_mb"`
	LocalMemory     int64  `json:"local_memory_kb"`
	SupportsFloat64 bool   `json:"supports_float64"`
}

// MemoryInfo contains current GPU memory usage
type MemoryInfo struct {
	TotalMemory      int64 `json:"total_memory_mb"`
	UsedMemory       int64 `json:"used_memory_mb"`
	FreeMemory       int64 `json:"free_memory_mb"`
	AllocatedBuffers int   `json:"allocated_buffers"`
}

// PerformanceStats contains GPU performance statistics
type PerformanceStats struct {
	TotalOperations   int64   `json:"total_operations"`
	TotalComputeTime  float64 `json:"total_compute_time_ms"`
	AverageLatency    float64 `json:"average_latency_ms"`
	Throughput        float64 `json:"throughput_ops_per_sec"`
	MemoryBandwidth   float64 `json:"memory_bandwidth_gb_per_sec"`
	GPUUtilization    float64 `json:"gpu_utilization_percent"`
	LastOperationTime float64 `json:"last_operation_time_ms"`
}

// GPUManager manages GPU resources and operations
type GPUManager struct {
	config         GPUConfig
	engine         GPUEngine
	initialized    bool
	mu             sync.RWMutex
	operationQueue chan DistanceBatch
	resultQueue    chan *DistanceResult
	stats          PerformanceStats
}

// NewGPUManager creates a new GPU manager
func NewGPUManager(config GPUConfig) *GPUManager {
	return &GPUManager{
		config:         config,
		operationQueue: make(chan DistanceBatch, 100),
		resultQueue:    make(chan *DistanceResult, 100),
	}
}

// Initialize initializes the GPU manager and selects the best backend
func (gm *GPUManager) Initialize() error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	if gm.initialized {
		return nil
	}

	var err error

	// Auto-detect best backend if enabled
	if gm.config.EnableAutoDetect {
		gm.config.Backend = gm.detectBestBackend()
	}

	// Create appropriate engine
	switch gm.config.Backend {
	case BackendCUDA:
		gm.engine, err = NewCUDAEngine()
	case BackendOpenCL:
		gm.engine, err = NewOpenCLEngine()
	case BackendCPU:
		gm.engine, err = NewCPUEngine()
	default:
		return fmt.Errorf("unsupported GPU backend: %s", gm.config.Backend)
	}

	if err != nil {
		if gm.config.FallbackToCPU && gm.config.Backend != BackendCPU {
			// Fallback to CPU
			gm.config.Backend = BackendCPU
			gm.engine, err = NewCPUEngine()
		}
		if err != nil {
			return fmt.Errorf("failed to initialize GPU engine: %w", err)
		}
	}

	// Initialize the engine
	if err := gm.engine.Initialize(gm.config); err != nil {
		return fmt.Errorf("failed to initialize GPU engine: %w", err)
	}

	gm.initialized = true

	// Start background processing
	go gm.processOperations()

	return nil
}

// ComputeDistancesAsync submits a batch for asynchronous distance computation
func (gm *GPUManager) ComputeDistancesAsync(ctx context.Context, batch DistanceBatch) error {
	if !gm.initialized {
		return fmt.Errorf("GPU manager not initialized")
	}

	select {
	case gm.operationQueue <- batch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	default:
		return fmt.Errorf("operation queue full")
	}
}

// GetResult retrieves a completed distance computation result
func (gm *GPUManager) GetResult(ctx context.Context) (*DistanceResult, error) {
	select {
	case result := <-gm.resultQueue:
		return result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ComputeDistancesSync performs synchronous distance computation
func (gm *GPUManager) ComputeDistancesSync(ctx context.Context, queries [][]float32, vectors [][]float32, metric core.DistanceMetric) ([][]float32, error) {
	if !gm.initialized {
		return nil, fmt.Errorf("GPU manager not initialized")
	}

	gm.mu.RLock()
	defer gm.mu.RUnlock()

	return gm.engine.ComputeDistances(ctx, queries, vectors, metric)
}

// processOperations processes queued operations in the background
func (gm *GPUManager) processOperations() {
	for batch := range gm.operationQueue {
		ctx := context.Background() // Could be improved with proper context management

		result, err := gm.engine.BatchComputeDistances(ctx, batch)
		if err != nil {
			// Log error and continue
			continue
		}

		// Update statistics
		gm.updateStats(result)

		// Send result
		select {
		case gm.resultQueue <- result:
		default:
			// Result queue full, drop result
		}
	}
}

// updateStats updates performance statistics
func (gm *GPUManager) updateStats(result *DistanceResult) {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	gm.stats.TotalOperations++
	gm.stats.TotalComputeTime += result.ComputeTime
	gm.stats.LastOperationTime = result.ComputeTime

	if gm.stats.TotalOperations > 0 {
		gm.stats.AverageLatency = gm.stats.TotalComputeTime / float64(gm.stats.TotalOperations)
	}
}

// detectBestBackend detects the best available GPU backend
func (gm *GPUManager) detectBestBackend() GPUBackend {
	// Try CUDA first
	if cudaEngine, err := NewCUDAEngine(); err == nil {
		cudaEngine.Cleanup()
		return BackendCUDA
	}

	// Try OpenCL
	if openclEngine, err := NewOpenCLEngine(); err == nil {
		openclEngine.Cleanup()
		return BackendOpenCL
	}

	// Fallback to CPU
	return BackendCPU
}

// GetDeviceInfo returns information about the current GPU device
func (gm *GPUManager) GetDeviceInfo() (DeviceInfo, error) {
	if !gm.initialized {
		return DeviceInfo{}, fmt.Errorf("GPU manager not initialized")
	}

	gm.mu.RLock()
	defer gm.mu.RUnlock()

	return gm.engine.GetDeviceInfo(), nil
}

// GetMemoryInfo returns current GPU memory usage
func (gm *GPUManager) GetMemoryInfo() (MemoryInfo, error) {
	if !gm.initialized {
		return MemoryInfo{}, fmt.Errorf("GPU manager not initialized")
	}

	gm.mu.RLock()
	defer gm.mu.RUnlock()

	return gm.engine.GetMemoryInfo(), nil
}

// GetPerformanceStats returns current performance statistics
func (gm *GPUManager) GetPerformanceStats() PerformanceStats {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	stats := gm.stats
	if gm.initialized {
		engineStats := gm.engine.GetPerformanceStats()
		stats.GPUUtilization = engineStats.GPUUtilization
		stats.MemoryBandwidth = engineStats.MemoryBandwidth
		stats.Throughput = engineStats.Throughput
	}

	return stats
}

// Cleanup releases GPU resources
func (gm *GPUManager) Cleanup() error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	if !gm.initialized {
		return nil
	}

	// Close queues
	close(gm.operationQueue)
	close(gm.resultQueue)

	// Cleanup engine
	if gm.engine != nil {
		if err := gm.engine.Cleanup(); err != nil {
			return fmt.Errorf("failed to cleanup GPU engine: %w", err)
		}
	}

	gm.initialized = false
	return nil
}
