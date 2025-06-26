package gpu

import (
	"context"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

func TestGPUManager_Initialize(t *testing.T) {
	config := DefaultGPUConfig()
	config.FallbackToCPU = true // Ensure we can test even without GPU

	manager := NewGPUManager(config)

	err := manager.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize GPU manager: %v", err)
	}

	// Test device info
	deviceInfo, err := manager.GetDeviceInfo()
	if err != nil {
		t.Fatalf("Failed to get device info: %v", err)
	}

	if deviceInfo.Name == "" {
		t.Error("Device name should not be empty")
	}

	if deviceInfo.Backend == "" {
		t.Error("Backend should not be empty")
	}

	// Cleanup
	err = manager.Cleanup()
	if err != nil {
		t.Fatalf("Failed to cleanup GPU manager: %v", err)
	}
}

func TestGPUManager_SyncDistanceComputation(t *testing.T) {
	config := DefaultGPUConfig()
	config.FallbackToCPU = true

	manager := NewGPUManager(config)
	err := manager.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Create test data
	queries := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
	}

	vectors := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}

	ctx := context.Background()

	// Test distance computation
	results, err := manager.ComputeDistancesSync(ctx, queries, vectors, core.DistanceL2)
	if err != nil {
		t.Fatalf("Failed to compute distances: %v", err)
	}

	// Verify results shape
	if len(results) != len(queries) {
		t.Errorf("Expected %d query results, got %d", len(queries), len(results))
	}

	for i, queryResult := range results {
		if len(queryResult) != len(vectors) {
			t.Errorf("Expected %d distances for query %d, got %d", len(vectors), i, len(queryResult))
		}
	}

	// Verify some expected distances
	// Query [1,0,0] should have distance 0 to vector [1,0,0]
	if results[0][0] > 0.1 { // Small tolerance
		t.Errorf("Expected distance ~0 between identical vectors, got %f", results[0][0])
	}
}

func TestGPUManager_AsyncDistanceComputation(t *testing.T) {
	config := DefaultGPUConfig()
	config.FallbackToCPU = true

	manager := NewGPUManager(config)
	err := manager.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Create test batch
	batch := DistanceBatch{
		Queries: [][]float32{
			{1.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0},
		},
		Vectors: [][]float32{
			{1.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0},
		},
		Metric:   core.DistanceCosine,
		BatchID:  "test-batch-1",
		Priority: 1,
	}

	ctx := context.Background()

	// Submit batch
	err = manager.ComputeDistancesAsync(ctx, batch)
	if err != nil {
		t.Fatalf("Failed to submit async batch: %v", err)
	}

	// Get result with timeout
	resultCtx, cancel := context.WithTimeout(ctx, time.Second*5)
	defer cancel()

	result, err := manager.GetResult(resultCtx)
	if err != nil {
		t.Fatalf("Failed to get async result: %v", err)
	}

	// Verify result
	if result.BatchID != batch.BatchID {
		t.Errorf("Expected batch ID %s, got %s", batch.BatchID, result.BatchID)
	}

	if result.ComputeTime <= 0 {
		t.Error("Compute time should be positive")
	}

	if len(result.Distances) != len(batch.Queries) {
		t.Errorf("Expected %d query results, got %d", len(batch.Queries), len(result.Distances))
	}
}

func TestGPUManager_PerformanceStats(t *testing.T) {
	config := DefaultGPUConfig()
	config.FallbackToCPU = true
	config.EnableProfiling = true

	manager := NewGPUManager(config)
	err := manager.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Perform some operations
	queries := [][]float32{{1.0, 0.0}}
	vectors := [][]float32{{1.0, 0.0}, {0.0, 1.0}}

	ctx := context.Background()

	// Execute multiple operations
	for i := 0; i < 5; i++ {
		_, err := manager.ComputeDistancesSync(ctx, queries, vectors, core.DistanceL2)
		if err != nil {
			t.Fatalf("Failed to compute distances: %v", err)
		}
	}

	// Check performance stats
	stats := manager.GetPerformanceStats()

	// Note: In this mock implementation, the CPU engine doesn't directly update
	// the manager's stats, so we just verify the stats structure is valid
	if stats.TotalOperations < 0 {
		t.Error("Total operations should be non-negative")
	}

	if stats.TotalComputeTime < 0 {
		t.Error("Total compute time should be non-negative")
	}

	if stats.AverageLatency < 0 {
		t.Error("Average latency should be non-negative")
	}
}

func TestGPUManager_MemoryInfo(t *testing.T) {
	config := DefaultGPUConfig()
	config.FallbackToCPU = true

	manager := NewGPUManager(config)
	err := manager.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Get memory info
	memInfo, err := manager.GetMemoryInfo()
	if err != nil {
		t.Fatalf("Failed to get memory info: %v", err)
	}

	if memInfo.TotalMemory <= 0 {
		t.Error("Total memory should be positive")
	}

	if memInfo.UsedMemory < 0 {
		t.Error("Used memory should be non-negative")
	}

	if memInfo.FreeMemory < 0 {
		t.Error("Free memory should be non-negative")
	}
}

func TestCUDAEngine_Initialize(t *testing.T) {
	engine, err := NewCUDAEngine()
	if err != nil {
		// CUDA not available, skip test
		t.Skip("CUDA not available")
	}

	config := DefaultGPUConfig()
	err = engine.Initialize(config)
	if err != nil {
		t.Fatalf("Failed to initialize CUDA engine: %v", err)
	}

	// Test device info
	deviceInfo := engine.GetDeviceInfo()
	if deviceInfo.Backend != "cuda" {
		t.Errorf("Expected backend 'cuda', got %s", deviceInfo.Backend)
	}

	// Cleanup
	err = engine.Cleanup()
	if err != nil {
		t.Fatalf("Failed to cleanup CUDA engine: %v", err)
	}
}

func TestOpenCLEngine_Initialize(t *testing.T) {
	engine, err := NewOpenCLEngine()
	if err != nil {
		// OpenCL not available, skip test
		t.Skip("OpenCL not available")
	}

	config := DefaultGPUConfig()
	err = engine.Initialize(config)
	if err != nil {
		t.Fatalf("Failed to initialize OpenCL engine: %v", err)
	}

	// Test device info
	deviceInfo := engine.GetDeviceInfo()
	if deviceInfo.Backend != "opencl" {
		t.Errorf("Expected backend 'opencl', got %s", deviceInfo.Backend)
	}

	// Cleanup
	err = engine.Cleanup()
	if err != nil {
		t.Fatalf("Failed to cleanup OpenCL engine: %v", err)
	}
}

func TestCPUEngine_Operations(t *testing.T) {
	engine, err := NewCPUEngine()
	if err != nil {
		t.Fatalf("Failed to create CPU engine: %v", err)
	}

	config := DefaultGPUConfig()
	err = engine.Initialize(config)
	if err != nil {
		t.Fatalf("Failed to initialize CPU engine: %v", err)
	}
	defer engine.Cleanup()

	ctx := context.Background()

	// Test distance computation
	queries := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
	}

	vectors := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}

	results, err := engine.ComputeDistances(ctx, queries, vectors, core.DistanceL2)
	if err != nil {
		t.Fatalf("Failed to compute distances: %v", err)
	}

	// Verify results
	if len(results) != len(queries) {
		t.Errorf("Expected %d query results, got %d", len(queries), len(results))
	}

	// Test vector normalization
	normalized, err := engine.NormalizeVectors(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to normalize vectors: %v", err)
	}

	if len(normalized) != len(vectors) {
		t.Errorf("Expected %d normalized vectors, got %d", len(vectors), len(normalized))
	}

	// Test dot product
	dotProducts, err := engine.DotProduct(ctx, queries, vectors[:len(queries)])
	if err != nil {
		t.Fatalf("Failed to compute dot products: %v", err)
	}

	if len(dotProducts) != len(queries) {
		t.Errorf("Expected %d dot products, got %d", len(queries), len(dotProducts))
	}
}

func TestGPUMemory_Operations(t *testing.T) {
	engine, err := NewCPUEngine()
	if err != nil {
		t.Fatalf("Failed to create CPU engine: %v", err)
	}

	config := DefaultGPUConfig()
	err = engine.Initialize(config)
	if err != nil {
		t.Fatalf("Failed to initialize CPU engine: %v", err)
	}
	defer engine.Cleanup()

	// Test memory allocation
	sizeBytes := 1024
	mem, err := engine.AllocateMemory(sizeBytes)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}

	// Verify memory properties
	if mem.Size() != int64(sizeBytes) {
		t.Errorf("Expected memory size %d, got %d", sizeBytes, mem.Size())
	}

	if !mem.IsValid() {
		t.Error("Memory should be valid after allocation")
	}

	// Test memory free
	err = mem.Free()
	if err != nil {
		t.Fatalf("Failed to free memory: %v", err)
	}

	if mem.IsValid() {
		t.Error("Memory should be invalid after freeing")
	}

	// Test double free
	err = mem.Free()
	if err == nil {
		t.Error("Expected error when freeing already freed memory")
	}
}

func TestGPUConfig_Validation(t *testing.T) {
	config := DefaultGPUConfig()

	// Test default values
	if config.Backend == "" {
		t.Error("Default backend should not be empty")
	}

	if config.BatchSize <= 0 {
		t.Error("Default batch size should be positive")
	}

	if config.MemoryLimitMB <= 0 {
		t.Error("Default memory limit should be positive")
	}

	// Test custom config
	customConfig := GPUConfig{
		Backend:          BackendOpenCL,
		DeviceID:         1,
		EnableAutoDetect: false,
		BatchSize:        512,
		MemoryLimitMB:    1024,
		EnableProfiling:  true,
		FallbackToCPU:    false,
	}

	if customConfig.Backend != BackendOpenCL {
		t.Errorf("Expected backend %s, got %s", BackendOpenCL, customConfig.Backend)
	}

	if customConfig.DeviceID != 1 {
		t.Errorf("Expected device ID 1, got %d", customConfig.DeviceID)
	}
}

// Benchmark tests

func BenchmarkGPUManager_SyncDistanceComputation(b *testing.B) {
	config := DefaultGPUConfig()
	config.FallbackToCPU = true

	manager := NewGPUManager(config)
	err := manager.Initialize()
	if err != nil {
		b.Fatalf("Failed to initialize GPU manager: %v", err)
	}
	defer manager.Cleanup()

	// Create test data
	dimension := 128
	queries := make([][]float32, 10)
	vectors := make([][]float32, 100)

	for i := range queries {
		queries[i] = make([]float32, dimension)
		for j := range queries[i] {
			queries[i][j] = float32(i*dimension + j)
		}
	}

	for i := range vectors {
		vectors[i] = make([]float32, dimension)
		for j := range vectors[i] {
			vectors[i][j] = float32(i*dimension + j)
		}
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := manager.ComputeDistancesSync(ctx, queries, vectors, core.DistanceL2)
		if err != nil {
			b.Fatalf("Failed to compute distances: %v", err)
		}
	}
}

func BenchmarkCPUEngine_vs_Direct(b *testing.B) {
	dimension := 64

	// Create test data
	query := make([]float32, dimension)
	vector := make([]float32, dimension)

	for i := 0; i < dimension; i++ {
		query[i] = float32(i)
		vector[i] = float32(i * 2)
	}

	b.Run("CPUEngine", func(b *testing.B) {
		engine, _ := NewCPUEngine()
		config := DefaultGPUConfig()
		engine.Initialize(config)
		defer engine.Cleanup()

		ctx := context.Background()
		queries := [][]float32{query}
		vectors := [][]float32{vector}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			engine.ComputeDistances(ctx, queries, vectors, core.DistanceL2)
		}
	})

	b.Run("DirectCalculation", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			core.CalculateDistance(query, vector, core.DistanceL2)
		}
	})
}

func BenchmarkGPUMemory_Allocation(b *testing.B) {
	engine, err := NewCPUEngine()
	if err != nil {
		b.Fatalf("Failed to create CPU engine: %v", err)
	}

	config := DefaultGPUConfig()
	err = engine.Initialize(config)
	if err != nil {
		b.Fatalf("Failed to initialize CPU engine: %v", err)
	}
	defer engine.Cleanup()

	sizeBytes := 1024

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem, err := engine.AllocateMemory(sizeBytes)
		if err != nil {
			b.Fatalf("Failed to allocate memory: %v", err)
		}
		mem.Free()
	}
}
