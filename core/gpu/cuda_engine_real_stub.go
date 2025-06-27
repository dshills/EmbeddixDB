// +build !cuda

package gpu

import (
	"context"
	"fmt"

	"github.com/dshills/EmbeddixDB/core"
)

// RealCUDAEngine is a stub for non-CUDA builds
type RealCUDAEngine struct {
	config GPUConfig
}

func NewRealCUDAEngine(config GPUConfig) (*RealCUDAEngine, error) {
	return nil, fmt.Errorf("CUDA support not compiled in this build")
}

func (e *RealCUDAEngine) Initialize(config GPUConfig) error {
	return fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) ComputeDistances(
	ctx context.Context,
	queries [][]float32,
	vectors [][]float32,
	metric core.DistanceMetric,
) ([][]float32, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) NormalizeVectors(ctx context.Context, vectors [][]float32) ([][]float32, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) DotProduct(
	ctx context.Context,
	a, b [][]float32,
) ([]float32, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) GetMemoryInfo() MemoryInfo {
	return MemoryInfo{}
}

func (e *RealCUDAEngine) GetPerformanceStats() PerformanceStats {
	return PerformanceStats{}
}

func (e *RealCUDAEngine) Cleanup() error {
	return nil
}

func (e *RealCUDAEngine) GetDeviceInfo() DeviceInfo {
	return DeviceInfo{
		Name:    "No CUDA Device",
		Backend: "cuda",
	}
}

func (e *RealCUDAEngine) AllocateMemory(sizeBytes int) (GPUMemory, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) CopyToGPU(data interface{}) (GPUMemory, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) CopyFromGPU(mem GPUMemory) (interface{}, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (e *RealCUDAEngine) BatchComputeDistances(ctx context.Context, batch DistanceBatch) (*DistanceResult, error) {
	return nil, fmt.Errorf("CUDA not available")
}