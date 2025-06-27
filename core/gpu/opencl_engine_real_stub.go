//go:build !opencl
// +build !opencl

package gpu

import (
	"context"
	"fmt"

	"github.com/dshills/EmbeddixDB/core"
)

// RealOpenCLEngine is a stub for non-OpenCL builds
type RealOpenCLEngine struct {
	config GPUConfig
}

func NewRealOpenCLEngine(config GPUConfig) (*RealOpenCLEngine, error) {
	return nil, fmt.Errorf("OpenCL support not compiled in this build")
}

func (e *RealOpenCLEngine) Initialize(config GPUConfig) error {
	return fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) ComputeDistances(
	ctx context.Context,
	queries [][]float32,
	vectors [][]float32,
	metric core.DistanceMetric,
) ([][]float32, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) NormalizeVectors(ctx context.Context, vectors [][]float32) ([][]float32, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) DotProduct(
	ctx context.Context,
	a, b [][]float32,
) ([]float32, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) GetMemoryInfo() MemoryInfo {
	return MemoryInfo{}
}

func (e *RealOpenCLEngine) GetPerformanceStats() PerformanceStats {
	return PerformanceStats{}
}

func (e *RealOpenCLEngine) Cleanup() error {
	return nil
}

func (e *RealOpenCLEngine) GetDeviceInfo() DeviceInfo {
	return DeviceInfo{
		Name:    "No OpenCL Device",
		Backend: "opencl",
	}
}

func (e *RealOpenCLEngine) AllocateMemory(sizeBytes int) (GPUMemory, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) CopyToGPU(data interface{}) (GPUMemory, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) CopyFromGPU(mem GPUMemory) (interface{}, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (e *RealOpenCLEngine) BatchComputeDistances(ctx context.Context, batch DistanceBatch) (*DistanceResult, error) {
	return nil, fmt.Errorf("OpenCL not available")
}
