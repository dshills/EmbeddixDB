// +build !cuda

package gpu

import (
	"fmt"
	"unsafe"
)

// CUDAKernelManager is a stub for non-CUDA builds
type CUDAKernelManager struct{}

func NewCUDAKernelManager() (*CUDAKernelManager, error) {
	return nil, fmt.Errorf("CUDA support not compiled in this build")
}

func (m *CUDAKernelManager) initialize() error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) AllocateMemory(size int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) FreeMemory(ptr unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) CopyToDevice(dst unsafe.Pointer, src []float32) error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) CopyFromDevice(dst []float32, src unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) Synchronize() error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) LaunchCosineDistanceKernel(
	query, vectors, distances unsafe.Pointer,
	dimension, numVectors int,
) error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) LaunchL2DistanceKernel(
	query, vectors, distances unsafe.Pointer,
	dimension, numVectors int,
) error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) LaunchDotProductKernel(
	query, vectors, distances unsafe.Pointer,
	dimension, numVectors int,
) error {
	return fmt.Errorf("CUDA not available")
}

func (m *CUDAKernelManager) Cleanup() error {
	return nil
}

func (m *CUDAKernelManager) GetDeviceInfo() (DeviceInfo, error) {
	return DeviceInfo{}, fmt.Errorf("CUDA not available")
}