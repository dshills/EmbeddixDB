//go:build !opencl
// +build !opencl

package gpu

import (
	"fmt"
	"unsafe"
)

// OpenCLKernelManager is a stub for non-OpenCL builds
type OpenCLKernelManager struct{}

func NewOpenCLKernelManager() (*OpenCLKernelManager, error) {
	return nil, fmt.Errorf("OpenCL support not compiled in this build")
}

func (m *OpenCLKernelManager) initialize() error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) compileKernels() error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) AllocateBuffer(size int, readOnly bool) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) FreeBuffer(buffer unsafe.Pointer) error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) WriteBuffer(buffer unsafe.Pointer, data []float32) error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) ReadBuffer(buffer unsafe.Pointer, data []float32) error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) LaunchCosineDistanceKernel(
	queryBuffer, vectorsBuffer, distancesBuffer unsafe.Pointer,
	dimension, numVectors int,
) error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) LaunchL2DistanceKernel(
	queryBuffer, vectorsBuffer, distancesBuffer unsafe.Pointer,
	dimension, numVectors int,
) error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) LaunchDotProductKernel(
	queryBuffer, vectorsBuffer, distancesBuffer unsafe.Pointer,
	dimension, numVectors int,
) error {
	return fmt.Errorf("OpenCL not available")
}

func (m *OpenCLKernelManager) Cleanup() error {
	return nil
}

func (m *OpenCLKernelManager) GetDeviceInfo() (DeviceInfo, error) {
	return DeviceInfo{}, fmt.Errorf("OpenCL not available")
}
