// +build opencl

package gpu

// #cgo LDFLAGS: -lOpenCL
// #include <CL/cl.h>
// #include <stdlib.h>
// #include <string.h>
//
// // OpenCL kernel sources
// const char* cosine_distance_cl_src = R"(
// __kernel void cosine_distance(
//     __global const float* query,
//     __global const float* vectors,
//     __global float* distances,
//     const int dimension,
//     const int num_vectors) {
//     
//     int idx = get_global_id(0);
//     if (idx >= num_vectors) return;
//     
//     // Local memory for query vector
//     __local float local_query[256]; // Adjust size as needed
//     
//     // Load query into local memory cooperatively
//     int local_id = get_local_id(0);
//     int local_size = get_local_size(0);
//     for (int i = local_id; i < dimension; i += local_size) {
//         local_query[i] = query[i];
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//     
//     // Compute dot product and norms
//     float dot = 0.0f, norm_q = 0.0f, norm_v = 0.0f;
//     __global const float* vec_ptr = vectors + idx * dimension;
//     
//     for (int i = 0; i < dimension; i++) {
//         float q = local_query[i];
//         float v = vec_ptr[i];
//         dot += q * v;
//         norm_q += q * q;
//         norm_v += v * v;
//     }
//     
//     // Compute cosine distance
//     float cosine_sim = dot / (sqrt(norm_q) * sqrt(norm_v) + 1e-8f);
//     distances[idx] = 1.0f - cosine_sim;
// }
// )";
//
// const char* l2_distance_cl_src = R"(
// __kernel void l2_distance(
//     __global const float* query,
//     __global const float* vectors,
//     __global float* distances,
//     const int dimension,
//     const int num_vectors) {
//     
//     int idx = get_global_id(0);
//     if (idx >= num_vectors) return;
//     
//     // Local memory for query vector
//     __local float local_query[256]; // Adjust size as needed
//     
//     // Load query into local memory cooperatively
//     int local_id = get_local_id(0);
//     int local_size = get_local_size(0);
//     for (int i = local_id; i < dimension; i += local_size) {
//         local_query[i] = query[i];
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//     
//     // Compute L2 distance
//     float sum = 0.0f;
//     __global const float* vec_ptr = vectors + idx * dimension;
//     
//     // Unroll loop for better performance
//     int i = 0;
//     for (; i < dimension - 3; i += 4) {
//         float d0 = local_query[i] - vec_ptr[i];
//         float d1 = local_query[i+1] - vec_ptr[i+1];
//         float d2 = local_query[i+2] - vec_ptr[i+2];
//         float d3 = local_query[i+3] - vec_ptr[i+3];
//         sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
//     }
//     
//     // Handle remaining elements
//     for (; i < dimension; i++) {
//         float diff = local_query[i] - vec_ptr[i];
//         sum += diff * diff;
//     }
//     
//     distances[idx] = sqrt(sum);
// }
// )";
//
// const char* dot_product_cl_src = R"(
// __kernel void dot_product(
//     __global const float* query,
//     __global const float* vectors,
//     __global float* distances,
//     const int dimension,
//     const int num_vectors) {
//     
//     int idx = get_global_id(0);
//     if (idx >= num_vectors) return;
//     
//     // Local memory for query vector
//     __local float local_query[256]; // Adjust size as needed
//     
//     // Load query into local memory cooperatively
//     int local_id = get_local_id(0);
//     int local_size = get_local_size(0);
//     for (int i = local_id; i < dimension; i += local_size) {
//         local_query[i] = query[i];
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//     
//     // Compute dot product
//     float dot = 0.0f;
//     __global const float* vec_ptr = vectors + idx * dimension;
//     
//     // Vectorized computation
//     int i = 0;
//     for (; i < dimension - 3; i += 4) {
//         dot += local_query[i] * vec_ptr[i];
//         dot += local_query[i+1] * vec_ptr[i+1];
//         dot += local_query[i+2] * vec_ptr[i+2];
//         dot += local_query[i+3] * vec_ptr[i+3];
//     }
//     
//     // Handle remaining elements
//     for (; i < dimension; i++) {
//         dot += local_query[i] * vec_ptr[i];
//     }
//     
//     distances[idx] = -dot; // Negative for similarity to distance
// }
// )";
import "C"

import (
	"fmt"
	"unsafe"
)

// OpenCLKernelManager manages OpenCL kernels and operations
type OpenCLKernelManager struct {
	platform       C.cl_platform_id
	device         C.cl_device_id
	context        C.cl_context
	commandQueue   C.cl_command_queue
	kernels        map[string]C.cl_kernel
	programs       map[string]C.cl_program
	initialized    bool
	maxWorkGroupSize int
	maxLocalMemory   int
}

// NewOpenCLKernelManager creates a new OpenCL kernel manager
func NewOpenCLKernelManager() (*OpenCLKernelManager, error) {
	manager := &OpenCLKernelManager{
		kernels:  make(map[string]C.cl_kernel),
		programs: make(map[string]C.cl_program),
	}

	if err := manager.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize OpenCL: %w", err)
	}

	if err := manager.compileKernels(); err != nil {
		manager.Cleanup()
		return nil, fmt.Errorf("failed to compile kernels: %w", err)
	}

	return manager, nil
}

// initialize sets up OpenCL platform, device and context
func (m *OpenCLKernelManager) initialize() error {
	var err C.cl_int
	
	// Get platform
	err = C.clGetPlatformIDs(1, &m.platform, nil)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to get platform: %d", int(err))
	}

	// Get GPU device
	err = C.clGetDeviceIDs(m.platform, C.CL_DEVICE_TYPE_GPU, 1, &m.device, nil)
	if err != C.CL_SUCCESS {
		// Try CPU fallback
		err = C.clGetDeviceIDs(m.platform, C.CL_DEVICE_TYPE_CPU, 1, &m.device, nil)
		if err != C.CL_SUCCESS {
			return fmt.Errorf("failed to get device: %d", int(err))
		}
	}

	// Create context
	m.context = C.clCreateContext(nil, 1, &m.device, nil, nil, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to create context: %d", int(err))
	}

	// Create command queue
	m.commandQueue = C.clCreateCommandQueue(m.context, m.device, 0, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to create command queue: %d", int(err))
	}

	// Get device capabilities
	var maxWorkGroupSize C.size_t
	C.clGetDeviceInfo(m.device, C.CL_DEVICE_MAX_WORK_GROUP_SIZE,
		C.size_t(unsafe.Sizeof(maxWorkGroupSize)), unsafe.Pointer(&maxWorkGroupSize), nil)
	m.maxWorkGroupSize = int(maxWorkGroupSize)

	var maxLocalMemory C.cl_ulong
	C.clGetDeviceInfo(m.device, C.CL_DEVICE_LOCAL_MEM_SIZE,
		C.size_t(unsafe.Sizeof(maxLocalMemory)), unsafe.Pointer(&maxLocalMemory), nil)
	m.maxLocalMemory = int(maxLocalMemory)

	m.initialized = true
	return nil
}

// compileKernels compiles all OpenCL kernels
func (m *OpenCLKernelManager) compileKernels() error {
	kernelSources := map[string]string{
		"cosine_distance": C.GoString(C.cosine_distance_cl_src),
		"l2_distance":     C.GoString(C.l2_distance_cl_src),
		"dot_product":     C.GoString(C.dot_product_cl_src),
	}

	for name, source := range kernelSources {
		program, kernel, err := m.compileKernel(source, name)
		if err != nil {
			return fmt.Errorf("failed to compile %s kernel: %w", name, err)
		}
		m.programs[name] = program
		m.kernels[name] = kernel
	}

	return nil
}

// compileKernel compiles a single OpenCL kernel
func (m *OpenCLKernelManager) compileKernel(source string, kernelName string) (C.cl_program, C.cl_kernel, error) {
	var err C.cl_int
	
	// Create program from source
	sourcePtr := C.CString(source)
	defer C.free(unsafe.Pointer(sourcePtr))
	
	sourceLen := C.size_t(len(source))
	program := C.clCreateProgramWithSource(m.context, 1, &sourcePtr, &sourceLen, &err)
	if err != C.CL_SUCCESS {
		return nil, nil, fmt.Errorf("failed to create program: %d", int(err))
	}

	// Build program
	err = C.clBuildProgram(program, 1, &m.device, nil, nil, nil)
	if err != C.CL_SUCCESS {
		// Get build log
		var logSize C.size_t
		C.clGetProgramBuildInfo(program, m.device, C.CL_PROGRAM_BUILD_LOG, 0, nil, &logSize)
		
		log := make([]byte, logSize)
		C.clGetProgramBuildInfo(program, m.device, C.CL_PROGRAM_BUILD_LOG, logSize, unsafe.Pointer(&log[0]), nil)
		
		C.clReleaseProgram(program)
		return nil, nil, fmt.Errorf("failed to build program: %s", string(log))
	}

	// Create kernel
	kernelNameC := C.CString(kernelName)
	defer C.free(unsafe.Pointer(kernelNameC))
	
	kernel := C.clCreateKernel(program, kernelNameC, &err)
	if err != C.CL_SUCCESS {
		C.clReleaseProgram(program)
		return nil, nil, fmt.Errorf("failed to create kernel: %d", int(err))
	}

	return program, kernel, nil
}

// AllocateBuffer allocates GPU buffer
func (m *OpenCLKernelManager) AllocateBuffer(size int, readOnly bool) (unsafe.Pointer, error) {
	if !m.initialized {
		return nil, fmt.Errorf("OpenCL not initialized")
	}

	var flags C.cl_mem_flags = C.CL_MEM_READ_WRITE
	if readOnly {
		flags = C.CL_MEM_READ_ONLY
	}

	var err C.cl_int
	buffer := C.clCreateBuffer(m.context, flags, C.size_t(size), nil, &err)
	if err != C.CL_SUCCESS {
		return nil, fmt.Errorf("failed to create buffer: %d", int(err))
	}

	return unsafe.Pointer(buffer), nil
}

// FreeBuffer releases GPU buffer
func (m *OpenCLKernelManager) FreeBuffer(buffer unsafe.Pointer) error {
	if buffer == nil {
		return nil
	}

	err := C.clReleaseMemObject(C.cl_mem(buffer))
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to release buffer: %d", int(err))
	}

	return nil
}

// WriteBuffer writes data to GPU buffer
func (m *OpenCLKernelManager) WriteBuffer(buffer unsafe.Pointer, data []float32) error {
	if !m.initialized {
		return fmt.Errorf("OpenCL not initialized")
	}

	size := len(data) * 4 // float32 = 4 bytes
	err := C.clEnqueueWriteBuffer(m.commandQueue, C.cl_mem(buffer), C.CL_TRUE,
		0, C.size_t(size), unsafe.Pointer(&data[0]), 0, nil, nil)
	
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to write buffer: %d", int(err))
	}

	return nil
}

// ReadBuffer reads data from GPU buffer
func (m *OpenCLKernelManager) ReadBuffer(buffer unsafe.Pointer, data []float32) error {
	if !m.initialized {
		return fmt.Errorf("OpenCL not initialized")
	}

	size := len(data) * 4 // float32 = 4 bytes
	err := C.clEnqueueReadBuffer(m.commandQueue, C.cl_mem(buffer), C.CL_TRUE,
		0, C.size_t(size), unsafe.Pointer(&data[0]), 0, nil, nil)
	
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to read buffer: %d", int(err))
	}

	return nil
}

// LaunchCosineDistanceKernel launches the cosine distance kernel
func (m *OpenCLKernelManager) LaunchCosineDistanceKernel(
	queryBuffer unsafe.Pointer,
	vectorsBuffer unsafe.Pointer,
	distancesBuffer unsafe.Pointer,
	dimension int,
	numVectors int,
) error {
	return m.launchKernel("cosine_distance", queryBuffer, vectorsBuffer, distancesBuffer, dimension, numVectors)
}

// LaunchL2DistanceKernel launches the L2 distance kernel
func (m *OpenCLKernelManager) LaunchL2DistanceKernel(
	queryBuffer unsafe.Pointer,
	vectorsBuffer unsafe.Pointer,
	distancesBuffer unsafe.Pointer,
	dimension int,
	numVectors int,
) error {
	return m.launchKernel("l2_distance", queryBuffer, vectorsBuffer, distancesBuffer, dimension, numVectors)
}

// LaunchDotProductKernel launches the dot product kernel
func (m *OpenCLKernelManager) LaunchDotProductKernel(
	queryBuffer unsafe.Pointer,
	vectorsBuffer unsafe.Pointer,
	distancesBuffer unsafe.Pointer,
	dimension int,
	numVectors int,
) error {
	return m.launchKernel("dot_product", queryBuffer, vectorsBuffer, distancesBuffer, dimension, numVectors)
}

// launchKernel launches a specific kernel
func (m *OpenCLKernelManager) launchKernel(
	kernelName string,
	queryBuffer unsafe.Pointer,
	vectorsBuffer unsafe.Pointer,
	distancesBuffer unsafe.Pointer,
	dimension int,
	numVectors int,
) error {
	if !m.initialized {
		return fmt.Errorf("OpenCL not initialized")
	}

	kernel, exists := m.kernels[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not found", kernelName)
	}

	// Set kernel arguments
	queryMem := C.cl_mem(queryBuffer)
	vectorsMem := C.cl_mem(vectorsBuffer)
	distancesMem := C.cl_mem(distancesBuffer)
	dim := C.cl_int(dimension)
	numVec := C.cl_int(numVectors)

	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(queryMem)), unsafe.Pointer(&queryMem))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(vectorsMem)), unsafe.Pointer(&vectorsMem))
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(distancesMem)), unsafe.Pointer(&distancesMem))
	C.clSetKernelArg(kernel, 3, C.size_t(unsafe.Sizeof(dim)), unsafe.Pointer(&dim))
	C.clSetKernelArg(kernel, 4, C.size_t(unsafe.Sizeof(numVec)), unsafe.Pointer(&numVec))

	// Calculate work sizes
	localSize := C.size_t(256)
	if dimension < 256 {
		localSize = C.size_t(128)
	}
	globalSize := C.size_t(((numVectors + int(localSize) - 1) / int(localSize)) * int(localSize))

	// Launch kernel
	err := C.clEnqueueNDRangeKernel(m.commandQueue, kernel, 1, nil, &globalSize, &localSize, 0, nil, nil)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to launch kernel: %d", int(err))
	}

	// Wait for completion
	err = C.clFinish(m.commandQueue)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to finish execution: %d", int(err))
	}

	return nil
}

// Cleanup releases OpenCL resources
func (m *OpenCLKernelManager) Cleanup() error {
	if !m.initialized {
		return nil
	}

	// Release kernels
	for _, kernel := range m.kernels {
		C.clReleaseKernel(kernel)
	}

	// Release programs
	for _, program := range m.programs {
		C.clReleaseProgram(program)
	}

	// Release command queue
	if m.commandQueue != nil {
		C.clReleaseCommandQueue(m.commandQueue)
	}

	// Release context
	if m.context != nil {
		C.clReleaseContext(m.context)
	}

	m.initialized = false
	return nil
}

// GetDeviceInfo returns information about the OpenCL device
func (m *OpenCLKernelManager) GetDeviceInfo() (DeviceInfo, error) {
	if !m.initialized {
		return DeviceInfo{}, fmt.Errorf("OpenCL not initialized")
	}

	var name [256]C.char
	C.clGetDeviceInfo(m.device, C.CL_DEVICE_NAME, 256, unsafe.Pointer(&name[0]), nil)

	var globalMem C.cl_ulong
	C.clGetDeviceInfo(m.device, C.CL_DEVICE_GLOBAL_MEM_SIZE,
		C.size_t(unsafe.Sizeof(globalMem)), unsafe.Pointer(&globalMem), nil)

	var deviceType C.cl_device_type
	C.clGetDeviceInfo(m.device, C.CL_DEVICE_TYPE,
		C.size_t(unsafe.Sizeof(deviceType)), unsafe.Pointer(&deviceType), nil)

	typeStr := "Unknown"
	if deviceType&C.CL_DEVICE_TYPE_GPU != 0 {
		typeStr = "GPU"
	} else if deviceType&C.CL_DEVICE_TYPE_CPU != 0 {
		typeStr = "CPU"
	}

	return DeviceInfo{
		Name:               C.GoString(&name[0]),
		TotalMemory:        int64(globalMem),
		ComputeCapability:  typeStr,
		MaxThreadsPerBlock: m.maxWorkGroupSize,
		MaxSharedMemory:    m.maxLocalMemory,
	}, nil
}