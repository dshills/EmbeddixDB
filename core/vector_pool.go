package core

import (
	"sync"
	"unsafe"
)

// AlignedVector represents a vector aligned to CPU cache line boundaries
type AlignedVector struct {
	_    [0]func() // prevent comparison
	data *[]float32
	len  int
	cap  int
}

// NewAlignedVector creates a new cache-line aligned vector
func NewAlignedVector(size int) *AlignedVector {
	// Allocate aligned memory (64-byte boundary for CPU cache line)
	data := make([]float32, size+15) // Extra space for alignment

	// Calculate aligned offset
	ptr := uintptr(unsafe.Pointer(&data[0]))
	aligned := (ptr + 63) &^ 63 // Align to 64-byte boundary
	offset := int(aligned-ptr) / int(unsafe.Sizeof(float32(0)))

	alignedSlice := data[offset : offset+size]

	return &AlignedVector{
		data: &alignedSlice,
		len:  size,
		cap:  size,
	}
}

// Data returns the underlying float32 slice
func (av *AlignedVector) Data() []float32 {
	return (*av.data)[:av.len]
}

// Len returns the vector length
func (av *AlignedVector) Len() int {
	return av.len
}

// VectorPool manages reusable aligned vectors by dimension
type VectorPool struct {
	pools map[int]*sync.Pool
	mutex sync.RWMutex
}

// NewVectorPool creates a new vector pool
func NewVectorPool() *VectorPool {
	return &VectorPool{
		pools: make(map[int]*sync.Pool),
	}
}

// Get retrieves or creates an aligned vector of specified dimension
func (vp *VectorPool) Get(dimension int) *AlignedVector {
	vp.mutex.RLock()
	pool, exists := vp.pools[dimension]
	vp.mutex.RUnlock()

	if !exists {
		vp.mutex.Lock()
		// Double-check pattern
		if pool, exists = vp.pools[dimension]; !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return NewAlignedVector(dimension)
				},
			}
			vp.pools[dimension] = pool
		}
		vp.mutex.Unlock()
	}

	return pool.Get().(*AlignedVector)
}

// Put returns an aligned vector to the pool for reuse
func (vp *VectorPool) Put(vec *AlignedVector) {
	if vec == nil {
		return
	}

	dimension := vec.len
	vp.mutex.RLock()
	pool, exists := vp.pools[dimension]
	vp.mutex.RUnlock()

	if exists {
		// Clear the vector data for security
		data := vec.Data()
		for i := range data {
			data[i] = 0
		}
		pool.Put(vec)
	}
}

// Stats returns pool statistics
func (vp *VectorPool) Stats() map[int]int {
	vp.mutex.RLock()
	defer vp.mutex.RUnlock()

	stats := make(map[int]int)
	for dimension := range vp.pools {
		stats[dimension] = 1 // Pool exists
	}
	return stats
}

// Global vector pool instance
var globalVectorPool = NewVectorPool()

// GetPooledVector gets a vector from the global pool
func GetPooledVector(dimension int) *AlignedVector {
	return globalVectorPool.Get(dimension)
}

// PutPooledVector returns a vector to the global pool
func PutPooledVector(vec *AlignedVector) {
	globalVectorPool.Put(vec)
}

// VectorBuffer manages a reusable buffer for batch operations
type VectorBuffer struct {
	vectors   []*AlignedVector
	pool      *VectorPool
	dimension int
	capacity  int
}

// NewVectorBuffer creates a buffer for batch vector operations
func NewVectorBuffer(dimension, capacity int) *VectorBuffer {
	return &VectorBuffer{
		vectors:   make([]*AlignedVector, 0, capacity),
		pool:      NewVectorPool(),
		dimension: dimension,
		capacity:  capacity,
	}
}

// Get returns a vector from the buffer, allocating if necessary
func (vb *VectorBuffer) Get() *AlignedVector {
	if len(vb.vectors) > 0 {
		vec := vb.vectors[len(vb.vectors)-1]
		vb.vectors = vb.vectors[:len(vb.vectors)-1]
		return vec
	}
	return vb.pool.Get(vb.dimension)
}

// Put returns a vector to the buffer
func (vb *VectorBuffer) Put(vec *AlignedVector) {
	if len(vb.vectors) < vb.capacity {
		vb.vectors = append(vb.vectors, vec)
	} else {
		vb.pool.Put(vec)
	}
}

// Clear returns all vectors to the pool
func (vb *VectorBuffer) Clear() {
	for _, vec := range vb.vectors {
		vb.pool.Put(vec)
	}
	vb.vectors = vb.vectors[:0]
}

// BatchVectorProcessor processes vectors in batches with optimal memory usage
type BatchVectorProcessor struct {
	buffer    *VectorBuffer
	batchSize int
}

// NewBatchVectorProcessor creates a new batch processor
func NewBatchVectorProcessor(dimension, batchSize int) *BatchVectorProcessor {
	return &BatchVectorProcessor{
		buffer:    NewVectorBuffer(dimension, batchSize*2), // 2x buffer for overlap
		batchSize: batchSize,
	}
}

// ProcessBatch processes a batch of vectors with the given function
func (bvp *BatchVectorProcessor) ProcessBatch(vectors []Vector, fn func([]Vector) error) error {
	for i := 0; i < len(vectors); i += bvp.batchSize {
		end := i + bvp.batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		batch := vectors[i:end]
		if err := fn(batch); err != nil {
			return err
		}
	}
	return nil
}

// MemoryAlignedVector wraps a regular vector with alignment information
type MemoryAlignedVector struct {
	Vector
	aligned *AlignedVector
}

// NewMemoryAlignedVector creates a memory-aligned wrapper for a vector
func NewMemoryAlignedVector(vec Vector) *MemoryAlignedVector {
	aligned := GetPooledVector(len(vec.Values))
	copy(aligned.Data(), vec.Values)

	return &MemoryAlignedVector{
		Vector:  vec,
		aligned: aligned,
	}
}

// AlignedData returns the aligned vector data
func (mav *MemoryAlignedVector) AlignedData() []float32 {
	return mav.aligned.Data()
}

// Release returns the aligned memory to the pool
func (mav *MemoryAlignedVector) Release() {
	if mav.aligned != nil {
		PutPooledVector(mav.aligned)
		mav.aligned = nil
	}
}

// IsAligned checks if a memory address is aligned to the specified boundary
func IsAligned(ptr uintptr, boundary uintptr) bool {
	return ptr&(boundary-1) == 0
}

// GetAlignment returns the memory alignment of a slice
func GetAlignment(data []float32) uintptr {
	if len(data) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&data[0])) & 63 // Check 64-byte alignment
}
