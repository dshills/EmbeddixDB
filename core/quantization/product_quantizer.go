package quantization

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// ProductQuantizer implements Product Quantization for vector compression
type ProductQuantizer struct {
	config         ProductQuantizerConfig
	codebooks      [][][]float32 // [subvector][centroid][dimension]
	subvectorSize  int
	trained        bool
	trainingStats  TrainingStats
	distanceCache  *DistanceTableCache
	mu             sync.RWMutex
}

// ProductQuantizerConfig configures the Product Quantizer
type ProductQuantizerConfig struct {
	NumSubvectors    int           `json:"num_subvectors"`     // Number of subvectors (M)
	BitsPerSubvector int           `json:"bits_per_subvector"` // Bits per subvector (typically 4, 6, 8)
	Dimension        int           `json:"dimension"`          // Original vector dimension
	DistanceMetric   string        `json:"distance_metric"`    // "l2", "cosine", "dot"
	TrainingTimeout  time.Duration `json:"training_timeout"`
	KMeansRestarts   int           `json:"kmeans_restarts"`
	KMeansMaxIters   int           `json:"kmeans_max_iters"`
	EnableCache      bool          `json:"enable_cache"`       // Enable distance table caching
	CacheSize        int           `json:"cache_size"`         // Distance table cache size
}

// DefaultProductQuantizerConfig returns sensible defaults
func DefaultProductQuantizerConfig(dimension int) ProductQuantizerConfig {
	// Choose number of subvectors based on dimension
	numSubvectors := 8
	if dimension >= 512 {
		numSubvectors = 16
	} else if dimension >= 256 {
		numSubvectors = 8
	} else {
		numSubvectors = 4
	}
	
	// Ensure dimension is divisible by numSubvectors
	for dimension%numSubvectors != 0 && numSubvectors > 1 {
		numSubvectors--
	}
	
	return ProductQuantizerConfig{
		NumSubvectors:    numSubvectors,
		BitsPerSubvector: 8, // 256 centroids per subvector
		Dimension:        dimension,
		DistanceMetric:   "l2",
		TrainingTimeout:  10 * time.Minute,
		KMeansRestarts:   3,
		KMeansMaxIters:   100,
		EnableCache:      true,
		CacheSize:        1000,
	}
}

// NewProductQuantizer creates a new Product Quantizer
func NewProductQuantizer(config ProductQuantizerConfig) (*ProductQuantizer, error) {
	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	
	if config.NumSubvectors <= 0 {
		return nil, fmt.Errorf("num_subvectors must be positive")
	}
	
	if config.Dimension%config.NumSubvectors != 0 {
		return nil, fmt.Errorf("dimension (%d) must be divisible by num_subvectors (%d)", 
			config.Dimension, config.NumSubvectors)
	}
	
	if config.BitsPerSubvector < 1 || config.BitsPerSubvector > 16 {
		return nil, fmt.Errorf("bits_per_subvector must be between 1 and 16")
	}
	
	subvectorSize := config.Dimension / config.NumSubvectors
	numCentroids := 1 << config.BitsPerSubvector // 2^bits
	
	// Initialize codebooks
	codebooks := make([][][]float32, config.NumSubvectors)
	for i := range codebooks {
		codebooks[i] = make([][]float32, numCentroids)
		for j := range codebooks[i] {
			codebooks[i][j] = make([]float32, subvectorSize)
		}
	}
	
	pq := &ProductQuantizer{
		config:        config,
		codebooks:     codebooks,
		subvectorSize: subvectorSize,
		trained:       false,
	}
	
	if config.EnableCache {
		pq.distanceCache = NewDistanceTableCache(config.CacheSize)
	}
	
	return pq, nil
}

// Train builds the Product Quantizer from training data
func (pq *ProductQuantizer) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}
	
	startTime := time.Now()
	
	// Validate input vectors
	for i, vec := range vectors {
		if len(vec) != pq.config.Dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), pq.config.Dimension)
		}
	}
	
	numCentroids := 1 << uint(pq.config.BitsPerSubvector)
	
	// Train each subvector independently
	var wg sync.WaitGroup
	errors := make(chan error, pq.config.NumSubvectors)
	
	for m := 0; m < pq.config.NumSubvectors; m++ {
		wg.Add(1)
		go func(subvectorIdx int) {
			defer wg.Done()
			
			err := pq.trainSubvector(ctx, vectors, subvectorIdx, numCentroids)
			if err != nil {
				errors <- fmt.Errorf("failed to train subvector %d: %w", subvectorIdx, err)
			}
		}(m)
	}
	
	wg.Wait()
	close(errors)
	
	// Check for training errors
	for err := range errors {
		return err
	}
	
	pq.mu.Lock()
	pq.trained = true
	pq.trainingStats = TrainingStats{
		TrainingTime:    time.Since(startTime),
		MemoryReduction: pq.calculateMemoryReduction(),
		CodebookSize:    numCentroids * pq.config.NumSubvectors,
		TrainingVectors: len(vectors),
	}
	pq.mu.Unlock()
	
	return nil
}

// trainSubvector trains a single subvector using K-means
func (pq *ProductQuantizer) trainSubvector(ctx context.Context, vectors [][]float32, subvectorIdx, numCentroids int) error {
	// Extract subvectors for this position
	subvectors := make([][]float32, len(vectors))
	start := subvectorIdx * pq.subvectorSize
	end := start + pq.subvectorSize
	
	for i, vec := range vectors {
		subvectors[i] = make([]float32, pq.subvectorSize)
		copy(subvectors[i], vec[start:end])
	}
	
	// Configure K-means for this subvector
	kmeansConfig := KMeansConfig{
		K:              numCentroids,
		MaxIterations:  pq.config.KMeansMaxIters,
		Tolerance:      1e-6,
		InitMethod:     InitKMeansPP,
		NumRestarts:    pq.config.KMeansRestarts,
		Seed:           int64(subvectorIdx), // Different seed per subvector
		MinPointsPerCluster: 1,
		ParallelWorkers: 2, // Limit parallelism within subvector training
	}
	
	kmeans := NewKMeans(kmeansConfig)
	result, err := kmeans.Fit(ctx, subvectors)
	if err != nil {
		return fmt.Errorf("K-means failed for subvector %d: %w", subvectorIdx, err)
	}
	
	// Store centroids in codebook
	pq.mu.Lock()
	for i, centroid := range result.Centroids {
		copy(pq.codebooks[subvectorIdx][i], centroid)
	}
	pq.mu.Unlock()
	
	return nil
}

// Encode compresses a vector into a quantized code
func (pq *ProductQuantizer) Encode(vector []float32) ([]byte, error) {
	if !pq.IsTrained() {
		return nil, fmt.Errorf("quantizer not trained")
	}
	
	if len(vector) != pq.config.Dimension {
		return nil, fmt.Errorf("vector dimension %d does not match expected %d", len(vector), pq.config.Dimension)
	}
	
	// Calculate code size in bytes
	totalBits := pq.config.NumSubvectors * pq.config.BitsPerSubvector
	codeSize := (totalBits + 7) / 8 // Round up to nearest byte
	code := make([]byte, codeSize)
	
	bitOffset := 0
	
	for m := 0; m < pq.config.NumSubvectors; m++ {
		// Extract subvector
		start := m * pq.subvectorSize
		end := start + pq.subvectorSize
		subvector := vector[start:end]
		
		// Find nearest centroid
		centroidIdx := pq.findNearestCentroid(subvector, m)
		
		// Pack centroid index into code
		pq.packBits(code, bitOffset, centroidIdx, pq.config.BitsPerSubvector)
		bitOffset += pq.config.BitsPerSubvector
	}
	
	return code, nil
}

// Decode reconstructs an approximate vector from a quantized code
func (pq *ProductQuantizer) Decode(code []byte) ([]float32, error) {
	if !pq.IsTrained() {
		return nil, fmt.Errorf("quantizer not trained")
	}
	
	expectedSize := (pq.config.NumSubvectors*pq.config.BitsPerSubvector + 7) / 8
	if len(code) != expectedSize {
		return nil, fmt.Errorf("code size %d does not match expected %d", len(code), expectedSize)
	}
	
	vector := make([]float32, pq.config.Dimension)
	bitOffset := 0
	
	for m := 0; m < pq.config.NumSubvectors; m++ {
		// Unpack centroid index from code
		centroidIdx := pq.unpackBits(code, bitOffset, pq.config.BitsPerSubvector)
		bitOffset += pq.config.BitsPerSubvector
		
		// Copy centroid values to output vector
		start := m * pq.subvectorSize
		end := start + pq.subvectorSize
		
		pq.mu.RLock()
		copy(vector[start:end], pq.codebooks[m][centroidIdx])
		pq.mu.RUnlock()
	}
	
	return vector, nil
}

// Distance computes approximate distance between two quantized codes
func (pq *ProductQuantizer) Distance(codeA, codeB []byte) (float32, error) {
	if !pq.IsTrained() {
		return 0, fmt.Errorf("quantizer not trained")
	}
	
	if len(codeA) != len(codeB) {
		return 0, fmt.Errorf("code lengths do not match")
	}
	
	var totalDistance float32
	bitOffset := 0
	
	for m := 0; m < pq.config.NumSubvectors; m++ {
		// Unpack centroid indices
		centroidA := pq.unpackBits(codeA, bitOffset, pq.config.BitsPerSubvector)
		centroidB := pq.unpackBits(codeB, bitOffset, pq.config.BitsPerSubvector)
		bitOffset += pq.config.BitsPerSubvector
		
		// Compute distance between centroids
		pq.mu.RLock()
		dist := pq.computeSubvectorDistance(pq.codebooks[m][centroidA], pq.codebooks[m][centroidB])
		pq.mu.RUnlock()
		
		totalDistance += dist
	}
	
	return totalDistance, nil
}

// AsymmetricDistance computes distance between a quantized code and full vector
func (pq *ProductQuantizer) AsymmetricDistance(code []byte, vector []float32) (float32, error) {
	if !pq.IsTrained() {
		return 0, fmt.Errorf("quantizer not trained")
	}
	
	if len(vector) != pq.config.Dimension {
		return 0, fmt.Errorf("vector dimension mismatch")
	}
	
	var totalDistance float32
	bitOffset := 0
	
	for m := 0; m < pq.config.NumSubvectors; m++ {
		// Unpack centroid index
		centroidIdx := pq.unpackBits(code, bitOffset, pq.config.BitsPerSubvector)
		bitOffset += pq.config.BitsPerSubvector
		
		// Extract subvector
		start := m * pq.subvectorSize
		end := start + pq.subvectorSize
		subvector := vector[start:end]
		
		// Compute distance between subvector and centroid
		pq.mu.RLock()
		dist := pq.computeSubvectorDistance(subvector, pq.codebooks[m][centroidIdx])
		pq.mu.RUnlock()
		
		totalDistance += dist
	}
	
	return totalDistance, nil
}

// findNearestCentroid finds the nearest centroid for a subvector
func (pq *ProductQuantizer) findNearestCentroid(subvector []float32, subvectorIdx int) int {
	minDist := float32(math.Inf(1))
	bestCentroid := 0
	
	pq.mu.RLock()
	codebook := pq.codebooks[subvectorIdx]
	pq.mu.RUnlock()
	
	for i, centroid := range codebook {
		dist := pq.computeSubvectorDistance(subvector, centroid)
		if dist < minDist {
			minDist = dist
			bestCentroid = i
		}
	}
	
	return bestCentroid
}

// computeSubvectorDistance computes distance between two subvectors
func (pq *ProductQuantizer) computeSubvectorDistance(a, b []float32) float32 {
	switch pq.config.DistanceMetric {
	case "l2":
		return pq.l2Distance(a, b)
	case "cosine":
		return pq.cosineDistance(a, b)
	case "dot":
		return pq.dotDistance(a, b)
	default:
		return pq.l2Distance(a, b)
	}
}

// l2Distance computes L2 (Euclidean) distance
func (pq *ProductQuantizer) l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // Return squared distance for efficiency
}

// cosineDistance computes cosine distance (1 - cosine similarity)
func (pq *ProductQuantizer) cosineDistance(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 1.0 // Maximum distance for zero vectors
	}
	
	cosine := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - cosine
}

// dotDistance computes negative dot product (for maximum inner product search)
func (pq *ProductQuantizer) dotDistance(a, b []float32) float32 {
	var dotProduct float32
	for i := range a {
		dotProduct += a[i] * b[i]
	}
	return -dotProduct
}

// packBits packs bits into a byte array
func (pq *ProductQuantizer) packBits(data []byte, bitOffset, value, numBits int) {
	for i := 0; i < numBits; i++ {
		if value&(1<<i) != 0 {
			byteIdx := (bitOffset + i) / 8
			bitIdx := (bitOffset + i) % 8
			data[byteIdx] |= 1 << bitIdx
		}
	}
}

// unpackBits extracts bits from a byte array
func (pq *ProductQuantizer) unpackBits(data []byte, bitOffset, numBits int) int {
	value := 0
	for i := 0; i < numBits; i++ {
		byteIdx := (bitOffset + i) / 8
		bitIdx := (bitOffset + i) % 8
		if data[byteIdx]&(1<<bitIdx) != 0 {
			value |= 1 << i
		}
	}
	return value
}

// calculateMemoryReduction computes the compression ratio
func (pq *ProductQuantizer) calculateMemoryReduction() float64 {
	originalSize := pq.config.Dimension * 4 // 4 bytes per float32
	compressedSize := (pq.config.NumSubvectors*pq.config.BitsPerSubvector + 7) / 8
	return float64(originalSize) / float64(compressedSize)
}

// Interface implementations

func (pq *ProductQuantizer) MemoryReduction() float64 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return pq.calculateMemoryReduction()
}

func (pq *ProductQuantizer) CodeSize() int {
	return (pq.config.NumSubvectors*pq.config.BitsPerSubvector + 7) / 8
}

func (pq *ProductQuantizer) IsTrained() bool {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return pq.trained
}

func (pq *ProductQuantizer) Config() QuantizerConfig {
	return QuantizerConfig{
		Type:             ProductQuantization,
		Dimension:        pq.config.Dimension,
		MemoryBudgetMB:   0, // Not applicable for PQ
		TrainingTimeout:  pq.config.TrainingTimeout,
		DistanceMetric:   pq.config.DistanceMetric,
		EnableAsymmetric: true,
	}
}

// GetTrainingStats returns training statistics
func (pq *ProductQuantizer) GetTrainingStats() TrainingStats {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return pq.trainingStats
}

// GetCodebooks returns the trained codebooks (for serialization)
func (pq *ProductQuantizer) GetCodebooks() [][][]float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	
	// Deep copy to avoid race conditions
	codebooks := make([][][]float32, len(pq.codebooks))
	for i, subcodebook := range pq.codebooks {
		codebooks[i] = make([][]float32, len(subcodebook))
		for j, centroid := range subcodebook {
			codebooks[i][j] = make([]float32, len(centroid))
			copy(codebooks[i][j], centroid)
		}
	}
	
	return codebooks
}

// SetCodebooks sets the codebooks (for deserialization)
func (pq *ProductQuantizer) SetCodebooks(codebooks [][][]float32) error {
	if len(codebooks) != pq.config.NumSubvectors {
		return fmt.Errorf("codebook count mismatch: expected %d, got %d", pq.config.NumSubvectors, len(codebooks))
	}
	
	expectedCentroids := 1 << uint(pq.config.BitsPerSubvector)
	
	for i, subcodebook := range codebooks {
		if len(subcodebook) != expectedCentroids {
			return fmt.Errorf("centroid count mismatch in subvector %d: expected %d, got %d", 
				i, expectedCentroids, len(subcodebook))
		}
		
		for j, centroid := range subcodebook {
			if len(centroid) != pq.subvectorSize {
				return fmt.Errorf("centroid dimension mismatch in subvector %d, centroid %d: expected %d, got %d", 
					i, j, pq.subvectorSize, len(centroid))
			}
		}
	}
	
	pq.mu.Lock()
	pq.codebooks = codebooks
	pq.trained = true
	pq.mu.Unlock()
	
	return nil
}

// DistanceTableCache caches distance tables for efficient search
type DistanceTableCache struct {
	cache    map[string]*DistanceTableImpl
	maxSize  int
	mu       sync.RWMutex
}

// NewDistanceTableCache creates a new distance table cache
func NewDistanceTableCache(maxSize int) *DistanceTableCache {
	return &DistanceTableCache{
		cache:   make(map[string]*DistanceTableImpl),
		maxSize: maxSize,
	}
}

// DistanceTableImpl implements precomputed distance tables
type DistanceTableImpl struct {
	tables [][]float32 // [subvector][centroid]
	query  []float32
}

// CreateDistanceTable creates a distance table for efficient quantized search
func (pq *ProductQuantizer) CreateDistanceTable(queryVector []float32) (DistanceTable, error) {
	if !pq.IsTrained() {
		return nil, fmt.Errorf("quantizer not trained")
	}
	
	if len(queryVector) != pq.config.Dimension {
		return nil, fmt.Errorf("query vector dimension mismatch")
	}
	
	numCentroids := 1 << uint(pq.config.BitsPerSubvector)
	tables := make([][]float32, pq.config.NumSubvectors)
	
	for m := 0; m < pq.config.NumSubvectors; m++ {
		tables[m] = make([]float32, numCentroids)
		
		// Extract query subvector
		start := m * pq.subvectorSize
		end := start + pq.subvectorSize
		querySubvector := queryVector[start:end]
		
		// Precompute distances to all centroids
		pq.mu.RLock()
		for k, centroid := range pq.codebooks[m] {
			tables[m][k] = pq.computeSubvectorDistance(querySubvector, centroid)
		}
		pq.mu.RUnlock()
	}
	
	return &DistanceTableImpl{
		tables: tables,
		query:  queryVector,
	}, nil
}

// Precompute builds distance table for a query vector
func (dt *DistanceTableImpl) Precompute(queryVector []float32) error {
	// Already precomputed in constructor
	return nil
}

// Distance returns precomputed distance for a quantized code
func (dt *DistanceTableImpl) Distance(code []byte) float32 {
	var totalDistance float32
	bitOffset := 0
	bitsPerSubvector := len(dt.tables[0]) // Infer from table size
	
	// Calculate bits per subvector from number of centroids
	bits := 0
	for (1 << bits) < len(dt.tables[0]) {
		bits++
	}
	bitsPerSubvector = bits
	
	for m := 0; m < len(dt.tables); m++ {
		centroidIdx := 0
		for i := 0; i < bitsPerSubvector; i++ {
			byteIdx := (bitOffset + i) / 8
			bitIdx := (bitOffset + i) % 8
			if code[byteIdx]&(1<<bitIdx) != 0 {
				centroidIdx |= 1 << i
			}
		}
		bitOffset += bitsPerSubvector
		
		totalDistance += dt.tables[m][centroidIdx]
	}
	
	return totalDistance
}

// BatchDistances computes distances for multiple codes
func (dt *DistanceTableImpl) BatchDistances(codes [][]byte) []float32 {
	distances := make([]float32, len(codes))
	for i, code := range codes {
		distances[i] = dt.Distance(code)
	}
	return distances
}

// Size returns memory usage of the distance table
func (dt *DistanceTableImpl) Size() int64 {
	size := int64(len(dt.query) * 4) // Query vector
	for _, table := range dt.tables {
		size += int64(len(table) * 4) // Each table
	}
	return size
}