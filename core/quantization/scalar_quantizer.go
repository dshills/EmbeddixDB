package quantization

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// ScalarQuantizer implements scalar quantization for vector compression
type ScalarQuantizer struct {
	config        ScalarQuantizerConfig
	minValues     []float32 // Per-dimension minimums
	maxValues     []float32 // Per-dimension maximums
	scales        []float32 // Quantization scales
	trained       bool
	trainingStats TrainingStats
	mu            sync.RWMutex
}

// ScalarQuantizerConfig configures the Scalar Quantizer
type ScalarQuantizerConfig struct {
	BitsPerComponent int           `json:"bits_per_component"` // 4, 6, 8, 16 bits per component
	Dimension        int           `json:"dimension"`          // Vector dimension
	DistanceMetric   string        `json:"distance_metric"`    // "l2", "cosine", "dot"
	RangeClipPercent float64       `json:"range_clip_percent"` // Clip outliers (e.g., 0.01 = 1%)
	TrainingTimeout  time.Duration `json:"training_timeout"`
	NormalizeInput   bool          `json:"normalize_input"`    // Normalize vectors before quantization
}

// DefaultScalarQuantizerConfig returns sensible defaults
func DefaultScalarQuantizerConfig(dimension int) ScalarQuantizerConfig {
	return ScalarQuantizerConfig{
		BitsPerComponent: 8, // 256 levels per component
		Dimension:        dimension,
		DistanceMetric:   "l2",
		RangeClipPercent: 0.01, // Clip 1% outliers
		TrainingTimeout:  5 * time.Minute,
		NormalizeInput:   false,
	}
}

// NewScalarQuantizer creates a new Scalar Quantizer
func NewScalarQuantizer(config ScalarQuantizerConfig) (*ScalarQuantizer, error) {
	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	
	if config.BitsPerComponent < 1 || config.BitsPerComponent > 16 {
		return nil, fmt.Errorf("bits_per_component must be between 1 and 16")
	}
	
	if config.RangeClipPercent < 0 || config.RangeClipPercent >= 0.5 {
		return nil, fmt.Errorf("range_clip_percent must be between 0 and 0.5")
	}
	
	return &ScalarQuantizer{
		config:    config,
		minValues: make([]float32, config.Dimension),
		maxValues: make([]float32, config.Dimension),
		scales:    make([]float32, config.Dimension),
		trained:   false,
	}, nil
}

// Train builds the Scalar Quantizer from training data
func (sq *ScalarQuantizer) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}
	
	startTime := time.Now()
	
	// Validate input vectors
	for i, vec := range vectors {
		if len(vec) != sq.config.Dimension {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vec), sq.config.Dimension)
		}
	}
	
	// Normalize input vectors if requested
	trainingVectors := vectors
	if sq.config.NormalizeInput {
		trainingVectors = sq.normalizeVectors(vectors)
	}
	
	// Find min/max for each dimension
	err := sq.computeRanges(trainingVectors)
	if err != nil {
		return fmt.Errorf("failed to compute ranges: %w", err)
	}
	
	// Compute quantization scales
	sq.computeScales()
	
	// Calculate training statistics
	mse := sq.calculateMSE(trainingVectors)
	
	sq.mu.Lock()
	sq.trained = true
	sq.trainingStats = TrainingStats{
		TrainingTime:    time.Since(startTime),
		MemoryReduction: sq.calculateMemoryReduction(),
		FinalMSE:        mse,
		TrainingVectors: len(vectors),
	}
	sq.mu.Unlock()
	
	return nil
}

// computeRanges finds min/max values for each dimension with optional clipping
func (sq *ScalarQuantizer) computeRanges(vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}
	
	// Initialize with first vector
	copy(sq.minValues, vectors[0])
	copy(sq.maxValues, vectors[0])
	
	// Find global min/max
	for _, vec := range vectors {
		for d := 0; d < sq.config.Dimension; d++ {
			if vec[d] < sq.minValues[d] {
				sq.minValues[d] = vec[d]
			}
			if vec[d] > sq.maxValues[d] {
				sq.maxValues[d] = vec[d]
			}
		}
	}
	
	// Apply range clipping if enabled
	if sq.config.RangeClipPercent > 0 {
		sq.applyRangeClipping(vectors)
	}
	
	return nil
}

// applyRangeClipping clips outliers based on percentiles
func (sq *ScalarQuantizer) applyRangeClipping(vectors [][]float32) {
	for d := 0; d < sq.config.Dimension; d++ {
		// Collect all values for this dimension
		values := make([]float32, len(vectors))
		for i, vec := range vectors {
			values[i] = vec[d]
		}
		
		// Sort values to find percentiles
		sq.sortFloat32Slice(values)
		
		// Find clipping bounds
		clipIndex := int(float64(len(values)) * sq.config.RangeClipPercent)
		if clipIndex < 1 {
			clipIndex = 1
		}
		
		sq.minValues[d] = values[clipIndex]
		sq.maxValues[d] = values[len(values)-1-clipIndex]
	}
}

// sortFloat32Slice sorts a slice of float32 (simple bubble sort for small slices)
func (sq *ScalarQuantizer) sortFloat32Slice(values []float32) {
	n := len(values)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if values[j] > values[j+1] {
				values[j], values[j+1] = values[j+1], values[j]
			}
		}
	}
}

// computeScales calculates quantization scales for each dimension
func (sq *ScalarQuantizer) computeScales() {
	maxLevel := float32(int(1<<uint(sq.config.BitsPerComponent)) - 1)
	
	for d := 0; d < sq.config.Dimension; d++ {
		rangeSize := sq.maxValues[d] - sq.minValues[d]
		if rangeSize > 0 {
			sq.scales[d] = maxLevel / rangeSize
		} else {
			sq.scales[d] = 1.0 // Avoid division by zero
		}
	}
}

// normalizeVectors normalizes input vectors to unit length
func (sq *ScalarQuantizer) normalizeVectors(vectors [][]float32) [][]float32 {
	normalized := make([][]float32, len(vectors))
	
	for i, vec := range vectors {
		normalized[i] = make([]float32, len(vec))
		
		// Calculate norm
		var norm float32
		for _, val := range vec {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))
		
		// Normalize
		if norm > 0 {
			for j, val := range vec {
				normalized[i][j] = val / norm
			}
		} else {
			copy(normalized[i], vec) // Keep zero vector as is
		}
	}
	
	return normalized
}

// Encode compresses a vector into a quantized code
func (sq *ScalarQuantizer) Encode(vector []float32) ([]byte, error) {
	if !sq.IsTrained() {
		return nil, fmt.Errorf("quantizer not trained")
	}
	
	if len(vector) != sq.config.Dimension {
		return nil, fmt.Errorf("vector dimension %d does not match expected %d", len(vector), sq.config.Dimension)
	}
	
	// Normalize if required
	inputVector := vector
	if sq.config.NormalizeInput {
		inputVector = sq.normalizeVector(vector)
	}
	
	// Calculate code size in bytes
	totalBits := sq.config.Dimension * sq.config.BitsPerComponent
	codeSize := (totalBits + 7) / 8
	code := make([]byte, codeSize)
	
	// Quantize each component
	bitOffset := 0
	for d := 0; d < sq.config.Dimension; d++ {
		quantized := sq.quantizeComponent(inputVector[d], d)
		sq.packBits(code, bitOffset, quantized, sq.config.BitsPerComponent)
		bitOffset += sq.config.BitsPerComponent
	}
	
	return code, nil
}

// Decode reconstructs an approximate vector from a quantized code
func (sq *ScalarQuantizer) Decode(code []byte) ([]float32, error) {
	if !sq.IsTrained() {
		return nil, fmt.Errorf("quantizer not trained")
	}
	
	expectedSize := (sq.config.Dimension*sq.config.BitsPerComponent + 7) / 8
	if len(code) != expectedSize {
		return nil, fmt.Errorf("code size %d does not match expected %d", len(code), expectedSize)
	}
	
	vector := make([]float32, sq.config.Dimension)
	bitOffset := 0
	
	for d := 0; d < sq.config.Dimension; d++ {
		quantized := sq.unpackBits(code, bitOffset, sq.config.BitsPerComponent)
		vector[d] = sq.dequantizeComponent(quantized, d)
		bitOffset += sq.config.BitsPerComponent
	}
	
	return vector, nil
}

// quantizeComponent quantizes a single component
func (sq *ScalarQuantizer) quantizeComponent(value float32, dimension int) int {
	// Clamp to range
	if value < sq.minValues[dimension] {
		value = sq.minValues[dimension]
	} else if value > sq.maxValues[dimension] {
		value = sq.maxValues[dimension]
	}
	
	// Scale to quantization range
	scaled := (value - sq.minValues[dimension]) * sq.scales[dimension]
	quantized := int(scaled + 0.5) // Round to nearest integer
	
	// Clamp to valid range
	maxLevel := (1 << uint(sq.config.BitsPerComponent)) - 1
	if quantized > maxLevel {
		quantized = maxLevel
	}
	
	return quantized
}

// dequantizeComponent dequantizes a single component
func (sq *ScalarQuantizer) dequantizeComponent(quantized int, dimension int) float32 {
	// Convert back to original range
	scaled := float32(quantized) / sq.scales[dimension]
	return scaled + sq.minValues[dimension]
}

// normalizeVector normalizes a single vector
func (sq *ScalarQuantizer) normalizeVector(vector []float32) []float32 {
	normalized := make([]float32, len(vector))
	
	// Calculate norm
	var norm float32
	for _, val := range vector {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	
	// Normalize
	if norm > 0 {
		for i, val := range vector {
			normalized[i] = val / norm
		}
	} else {
		copy(normalized, vector)
	}
	
	return normalized
}

// Distance computes approximate distance between two quantized codes
func (sq *ScalarQuantizer) Distance(codeA, codeB []byte) (float32, error) {
	if !sq.IsTrained() {
		return 0, fmt.Errorf("quantizer not trained")
	}
	
	if len(codeA) != len(codeB) {
		return 0, fmt.Errorf("code lengths do not match")
	}
	
	// Decode both vectors and compute distance
	vectorA, err := sq.Decode(codeA)
	if err != nil {
		return 0, fmt.Errorf("failed to decode codeA: %w", err)
	}
	
	vectorB, err := sq.Decode(codeB)
	if err != nil {
		return 0, fmt.Errorf("failed to decode codeB: %w", err)
	}
	
	return sq.computeDistance(vectorA, vectorB), nil
}

// AsymmetricDistance computes distance between a quantized code and full vector
func (sq *ScalarQuantizer) AsymmetricDistance(code []byte, vector []float32) (float32, error) {
	if !sq.IsTrained() {
		return 0, fmt.Errorf("quantizer not trained")
	}
	
	if len(vector) != sq.config.Dimension {
		return 0, fmt.Errorf("vector dimension mismatch")
	}
	
	// Decode quantized vector and compute distance
	quantizedVector, err := sq.Decode(code)
	if err != nil {
		return 0, fmt.Errorf("failed to decode code: %w", err)
	}
	
	inputVector := vector
	if sq.config.NormalizeInput {
		inputVector = sq.normalizeVector(vector)
	}
	
	return sq.computeDistance(quantizedVector, inputVector), nil
}

// computeDistance computes distance between two vectors
func (sq *ScalarQuantizer) computeDistance(a, b []float32) float32 {
	switch sq.config.DistanceMetric {
	case "l2":
		return sq.l2Distance(a, b)
	case "cosine":
		return sq.cosineDistance(a, b)
	case "dot":
		return sq.dotDistance(a, b)
	default:
		return sq.l2Distance(a, b)
	}
}

// l2Distance computes L2 (Euclidean) distance
func (sq *ScalarQuantizer) l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // Return squared distance for efficiency
}

// cosineDistance computes cosine distance (1 - cosine similarity)
func (sq *ScalarQuantizer) cosineDistance(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 1.0
	}
	
	cosine := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - cosine
}

// dotDistance computes negative dot product
func (sq *ScalarQuantizer) dotDistance(a, b []float32) float32 {
	var dotProduct float32
	for i := range a {
		dotProduct += a[i] * b[i]
	}
	return -dotProduct
}

// packBits packs bits into a byte array
func (sq *ScalarQuantizer) packBits(data []byte, bitOffset, value, numBits int) {
	for i := 0; i < numBits; i++ {
		if value&(1<<i) != 0 {
			byteIdx := (bitOffset + i) / 8
			bitIdx := (bitOffset + i) % 8
			data[byteIdx] |= 1 << bitIdx
		}
	}
}

// unpackBits extracts bits from a byte array
func (sq *ScalarQuantizer) unpackBits(data []byte, bitOffset, numBits int) int {
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
func (sq *ScalarQuantizer) calculateMemoryReduction() float64 {
	originalSize := sq.config.Dimension * 4 // 4 bytes per float32
	compressedSize := (sq.config.Dimension*sq.config.BitsPerComponent + 7) / 8
	return float64(originalSize) / float64(compressedSize)
}

// calculateMSE calculates mean squared error for training data
func (sq *ScalarQuantizer) calculateMSE(vectors [][]float32) float64 {
	var totalMSE float64
	
	for _, vec := range vectors {
		code, err := sq.Encode(vec)
		if err != nil {
			continue
		}
		
		decoded, err := sq.Decode(code)
		if err != nil {
			continue
		}
		
		var mse float64
		for d := 0; d < sq.config.Dimension; d++ {
			diff := float64(vec[d] - decoded[d])
			mse += diff * diff
		}
		
		totalMSE += mse / float64(sq.config.Dimension)
	}
	
	return totalMSE / float64(len(vectors))
}

// Interface implementations

func (sq *ScalarQuantizer) MemoryReduction() float64 {
	return sq.calculateMemoryReduction()
}

func (sq *ScalarQuantizer) CodeSize() int {
	return (sq.config.Dimension*sq.config.BitsPerComponent + 7) / 8
}

func (sq *ScalarQuantizer) IsTrained() bool {
	sq.mu.RLock()
	defer sq.mu.RUnlock()
	return sq.trained
}

func (sq *ScalarQuantizer) Config() QuantizerConfig {
	return QuantizerConfig{
		Type:             ScalarQuantization,
		Dimension:        sq.config.Dimension,
		MemoryBudgetMB:   0,
		TrainingTimeout:  sq.config.TrainingTimeout,
		DistanceMetric:   sq.config.DistanceMetric,
		EnableAsymmetric: true,
	}
}

// GetTrainingStats returns training statistics
func (sq *ScalarQuantizer) GetTrainingStats() TrainingStats {
	sq.mu.RLock()
	defer sq.mu.RUnlock()
	return sq.trainingStats
}

// GetRanges returns the quantization ranges (for serialization)
func (sq *ScalarQuantizer) GetRanges() ([]float32, []float32, []float32) {
	sq.mu.RLock()
	defer sq.mu.RUnlock()
	
	minVals := make([]float32, len(sq.minValues))
	maxVals := make([]float32, len(sq.maxValues))
	scales := make([]float32, len(sq.scales))
	
	copy(minVals, sq.minValues)
	copy(maxVals, sq.maxValues)
	copy(scales, sq.scales)
	
	return minVals, maxVals, scales
}

// SetRanges sets the quantization ranges (for deserialization)
func (sq *ScalarQuantizer) SetRanges(minValues, maxValues, scales []float32) error {
	if len(minValues) != sq.config.Dimension || 
	   len(maxValues) != sq.config.Dimension || 
	   len(scales) != sq.config.Dimension {
		return fmt.Errorf("range arrays must match dimension %d", sq.config.Dimension)
	}
	
	sq.mu.Lock()
	copy(sq.minValues, minValues)
	copy(sq.maxValues, maxValues)
	copy(sq.scales, scales)
	sq.trained = true
	sq.mu.Unlock()
	
	return nil
}