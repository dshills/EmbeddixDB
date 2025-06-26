package quantization

import (
	"fmt"
	"math"
)

// DefaultQuantizerFactory implements the QuantizerFactory interface
type DefaultQuantizerFactory struct{}

// NewQuantizerFactory creates a new quantizer factory
func NewQuantizerFactory() QuantizerFactory {
	return &DefaultQuantizerFactory{}
}

// CreateQuantizer creates a quantizer based on configuration
func (f *DefaultQuantizerFactory) CreateQuantizer(config QuantizerConfig) (Quantizer, error) {
	switch config.Type {
	case ProductQuantization:
		return f.createProductQuantizer(config)
	case ScalarQuantization:
		return f.createScalarQuantizer(config)
	case BinaryQuantization:
		return nil, fmt.Errorf("binary quantization not yet implemented")
	case AdditiveQuantization:
		return nil, fmt.Errorf("additive quantization not yet implemented")
	default:
		return nil, fmt.Errorf("unsupported quantizer type: %s", config.Type)
	}
}

// createProductQuantizer creates a Product Quantizer
func (f *DefaultQuantizerFactory) createProductQuantizer(config QuantizerConfig) (Quantizer, error) {
	// Convert generic config to PQ-specific config
	pqConfig := DefaultProductQuantizerConfig(config.Dimension)
	pqConfig.DistanceMetric = config.DistanceMetric
	pqConfig.TrainingTimeout = config.TrainingTimeout
	
	// Adjust parameters based on memory budget
	if config.MemoryBudgetMB > 0 {
		pqConfig = f.optimizePQForMemoryBudget(pqConfig, config.MemoryBudgetMB)
	}
	
	return NewProductQuantizer(pqConfig)
}

// createScalarQuantizer creates a Scalar Quantizer
func (f *DefaultQuantizerFactory) createScalarQuantizer(config QuantizerConfig) (Quantizer, error) {
	// Convert generic config to SQ-specific config
	sqConfig := DefaultScalarQuantizerConfig(config.Dimension)
	sqConfig.DistanceMetric = config.DistanceMetric
	sqConfig.TrainingTimeout = config.TrainingTimeout
	
	// Adjust parameters based on memory budget
	if config.MemoryBudgetMB > 0 {
		sqConfig = f.optimizeSQForMemoryBudget(sqConfig, config.MemoryBudgetMB)
	}
	
	return NewScalarQuantizer(sqConfig)
}

// optimizePQForMemoryBudget adjusts PQ parameters to fit memory budget
func (f *DefaultQuantizerFactory) optimizePQForMemoryBudget(config ProductQuantizerConfig, budgetMB int) ProductQuantizerConfig {
	originalSize := config.Dimension * 4 // 4 bytes per float32
	targetSize := budgetMB * 1024 * 1024 / 1000 // Rough estimate per 1k vectors
	
	if targetSize <= 0 || originalSize <= targetSize {
		return config // No optimization needed
	}
	
	// Calculate optimal bits per subvector
	compressionRatio := float64(originalSize) / float64(targetSize)
	targetBits := int(math.Log2(compressionRatio) * float64(config.NumSubvectors))
	
	if targetBits < 4 {
		targetBits = 4
	} else if targetBits > 8 {
		targetBits = 8
	}
	
	config.BitsPerSubvector = targetBits
	return config
}

// optimizeSQForMemoryBudget adjusts SQ parameters to fit memory budget
func (f *DefaultQuantizerFactory) optimizeSQForMemoryBudget(config ScalarQuantizerConfig, budgetMB int) ScalarQuantizerConfig {
	originalSize := config.Dimension * 4 // 4 bytes per float32
	targetSize := budgetMB * 1024 * 1024 / 1000 // Rough estimate per 1k vectors
	
	if targetSize <= 0 || originalSize <= targetSize {
		return config // No optimization needed
	}
	
	// Calculate optimal bits per component
	compressionRatio := float64(originalSize) / float64(targetSize)
	targetBits := int(32.0 / compressionRatio)
	
	if targetBits < 4 {
		targetBits = 4
	} else if targetBits > 16 {
		targetBits = 16
	}
	
	config.BitsPerComponent = targetBits
	return config
}

// SupportedTypes returns the quantizer types supported by this factory
func (f *DefaultQuantizerFactory) SupportedTypes() []QuantizerType {
	return []QuantizerType{
		ProductQuantization,
		ScalarQuantization,
		// BinaryQuantization,   // Not yet implemented
		// AdditiveQuantization, // Not yet implemented
	}
}

// RecommendConfig recommends optimal quantizer configuration
func (f *DefaultQuantizerFactory) RecommendConfig(dimension int, memoryBudgetMB int) QuantizerConfig {
	// Default recommendations based on dimension and memory budget
	if dimension <= 64 {
		// Small dimensions: use scalar quantization
		return QuantizerConfig{
			Type:             ScalarQuantization,
			Dimension:        dimension,
			MemoryBudgetMB:   memoryBudgetMB,
			DistanceMetric:   "l2",
			EnableAsymmetric: true,
		}
	} else if dimension <= 256 {
		// Medium dimensions: use PQ with fewer subvectors
		return QuantizerConfig{
			Type:             ProductQuantization,
			Dimension:        dimension,
			MemoryBudgetMB:   memoryBudgetMB,
			DistanceMetric:   "l2",
			EnableAsymmetric: true,
		}
	} else {
		// Large dimensions: use PQ with more subvectors
		return QuantizerConfig{
			Type:             ProductQuantization,
			Dimension:        dimension,
			MemoryBudgetMB:   memoryBudgetMB,
			DistanceMetric:   "l2",
			EnableAsymmetric: true,
		}
	}
}

// DefaultQuantizerPool implements the QuantizerPool interface
type DefaultQuantizerPool struct {
	quantizers map[int]Quantizer
	factory    QuantizerFactory
}

// NewQuantizerPool creates a new quantizer pool
func NewQuantizerPool(factory QuantizerFactory) QuantizerPool {
	return &DefaultQuantizerPool{
		quantizers: make(map[int]Quantizer),
		factory:    factory,
	}
}

// GetQuantizer retrieves a quantizer for the given dimension
func (p *DefaultQuantizerPool) GetQuantizer(dimension int) (Quantizer, error) {
	if quantizer, exists := p.quantizers[dimension]; exists {
		return quantizer, nil
	}
	
	// Create a new quantizer with recommended config
	config := p.factory.RecommendConfig(dimension, 0)
	quantizer, err := p.factory.CreateQuantizer(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create quantizer for dimension %d: %w", dimension, err)
	}
	
	p.quantizers[dimension] = quantizer
	return quantizer, nil
}

// RegisterQuantizer adds a quantizer to the pool
func (p *DefaultQuantizerPool) RegisterQuantizer(dimension int, quantizer Quantizer) error {
	if quantizer == nil {
		return fmt.Errorf("quantizer cannot be nil")
	}
	
	if quantizer.Config().Dimension != dimension {
		return fmt.Errorf("quantizer dimension %d does not match requested dimension %d",
			quantizer.Config().Dimension, dimension)
	}
	
	p.quantizers[dimension] = quantizer
	return nil
}

// RemoveQuantizer removes a quantizer from the pool
func (p *DefaultQuantizerPool) RemoveQuantizer(dimension int) error {
	delete(p.quantizers, dimension)
	return nil
}

// ListQuantizers returns all registered quantizers
func (p *DefaultQuantizerPool) ListQuantizers() map[int]Quantizer {
	result := make(map[int]Quantizer)
	for dim, quantizer := range p.quantizers {
		result[dim] = quantizer
	}
	return result
}

// Close closes all quantizers in the pool
func (p *DefaultQuantizerPool) Close() error {
	p.quantizers = make(map[int]Quantizer)
	return nil
}

// ValidatorImpl implements QuantizerValidator
type ValidatorImpl struct{}

// NewValidator creates a new quantizer validator
func NewValidator() QuantizerValidator {
	return &ValidatorImpl{}
}

// ValidateRecall measures recall@k for quantized vs exact search
func (v *ValidatorImpl) ValidateRecall(quantizer Quantizer, queries, database [][]float32, k int) (ValidationResult, error) {
	if !quantizer.IsTrained() {
		return ValidationResult{}, fmt.Errorf("quantizer not trained")
	}
	
	if len(queries) == 0 || len(database) == 0 {
		return ValidationResult{}, fmt.Errorf("empty queries or database")
	}
	
	var totalRecall1, totalRecall10, totalRecall100 float64
	validQueries := 0
	
	for _, query := range queries {
		// Find exact k-NN
		exactResults := v.exactKNN(query, database, k)
		if len(exactResults) == 0 {
			continue
		}
		
		// Find quantized k-NN
		quantizedResults, err := v.quantizedKNN(query, database, quantizer, k)
		if err != nil {
			continue
		}
		
		// Calculate recall@1, @10, @100
		recall1 := v.calculateRecall(exactResults, quantizedResults, 1)
		recall10 := v.calculateRecall(exactResults, quantizedResults, min(10, k))
		recall100 := v.calculateRecall(exactResults, quantizedResults, min(100, k))
		
		totalRecall1 += recall1
		totalRecall10 += recall10
		totalRecall100 += recall100
		validQueries++
	}
	
	if validQueries == 0 {
		return ValidationResult{}, fmt.Errorf("no valid queries")
	}
	
	return ValidationResult{
		RecallAt1:   totalRecall1 / float64(validQueries),
		RecallAt10:  totalRecall10 / float64(validQueries),
		RecallAt100: totalRecall100 / float64(validQueries),
	}, nil
}

// ValidateDistances measures distance computation accuracy
func (v *ValidatorImpl) ValidateDistances(quantizer Quantizer, vectors [][]float32) (ValidationResult, error) {
	if !quantizer.IsTrained() {
		return ValidationResult{}, fmt.Errorf("quantizer not trained")
	}
	
	if len(vectors) < 2 {
		return ValidationResult{}, fmt.Errorf("need at least 2 vectors")
	}
	
	var totalError, maxError float64
	var correlationSum float64
	comparisons := 0
	
	// Sample pairs for validation
	sampleSize := min(1000, len(vectors)*len(vectors)/2)
	
	for i := 0; i < sampleSize; i++ {
		// Select random pair
		idx1 := i % len(vectors)
		idx2 := (i + 1) % len(vectors)
		if idx1 == idx2 {
			continue
		}
		
		vec1, vec2 := vectors[idx1], vectors[idx2]
		
		// Compute exact distance
		exactDist := v.exactDistance(vec1, vec2)
		
		// Encode vectors and compute quantized distance
		code1, err := quantizer.Encode(vec1)
		if err != nil {
			continue
		}
		
		code2, err := quantizer.Encode(vec2)
		if err != nil {
			continue
		}
		
		quantizedDist, err := quantizer.Distance(code1, code2)
		if err != nil {
			continue
		}
		
		// Calculate error
		error := math.Abs(float64(exactDist - quantizedDist))
		totalError += error
		
		if error > maxError {
			maxError = error
		}
		
		// For correlation (simplified)
		correlationSum += float64(exactDist * quantizedDist)
		comparisons++
	}
	
	if comparisons == 0 {
		return ValidationResult{}, fmt.Errorf("no valid comparisons")
	}
	
	return ValidationResult{
		MAE:          totalError / float64(comparisons),
		MaxError:     maxError,
		DistanceCorr: correlationSum / float64(comparisons), // Simplified correlation
	}, nil
}

// ValidateReconstruction measures vector reconstruction quality
func (v *ValidatorImpl) ValidateReconstruction(quantizer Quantizer, vectors [][]float32) (ValidationResult, error) {
	if !quantizer.IsTrained() {
		return ValidationResult{}, fmt.Errorf("quantizer not trained")
	}
	
	var totalMSE, totalMAE, maxError float64
	validVectors := 0
	
	for _, vec := range vectors {
		// Encode and decode
		code, err := quantizer.Encode(vec)
		if err != nil {
			continue
		}
		
		decoded, err := quantizer.Decode(code)
		if err != nil {
			continue
		}
		
		// Calculate reconstruction errors
		var mse, mae float64
		for i := range vec {
			error := float64(vec[i] - decoded[i])
			absError := math.Abs(error)
			
			mse += error * error
			mae += absError
			
			if absError > maxError {
				maxError = absError
			}
		}
		
		totalMSE += mse / float64(len(vec))
		totalMAE += mae / float64(len(vec))
		validVectors++
	}
	
	if validVectors == 0 {
		return ValidationResult{}, fmt.Errorf("no valid vectors")
	}
	
	return ValidationResult{
		MSE:      totalMSE / float64(validVectors),
		MAE:      totalMAE / float64(validVectors),
		MaxError: maxError,
	}, nil
}

// Helper functions

func (v *ValidatorImpl) exactKNN(query []float32, database [][]float32, k int) []int {
	type distanceIdx struct {
		distance float32
		index    int
	}
	
	distances := make([]distanceIdx, len(database))
	
	for i, vec := range database {
		distances[i] = distanceIdx{
			distance: v.exactDistance(query, vec),
			index:    i,
		}
	}
	
	// Sort by distance
	for i := 0; i < len(distances)-1; i++ {
		for j := i + 1; j < len(distances); j++ {
			if distances[i].distance > distances[j].distance {
				distances[i], distances[j] = distances[j], distances[i]
			}
		}
	}
	
	// Return top k indices
	result := make([]int, min(k, len(distances)))
	for i := range result {
		result[i] = distances[i].index
	}
	
	return result
}

func (v *ValidatorImpl) quantizedKNN(query []float32, database [][]float32, quantizer Quantizer, k int) ([]int, error) {
	queryCode, err := quantizer.Encode(query)
	if err != nil {
		return nil, err
	}
	
	type distanceIdx struct {
		distance float32
		index    int
	}
	
	distances := make([]distanceIdx, len(database))
	
	for i, vec := range database {
		dist, err := quantizer.AsymmetricDistance(queryCode, vec)
		if err != nil {
			return nil, err
		}
		
		distances[i] = distanceIdx{
			distance: dist,
			index:    i,
		}
	}
	
	// Sort by distance
	for i := 0; i < len(distances)-1; i++ {
		for j := i + 1; j < len(distances); j++ {
			if distances[i].distance > distances[j].distance {
				distances[i], distances[j] = distances[j], distances[i]
			}
		}
	}
	
	// Return top k indices
	result := make([]int, min(k, len(distances)))
	for i := range result {
		result[i] = distances[i].index
	}
	
	return result, nil
}

func (v *ValidatorImpl) calculateRecall(exact, quantized []int, k int) float64 {
	if k <= 0 || len(exact) == 0 {
		return 0
	}
	
	exactSet := make(map[int]bool)
	for i := 0; i < min(k, len(exact)); i++ {
		exactSet[exact[i]] = true
	}
	
	matches := 0
	for i := 0; i < min(k, len(quantized)); i++ {
		if exactSet[quantized[i]] {
			matches++
		}
	}
	
	return float64(matches) / float64(min(k, len(exact)))
}

func (v *ValidatorImpl) exactDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}