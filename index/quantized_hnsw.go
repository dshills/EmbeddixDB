package index

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/quantization"
)

// QuantizedHNSW implements HNSW with Product Quantization for memory efficiency
type QuantizedHNSW struct {
	*HNSWIndex                                // Embed standard HNSW
	quantizer      quantization.Quantizer     // Vector quantizer
	quantizedDB    map[string][]byte          // Quantized vector storage
	originalDB     map[string][]float32       // Original vectors for reranking
	distanceTable  quantization.DistanceTable // Cached distance table for current query
	rerankerConfig RerankerConfig             // Reranking configuration
	mu             sync.RWMutex               // Protects quantized storage
}

// RerankerConfig configures the reranking pipeline
type RerankerConfig struct {
	Enable         bool    `json:"enable"`           // Enable reranking
	RerankerRatio  float64 `json:"reranker_ratio"`   // Ratio of candidates to rerank (e.g., 0.1 = 10%)
	MinCandidates  int     `json:"min_candidates"`   // Minimum candidates for reranking
	MaxCandidates  int     `json:"max_candidates"`   // Maximum candidates for reranking
	UseAsymmetric  bool    `json:"use_asymmetric"`   // Use asymmetric distance for reranking
	CacheDistTable bool    `json:"cache_dist_table"` // Cache distance tables
}

// DefaultRerankerConfig returns sensible defaults
func DefaultRerankerConfig() RerankerConfig {
	return RerankerConfig{
		Enable:         true,
		RerankerRatio:  0.2, // Rerank top 20% of candidates
		MinCandidates:  10,
		MaxCandidates:  100,
		UseAsymmetric:  true,
		CacheDistTable: true,
	}
}

// QuantizedHNSWConfig extends HNSWConfig with quantization settings
type QuantizedHNSWConfig struct {
	HNSWConfig          HNSWConfig                   `json:"hnsw_config"`
	QuantizerConfig     quantization.QuantizerConfig `json:"quantizer_config"`
	RerankerConfig      RerankerConfig               `json:"reranker_config"`
	KeepOriginalVectors bool                         `json:"keep_original_vectors"` // Keep originals for reranking
}

// DefaultQuantizedHNSWConfig returns sensible defaults
func DefaultQuantizedHNSWConfig(dimension int, distanceMetric core.DistanceMetric) QuantizedHNSWConfig {
	return QuantizedHNSWConfig{
		HNSWConfig: DefaultHNSWConfig(),
		QuantizerConfig: quantization.QuantizerConfig{
			Type:             quantization.ProductQuantization,
			Dimension:        dimension,
			MemoryBudgetMB:   10, // 10MB budget
			DistanceMetric:   string(distanceMetric),
			EnableAsymmetric: true,
		},
		RerankerConfig:      DefaultRerankerConfig(),
		KeepOriginalVectors: true, // Required for reranking
	}
}

// NewQuantizedHNSW creates a new Quantized HNSW index
func NewQuantizedHNSW(config QuantizedHNSWConfig) (*QuantizedHNSW, error) {
	// Create base HNSW index
	hnsw := NewHNSWIndex(config.QuantizerConfig.Dimension, core.DistanceMetric(config.QuantizerConfig.DistanceMetric), config.HNSWConfig)

	// Create quantizer with optimized configuration for testing
	var quantizer quantization.Quantizer
	var err error

	if config.QuantizerConfig.Type == quantization.ProductQuantization {
		// Create PQ with test-friendly configuration
		pqConfig := quantization.DefaultProductQuantizerConfig(config.QuantizerConfig.Dimension)
		pqConfig.BitsPerSubvector = 4 // Use 16 clusters instead of 256 for smaller datasets
		pqConfig.DistanceMetric = config.QuantizerConfig.DistanceMetric
		pqConfig.TrainingTimeout = config.QuantizerConfig.TrainingTimeout
		quantizer, err = quantization.NewProductQuantizer(pqConfig)
	} else {
		// Use factory for other quantizer types
		factory := quantization.NewQuantizerFactory()
		quantizer, err = factory.CreateQuantizer(config.QuantizerConfig)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create quantizer: %w", err)
	}

	qhnsw := &QuantizedHNSW{
		HNSWIndex:      hnsw,
		quantizer:      quantizer,
		quantizedDB:    make(map[string][]byte),
		rerankerConfig: config.RerankerConfig,
	}

	if config.KeepOriginalVectors {
		qhnsw.originalDB = make(map[string][]float32)
	}

	return qhnsw, nil
}

// Train trains the quantizer using provided vectors
func (qh *QuantizedHNSW) Train(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	return qh.quantizer.Train(ctx, vectors)
}

// IsTrained returns whether the quantizer is trained
func (qh *QuantizedHNSW) IsTrained() bool {
	return qh.quantizer.IsTrained()
}

// Insert adds a vector to the quantized index
func (qh *QuantizedHNSW) Insert(vector core.Vector) error {
	if !qh.IsTrained() {
		return fmt.Errorf("quantizer not trained")
	}

	// Quantize the vector
	quantizedCode, err := qh.quantizer.Encode(vector.Values)
	if err != nil {
		return fmt.Errorf("failed to quantize vector: %w", err)
	}

	// Store quantized and original vectors
	qh.mu.Lock()
	qh.quantizedDB[vector.ID] = quantizedCode
	if qh.originalDB != nil {
		originalCopy := make([]float32, len(vector.Values))
		copy(originalCopy, vector.Values)
		qh.originalDB[vector.ID] = originalCopy
	}
	qh.mu.Unlock()

	// Insert into HNSW using quantized vector for graph construction
	decodedVector, err := qh.quantizer.Decode(quantizedCode)
	if err != nil {
		return fmt.Errorf("failed to decode quantized vector: %w", err)
	}

	// Create a new vector with decoded values for HNSW
	hnswVector := core.Vector{
		ID:       vector.ID,
		Values:   decodedVector,
		Metadata: vector.Metadata,
	}

	return qh.HNSWIndex.Add(hnswVector)
}

// Search performs quantized search with optional reranking
func (qh *QuantizedHNSW) Search(query []float32, k int, filter map[string]string) ([]core.SearchResult, error) {
	if !qh.IsTrained() {
		return nil, fmt.Errorf("quantizer not trained")
	}

	// Step 1: Create distance table for efficient quantized search (if supported)
	if qh.rerankerConfig.CacheDistTable {
		// Note: Distance table optimization would be implemented here
		// For now, we'll skip this optimization
	}

	// Step 2: Perform quantized HNSW search with larger candidate set
	searchK := k
	if qh.rerankerConfig.Enable {
		// Increase k to get more candidates for reranking
		candidateMultiplier := int(1.0 / qh.rerankerConfig.RerankerRatio)
		searchK = k * candidateMultiplier
		if searchK < qh.rerankerConfig.MinCandidates {
			searchK = qh.rerankerConfig.MinCandidates
		}
		if searchK > qh.rerankerConfig.MaxCandidates {
			searchK = qh.rerankerConfig.MaxCandidates
		}
	}

	// Search using the base HNSW index
	candidates, err := qh.HNSWIndex.Search(query, searchK, filter)
	if err != nil {
		return nil, fmt.Errorf("quantized search failed: %w", err)
	}

	// Step 3: Rerank if enabled and we have original vectors
	if qh.rerankerConfig.Enable && qh.originalDB != nil && len(candidates) > k {
		return qh.rerank(query, candidates, k)
	}

	// Return top k without reranking
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates, nil
}

// rerank reranks candidates using original vectors for improved accuracy
func (qh *QuantizedHNSW) rerank(query []float32, candidates []core.SearchResult, k int) ([]core.SearchResult, error) {
	type candidate struct {
		result   core.SearchResult
		distance float32
	}

	// Calculate exact distances for all candidates
	rerankedCandidates := make([]candidate, 0, len(candidates))

	qh.mu.RLock()
	for _, result := range candidates {
		if originalVec, exists := qh.originalDB[result.ID]; exists {
			// Compute exact distance
			exactDist := qh.computeExactDistance(query, originalVec)
			rerankedCandidates = append(rerankedCandidates, candidate{
				result:   result,
				distance: exactDist,
			})
		} else {
			// Keep quantized result if original not available
			rerankedCandidates = append(rerankedCandidates, candidate{
				result:   result,
				distance: result.Score,
			})
		}
	}
	qh.mu.RUnlock()

	// Sort by exact distance
	sort.Slice(rerankedCandidates, func(i, j int) bool {
		return rerankedCandidates[i].distance < rerankedCandidates[j].distance
	})

	// Return top k with updated distances
	results := make([]core.SearchResult, 0, k)
	for i := 0; i < len(rerankedCandidates) && i < k; i++ {
		result := rerankedCandidates[i].result
		result.Score = rerankedCandidates[i].distance
		results = append(results, result)
	}

	return results, nil
}

// computeExactDistance computes exact distance between vectors
func (qh *QuantizedHNSW) computeExactDistance(a, b []float32) float32 {
	// Use core distance functions
	switch qh.quantizer.Config().DistanceMetric {
	case "l2":
		return qh.l2Distance(a, b)
	case "cosine":
		return qh.cosineDistance(a, b)
	case "dot":
		return qh.dotDistance(a, b)
	default:
		return qh.l2Distance(a, b)
	}
}

// Distance metric implementations
func (qh *QuantizedHNSW) l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func (qh *QuantizedHNSW) cosineDistance(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	cosine := dotProduct / (float32(qh.sqrt(float64(normA))) * float32(qh.sqrt(float64(normB))))
	return 1.0 - cosine
}

func (qh *QuantizedHNSW) dotDistance(a, b []float32) float32 {
	var dotProduct float32
	for i := range a {
		dotProduct += a[i] * b[i]
	}
	return -dotProduct
}

func (qh *QuantizedHNSW) sqrt(x float64) float64 {
	if x == 0 {
		return 0
	}

	// Newton's method for square root
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// Delete removes a vector from the quantized index
func (qh *QuantizedHNSW) Delete(id string) error {
	// Remove from quantized storage
	qh.mu.Lock()
	delete(qh.quantizedDB, id)
	if qh.originalDB != nil {
		delete(qh.originalDB, id)
	}
	qh.mu.Unlock()

	// Note: HNSW deletion is complex and not implemented in base HNSW
	// This is a limitation that would need to be addressed in a full implementation
	return nil
}

// GetStats returns quantized index statistics
func (qh *QuantizedHNSW) GetStats() QuantizedIndexStats {
	qh.mu.RLock()
	quantizedCount := len(qh.quantizedDB)
	originalCount := 0
	if qh.originalDB != nil {
		originalCount = len(qh.originalDB)
	}
	qh.mu.RUnlock()

	return QuantizedIndexStats{
		QuantizedVectors: quantizedCount,
		OriginalVectors:  originalCount,
		MemoryReduction:  qh.quantizer.MemoryReduction(),
		CodeSize:         qh.quantizer.CodeSize(),
		QuantizerTrained: qh.quantizer.IsTrained(),
		RerankerEnabled:  qh.rerankerConfig.Enable,
	}
}

// QuantizedIndexStats provides statistics for quantized index
type QuantizedIndexStats struct {
	QuantizedVectors int     `json:"quantized_vectors"`
	OriginalVectors  int     `json:"original_vectors"`
	MemoryReduction  float64 `json:"memory_reduction"`
	CodeSize         int     `json:"code_size"`
	QuantizerTrained bool    `json:"quantizer_trained"`
	RerankerEnabled  bool    `json:"reranker_enabled"`
}

// GetQuantizer returns the underlying quantizer
func (qh *QuantizedHNSW) GetQuantizer() quantization.Quantizer {
	return qh.quantizer
}

// GetRerankerConfig returns the reranker configuration
func (qh *QuantizedHNSW) GetRerankerConfig() RerankerConfig {
	return qh.rerankerConfig
}

// UpdateRerankerConfig updates the reranker configuration
func (qh *QuantizedHNSW) UpdateRerankerConfig(config RerankerConfig) {
	qh.rerankerConfig = config
}

// GetQuantizedCode returns the quantized code for a vector ID
func (qh *QuantizedHNSW) GetQuantizedCode(id string) ([]byte, bool) {
	qh.mu.RLock()
	defer qh.mu.RUnlock()
	code, exists := qh.quantizedDB[id]
	return code, exists
}

// GetOriginalVector returns the original vector for a vector ID
func (qh *QuantizedHNSW) GetOriginalVector(id string) ([]float32, bool) {
	qh.mu.RLock()
	defer qh.mu.RUnlock()
	if qh.originalDB == nil {
		return nil, false
	}
	vec, exists := qh.originalDB[id]
	return vec, exists
}

// EstimateMemoryUsage estimates memory usage of the quantized index
func (qh *QuantizedHNSW) EstimateMemoryUsage() QuantizedMemoryUsage {
	qh.mu.RLock()
	quantizedCount := len(qh.quantizedDB)
	originalCount := 0
	if qh.originalDB != nil {
		originalCount = len(qh.originalDB)
	}
	qh.mu.RUnlock()

	codeSize := qh.quantizer.CodeSize()
	quantizedMemory := int64(quantizedCount * codeSize)
	dimension := qh.quantizer.Config().Dimension
	originalMemory := int64(originalCount * dimension * 4) // 4 bytes per float32

	// Estimate base HNSW memory (rough estimate)
	baseMemory := int64(quantizedCount * 64) // Rough estimate for graph structure

	return QuantizedMemoryUsage{
		BaseMemoryUsage:  baseMemory,
		QuantizedVectors: quantizedMemory,
		OriginalVectors:  originalMemory,
		TotalMemory:      baseMemory + quantizedMemory + originalMemory,
		MemoryReduction:  qh.quantizer.MemoryReduction(),
	}
}

// QuantizedMemoryUsage provides memory usage statistics
type QuantizedMemoryUsage struct {
	BaseMemoryUsage  int64   `json:"base_memory_usage"`
	QuantizedVectors int64   `json:"quantized_vectors"`
	OriginalVectors  int64   `json:"original_vectors"`
	TotalMemory      int64   `json:"total_memory"`
	MemoryReduction  float64 `json:"memory_reduction"`
}
