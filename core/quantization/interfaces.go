package quantization

import (
	"context"
	"time"
)

// Quantizer defines the interface for vector quantization algorithms
type Quantizer interface {
	// Train builds the quantization model from training vectors
	Train(ctx context.Context, vectors [][]float32) error

	// Encode compresses a vector into a quantized code
	Encode(vector []float32) ([]byte, error)

	// Decode reconstructs an approximate vector from a quantized code
	Decode(code []byte) ([]float32, error)

	// Distance computes approximate distance between two quantized codes
	Distance(codeA, codeB []byte) (float32, error)

	// AsymmetricDistance computes distance between a quantized code and full vector
	AsymmetricDistance(code []byte, vector []float32) (float32, error)

	// MemoryReduction returns the compression ratio achieved
	MemoryReduction() float64

	// CodeSize returns the size in bytes of quantized codes
	CodeSize() int

	// IsTrained returns whether the quantizer has been trained
	IsTrained() bool

	// Config returns the quantizer configuration
	Config() QuantizerConfig
}

// QuantizerConfig holds common quantizer configuration
type QuantizerConfig struct {
	Type             QuantizerType `json:"type"`
	Dimension        int           `json:"dimension"`
	MemoryBudgetMB   int           `json:"memory_budget_mb"`
	TrainingTimeout  time.Duration `json:"training_timeout"`
	DistanceMetric   string        `json:"distance_metric"`
	EnableAsymmetric bool          `json:"enable_asymmetric"`
}

// QuantizerType represents different quantization algorithms
type QuantizerType string

const (
	ProductQuantization  QuantizerType = "product"
	ScalarQuantization   QuantizerType = "scalar"
	BinaryQuantization   QuantizerType = "binary"
	AdditiveQuantization QuantizerType = "additive"
)

// TrainingStats holds statistics from quantizer training
type TrainingStats struct {
	TrainingTime     time.Duration `json:"training_time"`
	ConvergenceIters int           `json:"convergence_iters"`
	FinalMSE         float64       `json:"final_mse"`
	MemoryReduction  float64       `json:"memory_reduction"`
	CodebookSize     int           `json:"codebook_size"`
	TrainingVectors  int           `json:"training_vectors"`
}

// QuantizationMetrics holds runtime performance metrics
type QuantizationMetrics struct {
	EncodeLatencyMs   float64 `json:"encode_latency_ms"`
	DecodeLatencyMs   float64 `json:"decode_latency_ms"`
	DistanceLatencyMs float64 `json:"distance_latency_ms"`
	CompressionRatio  float64 `json:"compression_ratio"`
	AccuracyLoss      float64 `json:"accuracy_loss"`
	ThroughputQPS     float64 `json:"throughput_qps"`
}

// QuantizerFactory creates quantizers based on configuration
type QuantizerFactory interface {
	CreateQuantizer(config QuantizerConfig) (Quantizer, error)
	SupportedTypes() []QuantizerType
	RecommendConfig(dimension int, memoryBudgetMB int) QuantizerConfig
}

// AdaptiveQuantizer automatically selects the best quantization strategy
type AdaptiveQuantizer interface {
	Quantizer

	// AutoTune finds optimal quantization parameters
	AutoTune(ctx context.Context, vectors [][]float32, targetAccuracy float64) error

	// Benchmark measures performance across different configurations
	Benchmark(ctx context.Context, vectors [][]float32) (map[QuantizerType]QuantizationMetrics, error)
}

// QuantizerPool manages multiple quantizers for different use cases
type QuantizerPool interface {
	// GetQuantizer retrieves a quantizer for the given dimension
	GetQuantizer(dimension int) (Quantizer, error)

	// RegisterQuantizer adds a quantizer to the pool
	RegisterQuantizer(dimension int, quantizer Quantizer) error

	// RemoveQuantizer removes a quantizer from the pool
	RemoveQuantizer(dimension int) error

	// ListQuantizers returns all registered quantizers
	ListQuantizers() map[int]Quantizer

	// Close closes all quantizers in the pool
	Close() error
}

// ValidationResult holds quantizer validation results
type ValidationResult struct {
	RecallAt1    float64 `json:"recall_at_1"`
	RecallAt10   float64 `json:"recall_at_10"`
	RecallAt100  float64 `json:"recall_at_100"`
	MSE          float64 `json:"mse"`
	MAE          float64 `json:"mae"`
	MaxError     float64 `json:"max_error"`
	DistanceCorr float64 `json:"distance_correlation"`
}

// QuantizerValidator validates quantizer accuracy
type QuantizerValidator interface {
	// ValidateRecall measures recall@k for quantized vs exact search
	ValidateRecall(quantizer Quantizer, queries, database [][]float32, k int) (ValidationResult, error)

	// ValidateDistances measures distance computation accuracy
	ValidateDistances(quantizer Quantizer, vectors [][]float32) (ValidationResult, error)

	// ValidateReconstruction measures vector reconstruction quality
	ValidateReconstruction(quantizer Quantizer, vectors [][]float32) (ValidationResult, error)
}

// Reranker performs full-precision reranking of quantized search results
type Reranker interface {
	// Rerank improves quantized search results using full precision
	Rerank(queryVector []float32, candidates []int, fullVectors [][]float32, k int) ([]int, []float32, error)

	// Config returns reranker configuration
	Config() RerankerConfig
}

// RerankerConfig configures the reranking process
type RerankerConfig struct {
	Enabled           bool    `json:"enabled"`
	CandidateMultiple int     `json:"candidate_multiple"` // Retrieve k*multiple candidates
	MaxCandidates     int     `json:"max_candidates"`
	MinImprovement    float64 `json:"min_improvement"` // Minimum recall improvement required
	CacheSize         int     `json:"cache_size"`
}

// DistanceTable precomputes distances for fast quantized search
type DistanceTable interface {
	// Precompute builds distance table for a query vector
	Precompute(queryVector []float32) error

	// Distance returns precomputed distance for a quantized code
	Distance(code []byte) float32

	// BatchDistances computes distances for multiple codes
	BatchDistances(codes [][]byte) []float32

	// Size returns memory usage of the distance table
	Size() int64
}

// ErrorCorrection provides error correction for quantized vectors
type ErrorCorrection interface {
	// CorrectErrors attempts to fix quantization errors
	CorrectErrors(original, quantized []float32) []float32

	// EstimateError estimates quantization error without full vector
	EstimateError(code []byte) float32
}
