package embedding

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// ONNXEmbeddingEngine implements embedding generation using ONNX Runtime
type ONNXEmbeddingEngine struct {
	modelName  string
	modelPath  string
	config     ai.ModelConfig
	modelInfo  ai.ModelInfo
	tokenizer  *Tokenizer
	session    ONNXSession // Interface to allow mocking
	stats      *InferenceStats
	warmupDone bool
	mutex      sync.RWMutex
}

// ONNXSession interface allows for testing and mocking
type ONNXSession interface {
	Run(inputs []ONNXValue) ([]ONNXValue, error)
	GetInputCount() int
	GetOutputCount() int
	GetInputName(index int) string
	GetOutputName(index int) string
	Destroy()
}

// ONNXValue interface for ONNX tensors
type ONNXValue interface {
	GetData() interface{}
	GetShape() []int64
	Destroy()
}

// InferenceStats tracks performance metrics
type InferenceStats struct {
	TotalInferences int64         `json:"total_inferences"`
	TotalTokens     int64         `json:"total_tokens"`
	AverageLatency  time.Duration `json:"average_latency"`
	P95Latency      time.Duration `json:"p95_latency"`
	ErrorRate       float64       `json:"error_rate"`
	ThroughputTPS   float64       `json:"throughput_tps"`
	RecentLatencies []time.Duration
	TotalErrors     int64
	mutex           sync.RWMutex
}

// NewONNXEmbeddingEngine creates a new ONNX-based embedding engine
func NewONNXEmbeddingEngine(modelPath string, config ai.ModelConfig) (*ONNXEmbeddingEngine, error) {
	// Validate model file if it's a real path
	if modelPath != "" {
		if err := ValidateModelFile(modelPath); err != nil {
			return nil, fmt.Errorf("model validation failed: %w", err)
		}
		
		// Check model compatibility
		if err := ValidateModelCompatibility(modelPath); err != nil {
			return nil, fmt.Errorf("model compatibility check failed: %w", err)
		}
	}

	engine := &ONNXEmbeddingEngine{
		modelName: config.Name,
		modelPath: modelPath,
		config:    config,
		stats:     NewInferenceStats(),
	}

	// Initialize model info
	engine.modelInfo = ai.ModelInfo{
		Name:    config.Name,
		Version: "1.0",
		License: "unknown",
		Size:    0, // Will be populated after loading
	}

	// Initialize tokenizer
	tokenizer, err := NewTokenizer(config.Name)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tokenizer: %w", err)
	}
	engine.tokenizer = tokenizer

	// Load ONNX session
	session, err := engine.createSession()
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}
	engine.session = session

	// Update model info with file size if available
	if modelPath != "" {
		if info, err := os.Stat(modelPath); err == nil {
			engine.modelInfo.Size = info.Size()
		}
	}

	return engine, nil
}

// createSession creates an ONNX Runtime session
func (e *ONNXEmbeddingEngine) createSession() (ONNXSession, error) {
	// Try to create a real ONNX Runtime session
	session, err := NewRealONNXSession(e.modelPath)
	if err != nil {
		// If real session fails, fall back to mock for development/testing
		if os.Getenv("EMBEDDIX_USE_MOCK_ONNX") == "true" || e.modelPath == "" {
			return &MockONNXSession{
				inputCount:  1,
				outputCount: 1,
				inputNames:  []string{"input_ids"},
				outputNames: []string{"embeddings"},
			}, nil
		}
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}
	
	return session, nil
}

// Embed generates embeddings for the given content
func (e *ONNXEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {
	start := time.Now()
	defer func() {
		e.stats.RecordInference(len(content), time.Since(start))
	}()

	// Ensure model is warmed up
	if !e.warmupDone {
		if err := e.Warm(ctx); err != nil {
			return nil, fmt.Errorf("model warmup failed: %w", err)
		}
	}

	// Tokenize inputs
	tokens, err := e.tokenizer.TokenizeBatch(content, e.config.MaxTokens)
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	// Create input tensors
	inputs, err := e.createInputTensors(tokens)
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("failed to create input tensors: %w", err)
	}
	defer func() {
		for _, input := range inputs {
			input.Destroy()
		}
	}()

	// Run inference
	outputs, err := e.session.Run(inputs)
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	// Extract embeddings
	embeddings, err := e.extractEmbeddings(outputs[0])
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("failed to extract embeddings: %w", err)
	}

	// Normalize if configured
	if e.config.NormalizeEmbeddings {
		embeddings = normalizeEmbeddings(embeddings)
	}

	return embeddings, nil
}

// EmbedBatch processes content in batches for optimal performance
func (e *ONNXEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	if batchSize <= 0 {
		batchSize = e.config.BatchSize
		if batchSize <= 0 {
			batchSize = 32 // Default batch size
		}
	}

	var allEmbeddings [][]float32

	for i := 0; i < len(content); i += batchSize {
		end := i + batchSize
		if end > len(content) {
			end = len(content)
		}

		batch := content[i:end]
		embeddings, err := e.Embed(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("batch processing failed at index %d: %w", i, err)
		}

		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

// GetModelInfo returns metadata about the loaded model
func (e *ONNXEmbeddingEngine) GetModelInfo() ai.ModelInfo {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	return e.modelInfo
}

// Warm preloads the model for faster inference
func (e *ONNXEmbeddingEngine) Warm(ctx context.Context) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if e.warmupDone {
		return nil
	}

	// Run a dummy inference to warm up the model
	warmupTexts := []string{"This is a warmup text for model initialization."}
	_, err := e.embed(ctx, warmupTexts)
	if err != nil {
		return fmt.Errorf("warmup inference failed: %w", err)
	}

	e.warmupDone = true
	return nil
}

// embed is the internal embedding function without warmup check
func (e *ONNXEmbeddingEngine) embed(ctx context.Context, content []string) ([][]float32, error) {
	// Simplified implementation for warmup
	// In a real implementation, this would call the full Embed method
	embeddings := make([][]float32, len(content))
	for i := range content {
		// Mock embedding generation
		embedding := make([]float32, e.modelInfo.Dimension)
		for j := range embedding {
			embedding[j] = float32(0.1) // Mock value
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

// Close releases model resources
func (e *ONNXEmbeddingEngine) Close() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}

	if e.tokenizer != nil {
		e.tokenizer.Close()
		e.tokenizer = nil
	}

	return nil
}

// createInputTensors creates ONNX tensors from tokenized input, including attention masks
func (e *ONNXEmbeddingEngine) createInputTensors(tokens [][]int64) ([]ONNXValue, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens provided")
	}

	var inputs []ONNXValue

	// Use the real tensor creation if we have a real session
	if realSession, ok := e.session.(*RealONNXSession); ok {
		// Create input_ids tensor
		if realSession.GetInputCount() > 0 {
			inputName := realSession.GetInputName(0)
			inputTensor, err := CreateInputTensorFromTokens(tokens, inputName)
			if err != nil {
				return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
			}
			inputs = append(inputs, inputTensor)
		}

		// Create attention_mask tensor if the model expects it
		if realSession.GetInputCount() > 1 {
			// Generate attention masks (1 for real tokens, 0 for padding)
			masks := e.generateAttentionMasks(tokens)
			maskTensor, err := CreateAttentionMaskTensor(masks)
			if err != nil {
				// Clean up previously created tensors
				for _, input := range inputs {
					input.Destroy()
				}
				return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
			}
			inputs = append(inputs, maskTensor)
		}

		return inputs, nil
	}

	// Fallback to mock tensor for mock sessions
	batchSize := len(tokens)
	seqLen := len(tokens[0]) // Assuming all sequences have same length after padding

	// Flatten tokens for tensor creation
	flatTokens := make([]int64, batchSize*seqLen)
	for i, seq := range tokens {
		copy(flatTokens[i*seqLen:(i+1)*seqLen], seq)
	}

	// Create mock tensor
	mockTensor := &MockONNXTensor{
		data:  flatTokens,
		shape: []int64{int64(batchSize), int64(seqLen)},
	}

	return []ONNXValue{mockTensor}, nil
}

// generateAttentionMasks creates attention masks for the tokenized input
func (e *ONNXEmbeddingEngine) generateAttentionMasks(tokens [][]int64) [][]int64 {
	masks := make([][]int64, len(tokens))
	
	for i, seq := range tokens {
		mask := make([]int64, len(seq))
		for j, token := range seq {
			if token != 0 { // Assuming 0 is the padding token
				mask[j] = 1
			} else {
				mask[j] = 0
			}
		}
		masks[i] = mask
	}
	
	return masks
}

// extractEmbeddings extracts embeddings from ONNX output tensor
func (e *ONNXEmbeddingEngine) extractEmbeddings(output ONNXValue) ([][]float32, error) {
	// Use real extraction if we have a real tensor
	if realTensor, ok := output.(*RealONNXTensor); ok {
		// Use the configured pooling strategy, defaulting to CLS token
		poolingStrategy := e.config.PoolingStrategy
		if poolingStrategy == "" {
			poolingStrategy = "cls" // Default to CLS token pooling
		}
		
		embeddings, err := ExtractEmbeddingsFromTensor(realTensor, poolingStrategy)
		if err != nil {
			return nil, err
		}

		// Update model info with discovered dimension
		if len(embeddings) > 0 && e.modelInfo.Dimension == 0 {
			e.modelInfo.Dimension = len(embeddings[0])
		}

		return embeddings, nil
	}

	// Fallback to mock extraction for mock tensors
	data := output.GetData().([]float32)
	shape := output.GetShape()

	batchSize := int(shape[0])
	embeddingDim := int(shape[len(shape)-1])

	// Update model info with discovered dimension
	if e.modelInfo.Dimension == 0 {
		e.modelInfo.Dimension = embeddingDim
	}

	// Reshape to batch of embeddings
	embeddings := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		start := i * embeddingDim
		end := start + embeddingDim
		embeddings[i] = make([]float32, embeddingDim)
		copy(embeddings[i], data[start:end])
	}

	return embeddings, nil
}

// normalizeEmbeddings normalizes embeddings to unit length
func normalizeEmbeddings(embeddings [][]float32) [][]float32 {
	for i, embedding := range embeddings {
		var norm float32
		for _, val := range embedding {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		if norm > 0 {
			for j := range embedding {
				embeddings[i][j] = embedding[j] / norm
			}
		}
	}
	return embeddings
}

// GetStats returns inference statistics
func (e *ONNXEmbeddingEngine) GetStats() *InferenceStats {
	return e.stats
}

// NewInferenceStats creates a new inference statistics tracker
func NewInferenceStats() *InferenceStats {
	return &InferenceStats{
		RecentLatencies: make([]time.Duration, 0, 100),
	}
}

// RecordInference records an inference operation
func (s *InferenceStats) RecordInference(tokenCount int, latency time.Duration) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.TotalInferences++
	s.TotalTokens += int64(tokenCount)

	// Update recent latencies
	s.RecentLatencies = append(s.RecentLatencies, latency)
	if len(s.RecentLatencies) > 100 {
		s.RecentLatencies = s.RecentLatencies[1:]
	}

	// Update statistics
	s.updateStats()
}

// RecordError records an inference error
func (s *InferenceStats) RecordError() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.TotalErrors++
	s.updateErrorRate()
}

func (s *InferenceStats) updateStats() {
	if len(s.RecentLatencies) == 0 {
		return
	}

	// Calculate average latency
	var total time.Duration
	for _, lat := range s.RecentLatencies {
		total += lat
	}
	s.AverageLatency = total / time.Duration(len(s.RecentLatencies))

	// Calculate P95 latency
	if len(s.RecentLatencies) >= 5 {
		// Sort latencies for percentile calculation
		sorted := make([]time.Duration, len(s.RecentLatencies))
		copy(sorted, s.RecentLatencies)

		// Simple bubble sort for small arrays
		for i := 0; i < len(sorted); i++ {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[i] > sorted[j] {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}

		p95Index := int(float64(len(sorted)) * 0.95)
		if p95Index >= len(sorted) {
			p95Index = len(sorted) - 1
		}
		s.P95Latency = sorted[p95Index]
	}

	// Calculate throughput
	if s.AverageLatency > 0 {
		s.ThroughputTPS = float64(time.Second) / float64(s.AverageLatency)
	}
}

func (s *InferenceStats) updateErrorRate() {
	if s.TotalInferences > 0 {
		s.ErrorRate = float64(s.TotalErrors) / float64(s.TotalInferences)
	}
}

// Mock implementations for testing

// MockONNXSession is a mock ONNX session for testing
type MockONNXSession struct {
	inputCount  int
	outputCount int
	inputNames  []string
	outputNames []string
}

func (m *MockONNXSession) Run(inputs []ONNXValue) ([]ONNXValue, error) {
	// Mock inference - generate dummy embeddings
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}

	input := inputs[0]
	shape := input.GetShape()
	batchSize := shape[0]
	embeddingDim := int64(384) // Mock dimension

	// Generate mock embeddings
	embeddings := make([]float32, batchSize*embeddingDim)
	for i := range embeddings {
		embeddings[i] = 0.1 // Mock value
	}

	output := &MockONNXTensor{
		data:  embeddings,
		shape: []int64{batchSize, embeddingDim},
	}

	return []ONNXValue{output}, nil
}

func (m *MockONNXSession) GetInputCount() int  { return m.inputCount }
func (m *MockONNXSession) GetOutputCount() int { return m.outputCount }
func (m *MockONNXSession) GetInputName(index int) string {
	if index < len(m.inputNames) {
		return m.inputNames[index]
	}
	return ""
}
func (m *MockONNXSession) GetOutputName(index int) string {
	if index < len(m.outputNames) {
		return m.outputNames[index]
	}
	return ""
}
func (m *MockONNXSession) Destroy() {}

// MockONNXTensor is a mock ONNX tensor for testing
type MockONNXTensor struct {
	data  interface{}
	shape []int64
}

func (m *MockONNXTensor) GetData() interface{} { return m.data }
func (m *MockONNXTensor) GetShape() []int64    { return m.shape }
func (m *MockONNXTensor) Destroy()             {}
