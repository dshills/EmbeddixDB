package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// OllamaEmbeddingEngine implements the EmbeddingEngine interface using Ollama's API
type OllamaEmbeddingEngine struct {
	config      *ai.ModelConfig
	httpClient  *http.Client
	modelInfo   ai.ModelInfo
	stats       *InferenceStats
	initialized bool
	mu          sync.RWMutex
}

// OllamaEmbedRequest represents the request payload for Ollama's embed API
type OllamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaEmbedResponse represents the response from Ollama's embed API
type OllamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// NewOllamaEmbeddingEngine creates a new Ollama embedding engine
func NewOllamaEmbeddingEngine(config *ai.ModelConfig) (*OllamaEmbeddingEngine, error) {
	if config == nil {
		return nil, NewEmbeddingError("NewOllamaEmbeddingEngine", "", ErrInvalidInput, "model config is nil", false)
	}

	// Set default Ollama endpoint if not specified
	if config.OllamaEndpoint == "" {
		config.OllamaEndpoint = "http://localhost:11434"
	}

	// Validate model name
	if config.Path == "" {
		return nil, NewEmbeddingError("NewOllamaEmbeddingEngine", "", ErrInvalidInput, "model name (Path) is required for Ollama", false)
	}

	engine := &OllamaEmbeddingEngine{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		modelInfo: ai.ModelInfo{
			Name:      config.Path,
			Dimension: config.Dimensions,
		},
		stats: &InferenceStats{
			RecentLatencies: make([]time.Duration, 0, 100),
		},
	}

	return engine, nil
}

// Embed generates embeddings for the given content
func (e *OllamaEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {
	e.mu.RLock()
	if !e.initialized {
		e.mu.RUnlock()
		return nil, NewEmbeddingError("Embed", e.config.Path, ErrModelNotLoaded, "engine not initialized", false)
	}
	e.mu.RUnlock()

	if len(content) == 0 {
		return [][]float32{}, nil
	}

	embeddings := make([][]float32, len(content))

	for i, text := range content {
		embedding, err := e.embedSingle(ctx, text)
		if err != nil {
			return nil, err
		}
		embeddings[i] = embedding
	}

	// Update stats
	e.updateStats(len(content), time.Now())

	return embeddings, nil
}

// EmbedBatch generates embeddings in batches
func (e *OllamaEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	if batchSize <= 0 {
		batchSize = e.config.BatchSize
	}
	if batchSize <= 0 {
		batchSize = 32 // Default batch size
	}

	e.mu.RLock()
	if !e.initialized {
		e.mu.RUnlock()
		return nil, NewEmbeddingError("EmbedBatch", e.config.Path, ErrModelNotLoaded, "engine not initialized", false)
	}
	e.mu.RUnlock()

	if len(content) == 0 {
		return [][]float32{}, nil
	}

	embeddings := make([][]float32, len(content))

	// Process in batches (Ollama doesn't support batch embedding natively, so we process sequentially)
	for i := 0; i < len(content); i += batchSize {
		end := i + batchSize
		if end > len(content) {
			end = len(content)
		}

		batch := content[i:end]
		for j, text := range batch {
			embedding, err := e.embedSingle(ctx, text)
			if err != nil {
				return nil, fmt.Errorf("failed to embed text at index %d: %w", i+j, err)
			}
			embeddings[i+j] = embedding
		}
	}

	// Update stats
	e.updateStats(len(content), time.Now())

	return embeddings, nil
}

// GetModelInfo returns information about the model
func (e *OllamaEmbeddingEngine) GetModelInfo() ai.ModelInfo {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.modelInfo
}

// Warm pre-initializes the engine
func (e *OllamaEmbeddingEngine) Warm(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.initialized {
		return nil
	}

	// Test connection and model availability
	testText := "test"
	_, err := e.embedSingle(ctx, testText)
	if err != nil {
		return NewEmbeddingError("Warm", e.config.Path, err, "failed to warm up Ollama engine", true)
	}

	e.initialized = true
	return nil
}

// Close releases resources
func (e *OllamaEmbeddingEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.initialized = false
	// HTTP client doesn't need explicit closing
	return nil
}

// embedSingle generates embedding for a single text
func (e *OllamaEmbeddingEngine) embedSingle(ctx context.Context, text string) ([]float32, error) {
	url := fmt.Sprintf("%s/api/embeddings", e.config.OllamaEndpoint)

	request := OllamaEmbedRequest{
		Model:  e.config.Path,
		Prompt: text,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, NewEmbeddingError("embedSingle", e.config.Path, err, "failed to marshal request", false)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, NewEmbeddingError("embedSingle", e.config.Path, err, "failed to create request", false)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, NewEmbeddingError("embedSingle", e.config.Path, err, "failed to send request", true)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewEmbeddingError("embedSingle", e.config.Path, fmt.Errorf("status %d: %s", resp.StatusCode, string(body)), "Ollama API error", resp.StatusCode >= 500)
	}

	var ollamaResp OllamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, NewEmbeddingError("embedSingle", e.config.Path, err, "failed to decode response", false)
	}

	// Normalize embeddings if configured
	if e.config.NormalizeEmbeddings {
		ollamaResp.Embedding = normalizeVector(ollamaResp.Embedding)
	}

	// Update model dimensions if not set
	if e.config.Dimensions == 0 && len(ollamaResp.Embedding) > 0 {
		e.mu.Lock()
		e.modelInfo.Dimension = len(ollamaResp.Embedding)
		e.config.Dimensions = len(ollamaResp.Embedding)
		e.mu.Unlock()
	}

	return ollamaResp.Embedding, nil
}

// updateStats updates inference statistics
func (e *OllamaEmbeddingEngine) updateStats(processed int, startTime time.Time) {
	e.stats.mutex.Lock()
	defer e.stats.mutex.Unlock()

	duration := time.Since(startTime)
	e.stats.TotalInferences++
	e.stats.TotalTokens += int64(processed)

	// Update recent latencies for P95 calculation
	e.stats.RecentLatencies = append(e.stats.RecentLatencies, duration)
	if len(e.stats.RecentLatencies) > 100 {
		e.stats.RecentLatencies = e.stats.RecentLatencies[1:]
	}

	// Calculate average latency
	if e.stats.TotalInferences > 0 {
		totalDuration := time.Duration(0)
		for _, lat := range e.stats.RecentLatencies {
			totalDuration += lat
		}
		e.stats.AverageLatency = totalDuration / time.Duration(len(e.stats.RecentLatencies))
	}

	// Update throughput
	if duration > 0 {
		e.stats.ThroughputTPS = float64(processed) / duration.Seconds()
	}
}

// normalizeVector normalizes a vector to unit length
func normalizeVector(v []float32) []float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}

	if sum == 0 {
		return v
	}

	norm := float32(1.0 / math.Sqrt(float64(sum)))
	normalized := make([]float32, len(v))
	for i, val := range v {
		normalized[i] = val * norm
	}

	return normalized
}
