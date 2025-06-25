package ai

import (
	"context"
	"time"
)

// MockEmbeddingEngine is a mock implementation for testing
type MockEmbeddingEngine struct {
	config     ModelConfig
	modelInfo  ModelInfo
	warmupDone bool
}

// NewMockEmbeddingEngine creates a new mock embedding engine
func NewMockEmbeddingEngine(config ModelConfig) *MockEmbeddingEngine {
	return &MockEmbeddingEngine{
		config: config,
		modelInfo: ModelInfo{
			Name:       config.Name,
			Version:    "1.0",
			Dimension:  384, // Default dimension
			MaxTokens:  config.MaxTokens,
			Languages:  []string{"en"},
			Modalities: []string{"text"},
			License:    "test",
			Size:       10 * 1024 * 1024, // 10MB mock size
			Accuracy:   0.85,
			Speed:      1000,
		},
	}
}

// Embed generates mock embeddings for the given content
func (m *MockEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {
	if !m.warmupDone {
		if err := m.Warm(ctx); err != nil {
			return nil, err
		}
	}

	embeddings := make([][]float32, len(content))
	for i := range content {
		// Generate mock embedding
		embedding := make([]float32, m.modelInfo.Dimension)
		for j := range embedding {
			embedding[j] = 0.1 // Mock value
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

// EmbedBatch processes content in batches for optimal performance
func (m *MockEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	if batchSize <= 0 {
		batchSize = m.config.BatchSize
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
		embeddings, err := m.Embed(ctx, batch)
		if err != nil {
			return nil, err
		}

		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

// GetModelInfo returns metadata about the loaded model
func (m *MockEmbeddingEngine) GetModelInfo() ModelInfo {
	return m.modelInfo
}

// Warm preloads the model for faster inference
func (m *MockEmbeddingEngine) Warm(ctx context.Context) error {
	// Simulate warmup time
	time.Sleep(10 * time.Millisecond)
	m.warmupDone = true
	return nil
}

// Close releases model resources
func (m *MockEmbeddingEngine) Close() error {
	m.warmupDone = false
	return nil
}
