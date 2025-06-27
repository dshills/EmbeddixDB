package mcp

import (
	"context"
	"fmt"
	"sync"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/ai/embedding"
)

// EmbeddingStore extends VectorStore with automatic text embedding capabilities
type EmbeddingStore struct {
	core.VectorStore
	embeddingEngine ai.EmbeddingEngine
	mu              sync.RWMutex
}

// NewEmbeddingStore creates a new VectorStore with embedding capabilities
func NewEmbeddingStore(baseStore core.VectorStore, embeddingEngine ai.EmbeddingEngine) *EmbeddingStore {
	return &EmbeddingStore{
		VectorStore:     baseStore,
		embeddingEngine: embeddingEngine,
	}
}

// EmbedText converts text to vector embeddings using the configured embedding engine
func (es *EmbeddingStore) EmbedText(ctx context.Context, text string) ([]float32, error) {
	es.mu.RLock()
	engine := es.embeddingEngine
	es.mu.RUnlock()

	if engine == nil {
		return nil, fmt.Errorf("embedding engine not configured")
	}

	// Embed single text
	embeddings, err := engine.Embed(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings generated")
	}

	return embeddings[0], nil
}

// EmbedTexts converts multiple texts to vector embeddings
func (es *EmbeddingStore) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	es.mu.RLock()
	engine := es.embeddingEngine
	es.mu.RUnlock()

	if engine == nil {
		return nil, fmt.Errorf("embedding engine not configured")
	}

	return engine.Embed(ctx, texts)
}

// SetEmbeddingEngine updates the embedding engine
func (es *EmbeddingStore) SetEmbeddingEngine(engine ai.EmbeddingEngine) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.embeddingEngine = engine
}

// GetEmbeddingEngine returns the current embedding engine
func (es *EmbeddingStore) GetEmbeddingEngine() ai.EmbeddingEngine {
	es.mu.RLock()
	defer es.mu.RUnlock()
	return es.embeddingEngine
}

// Close closes both the vector store and embedding engine
func (es *EmbeddingStore) Close() error {
	es.mu.Lock()
	defer es.mu.Unlock()

	var errs []error

	// Close the embedding engine first
	if es.embeddingEngine != nil {
		if closer, ok := es.embeddingEngine.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				errs = append(errs, fmt.Errorf("failed to close embedding engine: %w", err))
			}
		}
	}

	// Close the underlying vector store
	if err := es.VectorStore.Close(); err != nil {
		errs = append(errs, fmt.Errorf("failed to close vector store: %w", err))
	}

	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}

	return nil
}

// CreateEmbeddingStore is a convenience function to create an embedding store with default ONNX engine
func CreateEmbeddingStore(baseStore core.VectorStore, modelPath string, config ai.ModelConfig) (*EmbeddingStore, error) {
	// Create ONNX embedding engine
	engine, err := embedding.NewONNXEmbeddingEngine(modelPath, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding engine: %w", err)
	}

	return NewEmbeddingStore(baseStore, engine), nil
}

