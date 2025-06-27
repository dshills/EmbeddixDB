package mcp

import (
	"context"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

// MockEmbeddingEngine implements ai.EmbeddingEngine for testing
type MockEmbeddingEngine struct {
	dimension int
}

func (m *MockEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {
	embeddings := make([][]float32, len(content))
	for i := range content {
		// Generate mock embedding based on content length
		embedding := make([]float32, m.dimension)
		for j := range embedding {
			embedding[j] = float32(len(content[i])) / float32(j+1) * 0.1
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

func (m *MockEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	return m.Embed(ctx, content)
}

func (m *MockEmbeddingEngine) GetModelInfo() ai.ModelInfo {
	return ai.ModelInfo{
		Name:      "mock-model",
		Version:   "1.0",
		Dimension: m.dimension,
	}
}

func (m *MockEmbeddingEngine) Warm(ctx context.Context) error {
	return nil
}

func (m *MockEmbeddingEngine) Close() error {
	return nil
}

func TestEmbeddingIntegration(t *testing.T) {
	// Create base vector store
	persistenceBackend := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	baseStore := core.NewVectorStore(persistenceBackend, indexFactory)
	
	// Create embedding store with mock engine
	mockEngine := &MockEmbeddingEngine{dimension: 384}
	embeddingStore := NewEmbeddingStore(baseStore, mockEngine)
	defer embeddingStore.Close()
	
	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "test_embeddings",
		Dimension: 384,
		Distance:  "cosine",
		IndexType: "flat",
	}
	
	if err := embeddingStore.CreateCollection(ctx, collection); err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	
	// Test EmbedText method
	testText := "This is a test sentence for embedding"
	embedding, err := embeddingStore.EmbedText(ctx, testText)
	if err != nil {
		t.Fatalf("Failed to embed text: %v", err)
	}
	
	if len(embedding) != 384 {
		t.Errorf("Expected embedding dimension 384, got %d", len(embedding))
	}
	
	// Test that handlers can use the embedding functionality
	handler := &AddVectorsHandler{store: embeddingStore}
	
	args := map[string]interface{}{
		"collection": "test_embeddings",
		"vectors": []interface{}{
			map[string]interface{}{
				"id":      "test1",
				"content": "Hello, this is a test document",
				"metadata": map[string]interface{}{
					"type": "test",
				},
			},
		},
	}
	
	result, err := handler.Execute(ctx, args)
	if err != nil {
		t.Fatalf("Failed to add vector with content: %v", err)
	}
	
	if result.IsError {
		t.Errorf("Expected success, got error: %v", result.Content)
	}
	
	// Verify the vector was added
	vec, err := embeddingStore.GetVector(ctx, "test_embeddings", "test1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}
	
	if vec.ID != "test1" {
		t.Errorf("Expected vector ID 'test1', got '%s'", vec.ID)
	}
	
	if len(vec.Values) != 384 {
		t.Errorf("Expected vector dimension 384, got %d", len(vec.Values))
	}
	
	// Test search with text query
	searchHandler := &SearchVectorsHandler{store: embeddingStore}
	searchArgs := map[string]interface{}{
		"collection": "test_embeddings",
		"query":      "test document",
		"limit":      10,
	}
	
	searchResult, err := searchHandler.Execute(ctx, searchArgs)
	if err != nil {
		t.Fatalf("Failed to search with text query: %v", err)
	}
	
	if searchResult.IsError {
		t.Errorf("Expected search success, got error: %v", searchResult.Content)
	}
}