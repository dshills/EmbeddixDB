package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

// Mock implementations for testing

type MockContentAnalyzer struct{}

func (m *MockContentAnalyzer) AnalyzeContent(ctx context.Context, content string) (ContentInsights, error) {
	return ContentInsights{
		Language: LanguageInfo{
			Code:       "en",
			Name:       "English",
			Confidence: 0.95,
		},
		WordCount: len(strings.Fields(content)),
		Summary:   "Mock analysis summary",
	}, nil
}

func (m *MockContentAnalyzer) AnalyzeBatch(ctx context.Context, content []string) ([]ContentInsights, error) {
	insights := make([]ContentInsights, len(content))
	for i, text := range content {
		insight, _ := m.AnalyzeContent(ctx, text)
		insights[i] = insight
	}
	return insights, nil
}

func (m *MockContentAnalyzer) ExtractEntities(ctx context.Context, content string) ([]Entity, error) {
	return []Entity{}, nil
}

func (m *MockContentAnalyzer) DetectLanguage(ctx context.Context, content string) (LanguageInfo, error) {
	return LanguageInfo{
		Code:       "en",
		Name:       "English",
		Confidence: 0.95,
	}, nil
}

func setupTestHandler() (*APIHandler, *DefaultModelManager, core.VectorStore) {
	// Create test components
	persistence := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persistence, indexFactory)
	modelManager := NewModelManager(2)
	analyzer := &MockContentAnalyzer{}

	// Set the engine factory to create mock engines
	modelManager.SetEngineFactory(func(cfg ModelConfig) (EmbeddingEngine, error) {
		return NewMockEmbeddingEngine(cfg), nil
	})

	// Load a test model
	config := ModelConfig{
		Name:      "test-model",
		Type:      "onnx",
		MaxTokens: 512,
		Dimensions: 384,
	}
	err := modelManager.LoadModel(context.Background(), "test-model", config)
	if err != nil {
		panic(fmt.Sprintf("Failed to load test model: %v", err))
	}

	// Create test collections
	testCollections := []core.Collection{
		{
			Name:      "test-collection",
			Dimension: 384,
			IndexType: "flat",
			Distance:  "cosine",
		},
		{
			Name:      "batch-collection",
			Dimension: 384,
			IndexType: "flat",
			Distance:  "cosine",
		},
		{
			Name:      "benchmark-collection",
			Dimension: 384,
			IndexType: "flat",
			Distance:  "cosine",
		},
	}

	for _, collection := range testCollections {
		vectorStore.CreateCollection(context.Background(), collection)
	}

	handler := NewAPIHandler(modelManager, vectorStore, analyzer)
	return handler, modelManager, vectorStore
}

func TestAPIHandler_HandleEmbed(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Create test request
	req := EmbedRequest{
		Content:    []string{"This is a test document", "Another test document"},
		ModelName:  "test-model",
		Collection: "test-collection",
		Analyze:    true,
		Options: EmbedOptions{
			IncludeVector: true,
		},
	}

	// Convert to JSON
	reqBody, _ := json.Marshal(req)

	// Create HTTP request
	httpReq := httptest.NewRequest(http.MethodPost, "/v1/embed", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	// Create response recorder
	w := httptest.NewRecorder()

	// Call handler
	handler.HandleEmbed(w, httpReq)

	// Check response
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
		t.Logf("Response body: %s", w.Body.String())
	}

	var response EmbedResponse
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if !response.Success {
		t.Errorf("Expected success=true, got success=%v", response.Success)
		if response.Message != "" {
			t.Logf("Message: %s", response.Message)
		}
	}

	if len(response.Vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(response.Vectors))
	}

	if len(response.Analysis) != 2 {
		t.Errorf("Expected 2 analysis results, got %d", len(response.Analysis))
	}

	if response.Stats.ProcessedCount != 2 {
		t.Errorf("Expected processed count 2, got %d", response.Stats.ProcessedCount)
	}
}

func TestAPIHandler_HandleEmbedInvalidMethod(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Create GET request (should fail)
	httpReq := httptest.NewRequest(http.MethodGet, "/v1/embed", nil)
	w := httptest.NewRecorder()

	handler.HandleEmbed(w, httpReq)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405, got %d", w.Code)
	}
}

func TestAPIHandler_HandleEmbedMissingContent(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Create request with no content
	req := EmbedRequest{
		Collection: "test-collection",
	}

	reqBody, _ := json.Marshal(req)
	httpReq := httptest.NewRequest(http.MethodPost, "/v1/embed", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.HandleEmbed(w, httpReq)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestAPIHandler_HandleEmbedMissingCollection(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Create request with no collection
	req := EmbedRequest{
		Content: []string{"test content"},
	}

	reqBody, _ := json.Marshal(req)
	httpReq := httptest.NewRequest(http.MethodPost, "/v1/embed", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.HandleEmbed(w, httpReq)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestAPIHandler_HandleBatchEmbed(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Create batch request
	req := BatchEmbedRequest{
		Documents: []DocumentRequest{
			{
				ID:      "doc1",
				Content: "First document content",
				Metadata: map[string]interface{}{
					"author": "John Doe",
				},
				Type: "article",
			},
			{
				ID:      "doc2",
				Content: "Second document content",
				Metadata: map[string]interface{}{
					"author": "Jane Smith",
				},
				Type: "article",
			},
		},
		ModelName:  "test-model",
		Collection: "batch-collection",
		Options: EmbedOptions{
			IncludeVector: true,
			BatchSize:     2,
		},
	}

	reqBody, _ := json.Marshal(req)
	httpReq := httptest.NewRequest(http.MethodPost, "/v1/embed/batch", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.HandleBatchEmbed(w, httpReq)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
		t.Logf("Response body: %s", w.Body.String())
	}

	var response EmbedResponse
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if !response.Success {
		t.Errorf("Expected success=true, got success=%v", response.Success)
		if response.Message != "" {
			t.Logf("Message: %s", response.Message)
		}
	}

	if len(response.Vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(response.Vectors))
	}

	// Check that document IDs are preserved
	if len(response.Vectors) > 0 && response.Vectors[0].ID != "doc1" {
		t.Errorf("Expected vector ID 'doc1', got '%s'", response.Vectors[0].ID)
	}

	// Check metadata
	if response.Vectors[0].Metadata["author"] != "John Doe" {
		t.Errorf("Expected author 'John Doe', got '%v'", response.Vectors[0].Metadata["author"])
	}

	if response.Vectors[0].Metadata["content_type"] != "article" {
		t.Errorf("Expected content_type 'article', got '%v'", response.Vectors[0].Metadata["content_type"])
	}
}

func TestAPIHandler_HandleModelList(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	httpReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	handler.HandleModelList(w, httpReq)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if !response["success"].(bool) {
		t.Errorf("Expected success=true")
	}

	models := response["models"].([]interface{})
	if len(models) == 0 {
		t.Errorf("Expected at least one model")
	}
}

func TestAPIHandler_HandleModelHealth(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Test health for all models
	httpReq := httptest.NewRequest(http.MethodGet, "/v1/models/health", nil)
	w := httptest.NewRecorder()

	handler.HandleModelHealth(w, httpReq)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if !response["success"].(bool) {
		t.Errorf("Expected success=true")
	}

	// Test health for specific model
	httpReq = httptest.NewRequest(http.MethodGet, "/v1/models/health?model=test-model", nil)
	w = httptest.NewRecorder()

	handler.HandleModelHealth(w, httpReq)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestAPIHandler_HandleModelHealthNotFound(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	httpReq := httptest.NewRequest(http.MethodGet, "/v1/models/health?model=nonexistent", nil)
	w := httptest.NewRecorder()

	handler.HandleModelHealth(w, httpReq)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected status 404, got %d", w.Code)
	}
}

func TestAPIHandler_ProcessContent(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	// Test without chunking
	content := []string{"This is a test document"}
	metadata := []map[string]interface{}{{"author": "test"}}

	chunks, chunkMeta := handler.processContent(content, metadata, 0, 0)

	if len(chunks) != 1 {
		t.Errorf("Expected 1 chunk, got %d", len(chunks))
	}

	if chunks[0] != content[0] {
		t.Errorf("Content should be unchanged without chunking")
	}

	// Test with chunking
	longContent := []string{"This is a very long document that should be split into multiple chunks when chunking is enabled"}
	chunks, chunkMeta = handler.processContent(longContent, nil, 5, 1) // 5 words per chunk, 1 word overlap

	if len(chunks) <= 1 {
		t.Errorf("Expected multiple chunks, got %d", len(chunks))
	}

	// Verify metadata is properly set for chunks
	if chunkMeta[0]["chunk_index"].(int) != 0 {
		t.Errorf("Expected chunk_index 0, got %v", chunkMeta[0]["chunk_index"])
	}

	if chunkMeta[0]["source_index"].(int) != 0 {
		t.Errorf("Expected source_index 0, got %v", chunkMeta[0]["source_index"])
	}
}

func TestAPIHandler_EstimateTokens(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	content := []string{"Hello world", "This is a test"}
	tokens := handler.estimateTokens(content)

	expectedChars := len("Hello world") + len("This is a test")
	expectedTokens := expectedChars / 4

	if tokens != expectedTokens {
		t.Errorf("Expected %d tokens, got %d", expectedTokens, tokens)
	}
}

func TestAPIHandler_RegisterRoutes(t *testing.T) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)

	// Test that routes are registered by making requests
	testCases := []struct {
		path   string
		method string
	}{
		{"/v1/embed", http.MethodPost},
		{"/v1/embed/batch", http.MethodPost},
		{"/v1/models", http.MethodGet},
		{"/v1/models/health", http.MethodGet},
	}

	for _, tc := range testCases {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		w := httptest.NewRecorder()

		mux.ServeHTTP(w, req)

		// Should not return 404 (route not found)
		if w.Code == http.StatusNotFound {
			t.Errorf("Route %s %s not registered", tc.method, tc.path)
		}
	}
}

func BenchmarkAPIHandler_HandleEmbed(b *testing.B) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	req := EmbedRequest{
		Content:    []string{"This is a test document for benchmarking"},
		ModelName:  "test-model",
		Collection: "benchmark-collection",
	}

	reqBody, _ := json.Marshal(req)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		httpReq := httptest.NewRequest(http.MethodPost, "/v1/embed", bytes.NewReader(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler.HandleEmbed(w, httpReq)

		if w.Code != http.StatusOK {
			b.Fatalf("Request failed with status %d", w.Code)
		}
	}
}

func BenchmarkAPIHandler_ProcessContent(b *testing.B) {
	handler, modelManager, _ := setupTestHandler()
	defer modelManager.Close()

	content := []string{"This is a test document that will be used for benchmarking the content processing functionality"}
	metadata := []map[string]interface{}{{"test": "value"}}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		handler.processContent(content, metadata, 10, 2)
	}
}
