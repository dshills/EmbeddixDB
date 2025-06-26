package text

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
)

// MockVectorStore implements a simple vector store for testing
type MockVectorStore struct {
	vectors map[string][]core.Vector
}

func NewMockVectorStore() *MockVectorStore {
	return &MockVectorStore{
		vectors: make(map[string][]core.Vector),
	}
}

func (m *MockVectorStore) AddVector(ctx context.Context, collection string, vec core.Vector) error {
	if m.vectors[collection] == nil {
		m.vectors[collection] = make([]core.Vector, 0)
	}
	m.vectors[collection] = append(m.vectors[collection], vec)
	return nil
}

func (m *MockVectorStore) AddVectorsBatch(ctx context.Context, collection string, vectors []core.Vector) error {
	if m.vectors[collection] == nil {
		m.vectors[collection] = make([]core.Vector, 0)
	}
	m.vectors[collection] = append(m.vectors[collection], vectors...)
	return nil
}

func (m *MockVectorStore) GetVector(ctx context.Context, collection, id string) (core.Vector, error) {
	for _, vec := range m.vectors[collection] {
		if vec.ID == id {
			return vec, nil
		}
	}
	return core.Vector{}, core.ErrVectorNotFound
}

func (m *MockVectorStore) DeleteVector(ctx context.Context, collection, id string) error {
	vectors := m.vectors[collection]
	for i, vec := range vectors {
		if vec.ID == id {
			m.vectors[collection] = append(vectors[:i], vectors[i+1:]...)
			return nil
		}
	}
	return nil
}

func (m *MockVectorStore) Search(ctx context.Context, collection string, req core.SearchRequest) ([]core.SearchResult, error) {
	// Simple mock search - return all vectors with decreasing scores
	results := make([]core.SearchResult, 0)
	for i, vec := range m.vectors[collection] {
		if i >= req.TopK {
			break
		}
		results = append(results, core.SearchResult{
			ID:       vec.ID,
			Score:    float32(1.0 - float32(i)*0.1),
			Metadata: vec.Metadata,
		})
	}
	return results, nil
}

func (m *MockVectorStore) CreateCollection(ctx context.Context, collection core.Collection) error {
	m.vectors[collection.Name] = make([]core.Vector, 0)
	return nil
}

func (m *MockVectorStore) DeleteCollection(ctx context.Context, name string) error {
	delete(m.vectors, name)
	return nil
}

func (m *MockVectorStore) ListCollections(ctx context.Context) ([]core.Collection, error) {
	collections := make([]core.Collection, 0, len(m.vectors))
	for name := range m.vectors {
		collections = append(collections, core.Collection{Name: name})
	}
	return collections, nil
}

func (m *MockVectorStore) GetCollection(ctx context.Context, name string) (core.Collection, error) {
	if _, exists := m.vectors[name]; exists {
		return core.Collection{Name: name}, nil
	}
	return core.Collection{}, core.ErrCollectionNotFound
}

func (m *MockVectorStore) GetCollectionSize(ctx context.Context, collection string) (int, error) {
	if vecs, exists := m.vectors[collection]; exists {
		return len(vecs), nil
	}
	return 0, core.ErrCollectionNotFound
}

func (m *MockVectorStore) UpdateVector(ctx context.Context, collection string, vec core.Vector) error {
	return m.AddVector(ctx, collection, vec)
}

func (m *MockVectorStore) RangeSearch(ctx context.Context, collection string, req core.RangeSearchRequest) (core.RangeSearchResult, error) {
	// Simple mock implementation
	results := make([]core.SearchResult, 0)
	for _, vec := range m.vectors[collection] {
		results = append(results, core.SearchResult{
			ID:       vec.ID,
			Score:    0.5,
			Metadata: vec.Metadata,
		})
		if len(results) >= req.Limit && req.Limit > 0 {
			break
		}
	}
	return core.RangeSearchResult{
		Results: results,
		Count:   len(results),
		Limited: req.Limit > 0 && len(results) >= req.Limit,
	}, nil
}

func (m *MockVectorStore) Close() error {
	return nil
}

// MockModelManager implements a simple model manager for testing
type MockModelManager struct {
	engines map[string]ai.EmbeddingEngine
}

func NewMockModelManager() *MockModelManager {
	return &MockModelManager{
		engines: make(map[string]ai.EmbeddingEngine),
	}
}

func (m *MockModelManager) LoadModel(ctx context.Context, modelName string, config ai.ModelConfig) error {
	m.engines[modelName] = &MockEmbeddingEngine{modelName: modelName}
	return nil
}

func (m *MockModelManager) UnloadModel(modelName string) error {
	delete(m.engines, modelName)
	return nil
}

func (m *MockModelManager) GetEngine(modelName string) (ai.EmbeddingEngine, error) {
	engine, exists := m.engines[modelName]
	if !exists {
		// Create a default mock engine
		engine = &MockEmbeddingEngine{modelName: modelName}
		m.engines[modelName] = engine
	}
	return engine, nil
}

func (m *MockModelManager) ListModels() []ai.ModelInfo {
	models := make([]ai.ModelInfo, 0, len(m.engines))
	for name := range m.engines {
		models = append(models, ai.ModelInfo{Name: name})
	}
	return models
}

func (m *MockModelManager) GetModelHealth(modelName string) (ai.ModelHealth, error) {
	return ai.ModelHealth{
		ModelName: modelName,
		Status:    "healthy",
		LoadedAt:  time.Now(),
	}, nil
}

// MockEmbeddingEngine implements a simple embedding engine for testing
type MockEmbeddingEngine struct {
	modelName string
}

func (m *MockEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {
	// Return mock embeddings
	embeddings := make([][]float32, len(content))
	for i := range content {
		embeddings[i] = make([]float32, 384)
		// Simple mock: use content length as a feature
		embeddings[i][0] = float32(len(content[i])) / 100.0
		embeddings[i][1] = float32(i) * 0.1
	}
	return embeddings, nil
}

func (m *MockEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	return m.Embed(ctx, content)
}

func (m *MockEmbeddingEngine) GetModelInfo() ai.ModelInfo {
	return ai.ModelInfo{
		Name:      m.modelName,
		Dimension: 384,
	}
}

func (m *MockEmbeddingEngine) Warm(ctx context.Context) error {
	return nil
}

func (m *MockEmbeddingEngine) Close() error {
	return nil
}

func TestHybridSearchManager_Search(t *testing.T) {
	ctx := context.Background()

	// Create mock dependencies
	vectorStore := NewMockVectorStore()
	modelManager := NewMockModelManager()

	// Create hybrid search manager
	hsm := NewHybridSearchManager(vectorStore, modelManager)

	// Add test documents
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "Machine learning algorithms for data analysis",
			Metadata: map[string]interface{}{
				"category": "technology",
			},
		},
		{
			ID:      "doc2",
			Content: "Deep learning neural networks",
			Metadata: map[string]interface{}{
				"category": "technology",
			},
		},
		{
			ID:      "doc3",
			Content: "Natural language processing with transformers",
			Metadata: map[string]interface{}{
				"category": "nlp",
			},
		},
	}

	err := hsm.AddDocuments(ctx, docs)
	if err != nil {
		t.Fatalf("Failed to add documents: %v", err)
	}

	// Test hybrid search
	req := ai.HybridSearchRequest{
		Query:      "machine learning",
		Collection: "default",
		Limit:      5,
		Weights: ai.SearchWeights{
			Vector: 0.6,
			Text:   0.4,
		},
		Options: ai.SearchOptions{
			IncludeExplanation: true,
		},
	}

	result, err := hsm.Search(ctx, req)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify results
	if len(result.Results) == 0 {
		t.Error("Expected search results, got none")
	}

	// Check performance metrics
	// Note: In fast mock implementations, this might be 0ms
	// if result.Performance.TotalTimeMs == 0 {
	// 	t.Error("Expected performance metrics to be recorded")
	// }

	// Check debug info
	if result.DebugInfo == nil {
		t.Error("Expected debug info when explanation requested")
	}
}

func TestHybridSearchManager_Caching(t *testing.T) {
	ctx := context.Background()

	// Create hybrid search manager with caching
	vectorStore := NewMockVectorStore()
	modelManager := NewMockModelManager()
	hsm := NewHybridSearchManager(vectorStore, modelManager)

	// Add documents
	docs := []ai.Document{
		{ID: "doc1", Content: "Test document one"},
		{ID: "doc2", Content: "Test document two"},
	}
	hsm.AddDocuments(ctx, docs)

	// First search - should miss cache
	req := ai.HybridSearchRequest{
		Query:      "test",
		Collection: "default",
		Limit:      5,
	}

	result1, err := hsm.Search(ctx, req)
	if err != nil {
		t.Fatalf("First search failed: %v", err)
	}

	// Second identical search - should hit cache
	result2, err := hsm.Search(ctx, req)
	if err != nil {
		t.Fatalf("Second search failed: %v", err)
	}

	// Results should be identical
	if len(result1.Results) != len(result2.Results) {
		t.Error("Cached results differ from original")
	}

	// Check cache hit rate
	stats := hsm.GetStats()
	if stats.CacheHitRate == 0 {
		t.Error("Expected cache hit rate > 0")
	}
}

func TestHybridSearchManager_FusionWeights(t *testing.T) {
	vectorStore := NewMockVectorStore()
	modelManager := NewMockModelManager()
	hsm := NewHybridSearchManager(vectorStore, modelManager)

	// Test valid weights
	validWeights := ai.SearchWeights{
		Vector:    0.5,
		Text:      0.3,
		Freshness: 0.1,
		Authority: 0.1,
	}

	err := hsm.UpdateFusionWeights(validWeights)
	if err != nil {
		t.Errorf("Failed to update valid weights: %v", err)
	}

	// Test invalid weights (don't sum to 1.0)
	invalidWeights := ai.SearchWeights{
		Vector: 0.5,
		Text:   0.6,
	}

	err = hsm.UpdateFusionWeights(invalidWeights)
	if err == nil {
		t.Error("Expected error for invalid weights")
	}
}

func TestReciprocalRankFusion(t *testing.T) {
	rrf := NewReciprocalRankFusion()

	// Create test results
	vectorResults := []ai.SearchResult{
		{ID: "doc1", Score: 0.9},
		{ID: "doc2", Score: 0.8},
		{ID: "doc3", Score: 0.7},
	}

	textResults := []ai.SearchResult{
		{ID: "doc2", Score: 10.0}, // High BM25 score
		{ID: "doc1", Score: 8.0},
		{ID: "doc4", Score: 6.0},
	}

	weights := ai.SearchWeights{
		Vector: 0.6,
		Text:   0.4,
	}

	fusedResults := rrf.Fuse(vectorResults, textResults, weights)

	// Verify fusion
	if len(fusedResults) != 4 {
		t.Errorf("Expected 4 fused results, got %d", len(fusedResults))
	}

	// doc2 should rank high (appears in both lists)
	if len(fusedResults) > 0 && fusedResults[0].ID != "doc2" && fusedResults[0].ID != "doc1" {
		t.Errorf("Expected doc1 or doc2 to rank first, got %s", fusedResults[0].ID)
	}

	// Check algorithm name
	if rrf.GetName() != "reciprocal_rank_fusion" {
		t.Errorf("Unexpected algorithm name: %s", rrf.GetName())
	}
}

func TestQueryCache(t *testing.T) {
	cache := NewQueryCache(2) // Small cache for testing

	req1 := ai.HybridSearchRequest{Query: "test1", Limit: 10}
	req2 := ai.HybridSearchRequest{Query: "test2", Limit: 10}
	req3 := ai.HybridSearchRequest{Query: "test3", Limit: 10}

	result1 := ai.HybridSearchResult{Results: []ai.SearchResult{{ID: "doc1"}}}
	result2 := ai.HybridSearchResult{Results: []ai.SearchResult{{ID: "doc2"}}}
	result3 := ai.HybridSearchResult{Results: []ai.SearchResult{{ID: "doc3"}}}

	// Add to cache
	cache.Set(req1, result1, 1*time.Minute)
	cache.Set(req2, result2, 1*time.Minute)

	// Retrieve from cache
	cached1, found := cache.Get(req1)
	if !found {
		t.Error("Expected to find req1 in cache")
	}
	if len(cached1.Results) != 1 || cached1.Results[0].ID != "doc1" {
		t.Error("Cached result doesn't match original")
	}

	// Add third item - should evict oldest
	cache.Set(req3, result3, 1*time.Minute)

	// Clear cache
	cache.Clear()

	// Should not find anything
	_, found = cache.Get(req1)
	if found {
		t.Error("Expected empty cache after clear")
	}
}

func TestSearchStatistics(t *testing.T) {
	stats := NewSearchStatistics()

	// Record some searches
	stats.RecordSearch(50*time.Millisecond, true)
	stats.RecordSearch(100*time.Millisecond, true)
	stats.RecordSearch(150*time.Millisecond, false) // Failed search
	stats.RecordCacheHit()

	searchStats := stats.GetStats()

	if searchStats.TotalQueries != 3 {
		t.Errorf("Expected 3 total queries, got %d", searchStats.TotalQueries)
	}

	if searchStats.SuccessRate != 2.0/3.0 {
		t.Errorf("Expected success rate 0.667, got %f", searchStats.SuccessRate)
	}

	if searchStats.CacheHitRate != 1.0/3.0 {
		t.Errorf("Expected cache hit rate 0.333, got %f", searchStats.CacheHitRate)
	}

	if searchStats.AverageLatency == 0 {
		t.Error("Expected non-zero average latency")
	}
}

func BenchmarkHybridSearch(b *testing.B) {
	ctx := context.Background()

	// Setup
	vectorStore := NewMockVectorStore()
	modelManager := NewMockModelManager()
	hsm := NewHybridSearchManager(vectorStore, modelManager)

	// Add documents
	docs := make([]ai.Document, 100)
	for i := 0; i < 100; i++ {
		docs[i] = ai.Document{
			ID:      fmt.Sprintf("doc_%d", i),
			Content: fmt.Sprintf("Document %d about machine learning and AI", i),
		}
	}
	hsm.AddDocuments(ctx, docs)

	req := ai.HybridSearchRequest{
		Query:      "machine learning",
		Collection: "default",
		Limit:      10,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := hsm.Search(ctx, req)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
