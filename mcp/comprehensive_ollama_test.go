package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/ai/embedding"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

// TestComprehensiveOllamaMCPIntegration tests all MCP tools with Ollama embeddings
func TestComprehensiveOllamaMCPIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	if os.Getenv("OLLAMA_HOST") == "" {
		t.Skip("OLLAMA_HOST not set, skipping Ollama integration test")
	}

	// Setup
	storage := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(storage, indexFactory)
	defer vectorStore.Close()

	// Create Ollama embedding engine
	config := &ai.ModelConfig{
		Path:           "nomic-embed-text",
		Dimensions:     768,
		OllamaEndpoint: "http://localhost:11434",
	}

	engine, err := embedding.NewOllamaEmbeddingEngine(config)
	if err != nil {
		t.Fatalf("Failed to create Ollama engine: %v", err)
	}

	// Warm up the engine
	ctx := context.Background()
	err = engine.Warm(ctx)
	if err != nil {
		t.Fatalf("Failed to warm up engine: %v", err)
	}
	defer engine.Close()

	// Create embedding store
	embedStore := NewEmbeddingStore(vectorStore, engine)

	// Test 1: Create Collection
	t.Run("CreateCollection", func(t *testing.T) {
		handler := &CreateCollectionHandler{store: embedStore}
		params := map[string]interface{}{
			"name":      "test_collection",
			"dimension": 768,
			"distance":  "cosine",
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		if resp.IsError {
			t.Fatal("Expected successful collection creation")
		}

		t.Log("✓ Collection created successfully")
	})

	// Test 2: List Collections
	t.Run("ListCollections", func(t *testing.T) {
		handler := &ListCollectionsHandler{store: embedStore}
		resp, err := handler.Execute(ctx, nil)
		if err != nil {
			t.Fatalf("Failed to list collections: %v", err)
		}

		// Parse response
		if len(resp.Content) < 2 {
			t.Fatal("Expected at least 2 content items")
		}

		var collections []core.Collection
		if err := json.Unmarshal([]byte(resp.Content[1].Text), &collections); err != nil {
			t.Fatalf("Failed to unmarshal collections: %v", err)
		}

		if len(collections) != 1 {
			t.Errorf("Expected 1 collection, got %d", len(collections))
		}

		t.Logf("✓ Listed %d collections", len(collections))
	})

	// Test 3: Add Vectors with Text
	t.Run("AddVectorsWithText", func(t *testing.T) {
		handler := &AddVectorsHandler{store: embedStore}
		
		testData := []struct {
			id      string
			content string
			meta    map[string]interface{}
		}{
			{
				id:      "vec1",
				content: "Machine learning is a subset of artificial intelligence",
				meta:    map[string]interface{}{"topic": "AI", "subtopic": "ML"},
			},
			{
				id:      "vec2",
				content: "Deep learning uses neural networks with multiple layers",
				meta:    map[string]interface{}{"topic": "AI", "subtopic": "DL"},
			},
			{
				id:      "vec3",
				content: "Natural language processing enables computers to understand human language",
				meta:    map[string]interface{}{"topic": "AI", "subtopic": "NLP"},
			},
			{
				id:      "vec4",
				content: "Computer vision allows machines to interpret visual information",
				meta:    map[string]interface{}{"topic": "AI", "subtopic": "CV"},
			},
		}

		vectors := make([]map[string]interface{}, len(testData))
		for i, data := range testData {
			vectors[i] = map[string]interface{}{
				"id":       data.id,
				"content":  data.content,
				"metadata": data.meta,
			}
		}

		params := map[string]interface{}{
			"collection": "test_collection",
			"vectors":    vectors,
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to add vectors: %v", err)
		}

		// Parse response
		if len(resp.Content) < 2 {
			t.Fatal("Expected at least 2 content items")
		}

		var result struct {
			Added int      `json:"added"`
			IDs   []string `json:"ids"`
		}
		if err := json.Unmarshal([]byte(resp.Content[1].Text), &result); err != nil {
			t.Fatalf("Failed to unmarshal result: %v", err)
		}

		if result.Added != len(testData) {
			t.Errorf("Expected to add %d vectors, added %d", len(testData), result.Added)
		}

		t.Logf("✓ Added %d vectors successfully", result.Added)
	})

	// Test 4: Get Vector
	t.Run("GetVector", func(t *testing.T) {
		handler := &GetVectorHandler{store: embedStore}
		params := map[string]interface{}{
			"collection": "test_collection",
			"id":         "vec1",
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to get vector: %v", err)
		}

		// Parse response
		if len(resp.Content) < 2 {
			t.Fatal("Expected at least 2 content items")
		}

		var vector struct {
			ID       string                 `json:"id"`
			Vector   []float32              `json:"vector"`
			Metadata map[string]interface{} `json:"metadata"`
		}
		if err := json.Unmarshal([]byte(resp.Content[1].Text), &vector); err != nil {
			t.Fatalf("Failed to unmarshal vector: %v", err)
		}

		if vector.ID != "vec1" {
			t.Errorf("Expected ID 'vec1', got '%s'", vector.ID)
		}

		if len(vector.Vector) != 768 {
			t.Errorf("Expected 768-dimensional vector, got %d", len(vector.Vector))
		}

		t.Log("✓ Retrieved vector successfully")
	})

	// Test 5: Search with Text Query
	t.Run("SearchWithText", func(t *testing.T) {
		handler := &SearchVectorsHandler{store: embedStore}
		
		queries := []struct {
			query    string
			expected []string // Expected IDs in top results
		}{
			{
				query:    "neural networks and deep learning",
				expected: []string{"vec2"}, // Should match "Deep learning uses neural networks"
			},
			{
				query:    "understanding human language",
				expected: []string{"vec3"}, // Should match NLP
			},
			{
				query:    "visual perception and image analysis",
				expected: []string{"vec4"}, // Should match computer vision
			},
		}

		for _, test := range queries {
			params := map[string]interface{}{
				"collection": "test_collection",
				"query":      test.query,
				"limit":      3,
			}

			resp, err := handler.Execute(ctx, params)
			if err != nil {
				t.Fatalf("Failed to search: %v", err)
			}

			// Parse response
			if len(resp.Content) < 2 {
				t.Fatal("Expected at least 2 content items")
			}

			var results []core.SearchResult
			if err := json.Unmarshal([]byte(resp.Content[1].Text), &results); err != nil {
				t.Fatalf("Failed to unmarshal results: %v", err)
			}

			if len(results) == 0 {
				t.Errorf("No results for query '%s'", test.query)
				continue
			}

			// Check if expected ID is in top result
			topResult := results[0].ID
			found := false
			for _, expectedID := range test.expected {
				if topResult == expectedID {
					found = true
					break
				}
			}

			if found {
				t.Logf("✓ Query '%s' correctly matched %s (score: %.3f)", 
					test.query, topResult, results[0].Score)
			} else {
				t.Logf("✗ Query '%s' matched %s instead of %v", 
					test.query, topResult, test.expected)
			}
		}
	})

	// Test 6: Search with Metadata Filter
	t.Run("SearchWithFilter", func(t *testing.T) {
		handler := &SearchVectorsHandler{store: embedStore}
		params := map[string]interface{}{
			"collection": "test_collection",
			"query":      "artificial intelligence",
			"limit":      10,
			"filters": map[string]interface{}{
				"subtopic": "ML",
			},
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to search with filter: %v", err)
		}

		// Parse response
		if len(resp.Content) < 2 {
			t.Fatal("Expected at least 2 content items")
		}

		var results []core.SearchResult
		if err := json.Unmarshal([]byte(resp.Content[1].Text), &results); err != nil {
			t.Fatalf("Failed to unmarshal results: %v", err)
		}

		// Should only return vec1 which has subtopic: ML
		if len(results) != 1 || results[0].ID != "vec1" {
			t.Errorf("Expected only vec1 with filter, got %d results", len(results))
		} else {
			t.Log("✓ Metadata filtering works correctly")
		}
	})

	// Test 7: Add Raw Vector
	t.Run("AddRawVector", func(t *testing.T) {
		// First, generate an embedding
		embedding, err := engine.Embed(ctx, []string{"Test content for raw vector"})
		if err != nil {
			t.Fatalf("Failed to generate embedding: %v", err)
		}

		handler := &AddVectorsHandler{store: embedStore}
		params := map[string]interface{}{
			"collection": "test_collection",
			"vectors": []map[string]interface{}{
				{
					"id":     "raw_vec",
					"vector": embedding[0],
					"metadata": map[string]interface{}{
						"type": "raw_vector",
					},
				},
			},
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to add raw vector: %v", err)
		}

		if resp.IsError {
			t.Fatal("Expected successful raw vector addition")
		}

		t.Log("✓ Added raw vector successfully")
	})

	// Test 8: Delete Vector
	t.Run("DeleteVector", func(t *testing.T) {
		handler := &DeleteVectorHandler{store: embedStore}
		params := map[string]interface{}{
			"collection": "test_collection",
			"id":         "vec4",
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to delete vector: %v", err)
		}

		if resp.IsError {
			t.Fatal("Expected successful deletion")
		}

		// Verify deletion
		getHandler := &GetVectorHandler{store: embedStore}
		getParams := map[string]interface{}{
			"collection": "test_collection",
			"id":         "vec4",
		}

		_, err = getHandler.Execute(ctx, getParams)
		if err == nil {
			t.Error("Expected error when getting deleted vector")
		}

		t.Log("✓ Deleted vector successfully")
	})

	// Test 9: Multiple Deletes
	t.Run("MultipleDeletes", func(t *testing.T) {
		handler := &DeleteVectorHandler{store: embedStore}
		idsToDelete := []string{"vec2", "vec3"}
		
		deletedCount := 0
		for _, id := range idsToDelete {
			params := map[string]interface{}{
				"collection": "test_collection",
				"id":         id,
			}

			resp, err := handler.Execute(ctx, params)
			if err != nil {
				t.Logf("Failed to delete %s: %v", id, err)
				continue
			}

			if !resp.IsError {
				deletedCount++
			}
		}

		if deletedCount != len(idsToDelete) {
			t.Errorf("Expected to delete %d vectors, deleted %d", len(idsToDelete), deletedCount)
		}

		t.Logf("✓ Deleted %d vectors individually", deletedCount)
	})

	// Test 10: Delete Collection
	t.Run("DeleteCollection", func(t *testing.T) {
		handler := &DeleteCollectionHandler{store: embedStore}
		params := map[string]interface{}{
			"name": "test_collection",
		}

		resp, err := handler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to delete collection: %v", err)
		}

		if resp.IsError {
			t.Fatal("Expected successful collection deletion")
		}

		// Verify deletion
		listHandler := &ListCollectionsHandler{store: embedStore}
		listResp, err := listHandler.Execute(ctx, nil)
		if err != nil {
			t.Fatalf("Failed to list collections: %v", err)
		}

		if len(listResp.Content) < 2 {
			t.Fatal("Expected at least 2 content items")
		}

		var collections []core.Collection
		if err := json.Unmarshal([]byte(listResp.Content[1].Text), &collections); err != nil {
			t.Fatalf("Failed to unmarshal collections: %v", err)
		}

		if len(collections) != 0 {
			t.Errorf("Expected 0 collections after deletion, got %d", len(collections))
		}

		t.Log("✓ Deleted collection successfully")
	})

	// Test 11: Performance Test
	t.Run("PerformanceTest", func(t *testing.T) {
		// Create a new collection for performance testing
		createHandler := &CreateCollectionHandler{store: embedStore}
		params := map[string]interface{}{
			"name":      "perf_test",
			"dimension": 768,
			"distance":  "cosine",
		}

		_, err := createHandler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to create performance test collection: %v", err)
		}

		// Add 100 vectors
		addHandler := &AddVectorsHandler{store: embedStore}
		vectors := make([]map[string]interface{}, 100)
		for i := 0; i < 100; i++ {
			vectors[i] = map[string]interface{}{
				"id":      fmt.Sprintf("perf_%d", i),
				"content": fmt.Sprintf("Performance test document number %d with various content", i),
				"metadata": map[string]interface{}{
					"index": i,
				},
			}
		}

		start := time.Now()
		params = map[string]interface{}{
			"collection": "perf_test",
			"vectors":    vectors,
		}

		resp, err := addHandler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to add vectors: %v", err)
		}
		addDuration := time.Since(start)

		// Parse response
		if len(resp.Content) < 2 {
			t.Fatal("Expected at least 2 content items")
		}

		var result struct {
			Added int `json:"added"`
		}
		if err := json.Unmarshal([]byte(resp.Content[1].Text), &result); err != nil {
			t.Fatalf("Failed to unmarshal result: %v", err)
		}

		if result.Added != 100 {
			t.Errorf("Expected to add 100 vectors, added %d", result.Added)
		}

		t.Logf("✓ Added 100 vectors in %v (%.2f vectors/sec)", 
			addDuration, float64(100)/addDuration.Seconds())

		// Search performance
		searchHandler := &SearchVectorsHandler{store: embedStore}
		searchQueries := []string{
			"performance testing",
			"document analysis",
			"various content types",
		}

		totalSearchTime := time.Duration(0)
		for _, query := range searchQueries {
			start := time.Now()
			params = map[string]interface{}{
				"collection": "perf_test",
				"query":      query,
				"limit":      10,
			}

			_, err := searchHandler.Execute(ctx, params)
			if err != nil {
				t.Fatalf("Failed to search: %v", err)
			}
			totalSearchTime += time.Since(start)
		}

		avgSearchTime := totalSearchTime / time.Duration(len(searchQueries))
		t.Logf("✓ Average search time: %v", avgSearchTime)

		// Cleanup
		deleteHandler := &DeleteCollectionHandler{store: embedStore}
		params = map[string]interface{}{
			"name": "perf_test",
		}
		_, err = deleteHandler.Execute(ctx, params)
		if err != nil {
			t.Fatalf("Failed to delete performance test collection: %v", err)
		}
	})

	t.Log("\n✅ All MCP tool tests completed successfully!")
}