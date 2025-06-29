package mcp

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/ai/embedding"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func TestOllamaEmbeddingIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	if os.Getenv("OLLAMA_HOST") == "" {
		t.Skip("OLLAMA_HOST not set, skipping Ollama integration test")
	}

	// Create in-memory storage
	storage := persistence.NewMemoryPersistence()

	// Create vector store
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
	err = engine.Warm(context.Background())
	if err != nil {
		t.Fatalf("Failed to warm up engine: %v", err)
	}
	defer engine.Close()

	// Test embedding generation
	text := "This is a test document for vector embedding"
	embeddings, err := engine.Embed(context.Background(), []string{text})
	if err != nil {
		t.Fatalf("Failed to embed text: %v", err)
	}

	if len(embeddings) != 1 {
		t.Fatalf("Expected 1 embedding, got %d", len(embeddings))
	}
	embedding := embeddings[0]

	// Verify embedding dimensions
	if len(embedding) != 768 {
		t.Fatalf("Expected 768 dimensions, got %d", len(embedding))
	}

	// Create a collection
	collection := core.Collection{
		Name:      "test_collection",
		Dimension: 768,
		Distance:  "cosine",
		IndexType: "flat",
		Metadata:  map[string]interface{}{"test": true},
	}

	err = vectorStore.CreateCollection(context.Background(), collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add vector with embedding
	vector := core.Vector{
		ID:       "test_vector",
		Values:   embedding,
		Metadata: map[string]string{"content": text},
	}

	// Debug: check embedding
	t.Logf("Embedding dimension: %d", len(embedding))
	t.Logf("First few embedding values: %v", embedding[:5])

	err = vectorStore.AddVector(context.Background(), "test_collection", vector)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Verify collection
	collections, err := vectorStore.ListCollections(context.Background())
	if err != nil {
		t.Fatalf("Failed to list collections: %v", err)
	}
	t.Logf("Collections: %+v", collections)

	// Search for similar vectors
	searchReq := core.SearchRequest{
		Query: embedding,
		TopK:  5,
	}

	results, err := vectorStore.Search(context.Background(), "test_collection", searchReq)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("No results returned from search")
	}

	// First result should be our vector with perfect score
	if results[0].ID != "test_vector" {
		t.Errorf("Expected first result to be 'test_vector', got %s", results[0].ID)
	}

	if results[0].Score < 0.99 {
		t.Errorf("Expected near-perfect score for identical vector, got %f", results[0].Score)
	}

	t.Logf("Successfully completed Ollama integration test")
	t.Logf("Search results:")
	for i, result := range results {
		t.Logf("  Result %d: ID=%s, Score=%f", i+1, result.ID, result.Score)
	}
}

func TestMCPServerWithOllama(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	if os.Getenv("OLLAMA_HOST") == "" {
		t.Skip("OLLAMA_HOST not set, skipping MCP server test")
	}

	// Create in-memory storage
	storage := persistence.NewMemoryPersistence()

	// Create vector store
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
	err = engine.Warm(context.Background())
	if err != nil {
		t.Fatalf("Failed to warm up engine: %v", err)
	}
	defer engine.Close()

	// Create embedding store
	embedStore := NewEmbeddingStore(vectorStore, engine)

	// Create MCP server
	server := NewServer(embedStore)
	_ = server // suppress unused variable warning

	// Test creating a collection through MCP handler
	handler := &CreateCollectionHandler{store: embedStore}
	params := map[string]interface{}{
		"name":      "mcp_test",
		"dimension": 768,
		"distance":  "cosine",
	}

	// Debug: print params
	t.Logf("Create collection params: %+v", params)

	resp, err := handler.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	if resp.IsError {
		t.Fatal("Expected successful collection creation")
	}

	// Test adding a vector with text
	addHandler := &AddVectorsHandler{store: embedStore}
	addParams := map[string]interface{}{
		"collection": "mcp_test",
		"vectors": []map[string]interface{}{
			{
				"id":      "vec1",
				"content": "Machine learning and artificial intelligence",
				"metadata": map[string]interface{}{
					"topic": "AI",
				},
			},
		},
	}

	resp, err = addHandler.Execute(context.Background(), addParams)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Debug: print add response
	if resp.Content != nil {
		addContentBytes, _ := json.Marshal(resp.Content)
		t.Logf("Add vectors response: %s", string(addContentBytes))
	}

	// Test searching
	searchHandler := &SearchVectorsHandler{store: embedStore}
	searchParams := map[string]interface{}{
		"collection": "mcp_test",
		"query":      "AI and ML technologies",
		"limit":      5,
	}

	resp, err = searchHandler.Execute(context.Background(), searchParams)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if resp.IsError {
		t.Fatal("Search returned error")
	}

	// Check that we got results
	if resp.Content == nil {
		t.Fatal("Expected search results in response content")
	}

	contentBytes, err := json.Marshal(resp.Content)
	if err != nil {
		t.Fatalf("Failed to marshal content: %v", err)
	}

	// Debug: print the raw response
	t.Logf("Search response content: %s", string(contentBytes))

	// resp.Content is likely an array of ToolContent, not the actual results
	// We need to extract the text content from the response
	if len(resp.Content) < 2 {
		t.Fatalf("Expected at least 2 content items, got %d", len(resp.Content))
	}

	// The second content item should contain the JSON results
	var searchResults []core.SearchResult
	if err := json.Unmarshal([]byte(resp.Content[1].Text), &searchResults); err != nil {
		t.Fatalf("Failed to unmarshal search response: %v", err)
	}

	if len(searchResults) == 0 {
		t.Fatal("Expected non-empty search results - check if vector was added successfully")
	}

	if searchResults[0].ID != "vec1" {
		t.Errorf("Expected first result to be 'vec1', got %s", searchResults[0].ID)
	}

	t.Logf("Successfully completed MCP server test with Ollama")
	t.Logf("Search score: %f", searchResults[0].Score)
}

