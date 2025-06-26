package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func TestAPIEndpoints(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Test health endpoint
	t.Run("Health", func(t *testing.T) {
		req, err := http.NewRequest("GET", "/health", nil)
		if err != nil {
			t.Fatal(err)
		}

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
		}

		var response HealthResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
			t.Errorf("failed to unmarshal response: %v", err)
		}

		if response.Status != "healthy" {
			t.Errorf("expected status 'healthy', got %s", response.Status)
		}
	})

	// Test create collection
	t.Run("CreateCollection", func(t *testing.T) {
		reqBody := CreateCollectionRequest{
			Name:      "test_collection",
			Dimension: 128,
			IndexType: "flat",
			Distance:  "cosine",
		}

		body, _ := json.Marshal(reqBody)
		req, err := http.NewRequest("POST", "/collections", bytes.NewBuffer(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusCreated {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusCreated)
			t.Errorf("Response body: %s", rr.Body.String())
		}
	})

	// Test list collections
	t.Run("ListCollections", func(t *testing.T) {
		req, err := http.NewRequest("GET", "/collections", nil)
		if err != nil {
			t.Fatal(err)
		}

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
		}

		var collections []CollectionResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &collections); err != nil {
			t.Errorf("failed to unmarshal response: %v", err)
		}

		if len(collections) != 1 {
			t.Errorf("expected 1 collection, got %d", len(collections))
		}
	})

	// Test add vector
	t.Run("AddVector", func(t *testing.T) {
		vecReq := AddVectorRequest{
			ID:     "vec1",
			Values: []float32{1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9},
			Metadata: map[string]string{
				"type": "test",
			},
		}

		body, _ := json.Marshal(vecReq)
		req, err := http.NewRequest("POST", "/collections/test_collection/vectors", bytes.NewBuffer(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusCreated {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusCreated)
			t.Errorf("Response body: %s", rr.Body.String())
		}
	})

	// Test get vector
	t.Run("GetVector", func(t *testing.T) {
		req, err := http.NewRequest("GET", "/collections/test_collection/vectors/vec1", nil)
		if err != nil {
			t.Fatal(err)
		}

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			t.Errorf("Response body: %s", rr.Body.String())
		}

		var response VectorResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
			t.Errorf("failed to unmarshal response: %v", err)
		}

		if response.ID != "vec1" {
			t.Errorf("expected vector ID 'vec1', got %s", response.ID)
		}
	})

	// Test search
	t.Run("Search", func(t *testing.T) {
		// Add another vector for search
		vec2 := AddVectorRequest{
			ID:     "vec2",
			Values: make([]float32, 128),
			Metadata: map[string]string{
				"type": "test",
			},
		}
		// Initialize with different values
		for i := range vec2.Values {
			vec2.Values[i] = float32(i) * 0.01
		}

		body, _ := json.Marshal(vec2)
		req, _ := http.NewRequest("POST", "/collections/test_collection/vectors", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		server.router.ServeHTTP(httptest.NewRecorder(), req)

		// Now search
		searchReq := SearchRequest{
			Query:          make([]float32, 128),
			TopK:           2,
			IncludeVectors: false,
		}
		// Initialize query vector
		for i := range searchReq.Query {
			searchReq.Query[i] = 1.0
		}

		body, _ = json.Marshal(searchReq)
		req, err := http.NewRequest("POST", "/collections/test_collection/search", bytes.NewBuffer(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			t.Errorf("Response body: %s", rr.Body.String())
		}

		var results []SearchResult
		if err := json.Unmarshal(rr.Body.Bytes(), &results); err != nil {
			t.Errorf("failed to unmarshal response: %v", err)
		}

		if len(results) != 2 {
			t.Errorf("expected 2 results, got %d", len(results))
		}
	})

	// Test batch add
	t.Run("BatchAdd", func(t *testing.T) {
		// Create a new collection for batch test
		collection := CreateCollectionRequest{
			Name:      "batch_collection",
			Dimension: 64,
			IndexType: "flat",
			Distance:  "l2",
		}

		body, _ := json.Marshal(collection)
		req, _ := http.NewRequest("POST", "/collections", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		server.router.ServeHTTP(httptest.NewRecorder(), req)

		// Batch add vectors
		vectors := []AddVectorRequest{
			{ID: "b1", Values: make([]float32, 64)},
			{ID: "b2", Values: make([]float32, 64)},
			{ID: "b3", Values: make([]float32, 64)},
		}

		// Initialize with different values
		for i := range vectors {
			for j := range vectors[i].Values {
				vectors[i].Values[j] = float32(i*64+j) * 0.01
			}
		}

		body, _ = json.Marshal(vectors)
		req, err := http.NewRequest("POST", "/collections/batch_collection/vectors/batch", bytes.NewBuffer(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusCreated {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusCreated)
			t.Errorf("Response body: %s", rr.Body.String())
		}
	})

	// Test stats
	t.Run("Stats", func(t *testing.T) {
		req, err := http.NewRequest("GET", "/stats", nil)
		if err != nil {
			t.Fatal(err)
		}

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			t.Errorf("Response body: %s", rr.Body.String())
		}

		var stats StatsResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &stats); err != nil {
			t.Errorf("failed to unmarshal response: %v", err)
		}

		if stats.TotalCollections != 2 {
			t.Errorf("expected 2 collections, got %d", stats.TotalCollections)
		}

		if stats.TotalVectors != 5 {
			t.Errorf("expected 5 total vectors, got %d", stats.TotalVectors)
		}
	})
}

func TestValidation(t *testing.T) {
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Test invalid collection creation
	t.Run("InvalidCollection", func(t *testing.T) {
		reqBody := CreateCollectionRequest{
			Name:      "", // Invalid: empty name
			Dimension: 128,
			IndexType: "flat",
			Distance:  "cosine",
		}

		body, _ := json.Marshal(reqBody)
		req, err := http.NewRequest("POST", "/collections", bytes.NewBuffer(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusBadRequest {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusBadRequest)
		}
	})

	// Test invalid vector
	t.Run("InvalidVector", func(t *testing.T) {
		// First create a valid collection
		ctx := context.Background()
		collection := core.Collection{
			Name:      "validation_test",
			Dimension: 4,
			IndexType: "flat",
			Distance:  "cosine",
		}
		vectorStore.CreateCollection(ctx, collection)

		// Try to add vector with wrong dimension
		vecReq := AddVectorRequest{
			ID:     "invalid_vec",
			Values: []float32{1.0, 0.5}, // Wrong dimension
		}

		body, _ := json.Marshal(vecReq)
		req, err := http.NewRequest("POST", "/collections/validation_test/vectors", bytes.NewBuffer(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		server.router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusInternalServerError {
			t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusInternalServerError)
		}
	})
}
