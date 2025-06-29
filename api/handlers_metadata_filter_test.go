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

// TestSearchVectorsHandlerWithMetadataFilter tests metadata filtering in the search handler
func TestSearchVectorsHandlerWithMetadataFilter(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "test-collection",
		Dimension: 3,
		IndexType: "flat",
		Distance:  "cosine",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add test vectors with different metadata
	testVectors := []core.Vector{
		{
			ID:     "v1",
			Values: []float32{1.0, 0.0, 0.0},
			Metadata: map[string]string{
				"category": "electronics",
				"price":    "high",
				"brand":    "apple",
			},
		},
		{
			ID:     "v2",
			Values: []float32{0.9, 0.1, 0.0},
			Metadata: map[string]string{
				"category": "electronics",
				"price":    "low",
				"brand":    "samsung",
			},
		},
		{
			ID:     "v3",
			Values: []float32{0.0, 1.0, 0.0},
			Metadata: map[string]string{
				"category": "clothing",
				"price":    "high",
				"brand":    "nike",
			},
		},
		{
			ID:     "v4",
			Values: []float32{0.8, 0.2, 0.0},
			Metadata: map[string]string{
				"category": "electronics",
				"price":    "medium",
				"brand":    "sony",
			},
		},
	}

	// Add vectors to the collection
	for _, vec := range testVectors {
		err := vectorStore.AddVector(ctx, "test-collection", vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test cases for metadata filtering
	testCases := []struct {
		name           string
		searchRequest  SearchRequest
		expectedIDs    []string
		expectedCount  int
		shouldContain  []string
		shouldNotHave  []string
	}{
		{
			name: "Filter by single metadata field - category",
			searchRequest: SearchRequest{
				Query: []float32{1.0, 0.0, 0.0},
				TopK:  10,
				Filter: map[string]string{
					"category": "electronics",
				},
			},
			expectedCount: 3,
			shouldContain: []string{"v1", "v2", "v4"},
			shouldNotHave: []string{"v3"},
		},
		{
			name: "Filter by multiple metadata fields",
			searchRequest: SearchRequest{
				Query: []float32{1.0, 0.0, 0.0},
				TopK:  10,
				Filter: map[string]string{
					"category": "electronics",
					"price":    "high",
				},
			},
			expectedCount: 1,
			shouldContain: []string{"v1"},
			shouldNotHave: []string{"v2", "v3", "v4"},
		},
		{
			name: "Filter with no matches",
			searchRequest: SearchRequest{
				Query: []float32{1.0, 0.0, 0.0},
				TopK:  10,
				Filter: map[string]string{
					"category": "furniture",
				},
			},
			expectedCount: 0,
			shouldContain: []string{},
			shouldNotHave: []string{"v1", "v2", "v3", "v4"},
		},
		{
			name: "Filter by brand",
			searchRequest: SearchRequest{
				Query: []float32{1.0, 0.0, 0.0},
				TopK:  10,
				Filter: map[string]string{
					"brand": "apple",
				},
			},
			expectedCount: 1,
			shouldContain: []string{"v1"},
			shouldNotHave: []string{"v2", "v3", "v4"},
		},
		{
			name: "No filter - should return all vectors",
			searchRequest: SearchRequest{
				Query:  []float32{1.0, 0.0, 0.0},
				TopK:   10,
				Filter: nil,
			},
			expectedCount: 4,
			shouldContain: []string{"v1", "v2", "v3", "v4"},
			shouldNotHave: []string{},
		},
		{
			name: "Empty filter - should return all vectors",
			searchRequest: SearchRequest{
				Query:  []float32{1.0, 0.0, 0.0},
				TopK:   10,
				Filter: map[string]string{},
			},
			expectedCount: 4,
			shouldContain: []string{"v1", "v2", "v3", "v4"},
			shouldNotHave: []string{},
		},
		{
			name: "Filter with non-existent metadata key",
			searchRequest: SearchRequest{
				Query: []float32{1.0, 0.0, 0.0},
				TopK:  10,
				Filter: map[string]string{
					"non_existent_key": "value",
				},
			},
			expectedCount: 0,
			shouldContain: []string{},
			shouldNotHave: []string{"v1", "v2", "v3", "v4"},
		},
		{
			name: "TopK limit with filter",
			searchRequest: SearchRequest{
				Query: []float32{1.0, 0.0, 0.0},
				TopK:  2,
				Filter: map[string]string{
					"category": "electronics",
				},
			},
			expectedCount: 2,
			// Should return the 2 closest electronics items
			shouldContain: []string{"v1"}, // v1 is closest to query
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Prepare request body
			reqBody, err := json.Marshal(tc.searchRequest)
			if err != nil {
				t.Fatalf("Failed to marshal request: %v", err)
			}

			// Create HTTP request
			req, err := http.NewRequest("POST", "/collections/test-collection/search", bytes.NewBuffer(reqBody))
			if err != nil {
				t.Fatalf("Failed to create request: %v", err)
			}
			req.Header.Set("Content-Type", "application/json")

			// Record response
			rr := httptest.NewRecorder()
			server.router.ServeHTTP(rr, req)

			// Check status code
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
				t.Logf("Response body: %s", rr.Body.String())
			}

			// Parse response
			var results []SearchResult
			if err := json.Unmarshal(rr.Body.Bytes(), &results); err != nil {
				t.Fatalf("Failed to unmarshal response: %v", err)
			}

			// Check result count
			if len(results) != tc.expectedCount {
				t.Errorf("Expected %d results, got %d", tc.expectedCount, len(results))
				t.Logf("Results: %+v", results)
			}

			// Check that expected IDs are present
			resultIDs := make(map[string]bool)
			for _, result := range results {
				resultIDs[result.ID] = true
			}

			for _, expectedID := range tc.shouldContain {
				if !resultIDs[expectedID] {
					t.Errorf("Expected result to contain ID %s, but it was not found", expectedID)
				}
			}

			// Check that unwanted IDs are not present
			for _, unexpectedID := range tc.shouldNotHave {
				if resultIDs[unexpectedID] {
					t.Errorf("Result should not contain ID %s, but it was found", unexpectedID)
				}
			}

			// Verify that all returned results match the filter
			if tc.searchRequest.Filter != nil && len(tc.searchRequest.Filter) > 0 {
				for _, result := range results {
					for filterKey, filterValue := range tc.searchRequest.Filter {
						if result.Metadata[filterKey] != filterValue {
							t.Errorf("Result %s does not match filter: expected %s=%s, got %s=%s",
								result.ID, filterKey, filterValue, filterKey, result.Metadata[filterKey])
						}
					}
				}
			}

			// Verify ordering (closest vectors should come first)
			if len(results) > 1 {
				for i := 1; i < len(results); i++ {
					if results[i-1].Score > results[i].Score {
						t.Errorf("Results not properly ordered by distance: %f > %f", 
							results[i-1].Score, results[i].Score)
					}
				}
			}
		})
	}
}

// TestSearchVectorsHandlerFilterEdgeCases tests edge cases for metadata filtering
func TestSearchVectorsHandlerFilterEdgeCases(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "edge-test-collection",
		Dimension: 2,
		IndexType: "hnsw",
		Distance:  "l2",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add test vectors with edge case metadata
	testVectors := []core.Vector{
		{
			ID:     "v1",
			Values: []float32{1.0, 0.0},
			Metadata: map[string]string{
				"empty_value": "",
				"spaces":      "value with spaces",
				"special":     "value@with#special$chars",
			},
		},
		{
			ID:       "v2",
			Values:   []float32{0.0, 1.0},
			Metadata: nil, // No metadata
		},
		{
			ID:       "v3",
			Values:   []float32{0.5, 0.5},
			Metadata: map[string]string{}, // Empty metadata
		},
		{
			ID:     "v4",
			Values: []float32{0.7, 0.3},
			Metadata: map[string]string{
				"unicode": "こんにちは",
				"number":  "123",
				"bool":    "true",
			},
		},
	}

	// Add vectors
	for _, vec := range testVectors {
		err := vectorStore.AddVector(ctx, "edge-test-collection", vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test edge cases
	testCases := []struct {
		name          string
		filter        map[string]string
		shouldMatch   []string
		shouldNotMatch []string
	}{
		{
			name: "Filter by empty value",
			filter: map[string]string{
				"empty_value": "",
			},
			shouldMatch:   []string{"v1"},
			shouldNotMatch: []string{"v2", "v3", "v4"},
		},
		{
			name: "Filter by value with spaces",
			filter: map[string]string{
				"spaces": "value with spaces",
			},
			shouldMatch:   []string{"v1"},
			shouldNotMatch: []string{"v2", "v3", "v4"},
		},
		{
			name: "Filter by special characters",
			filter: map[string]string{
				"special": "value@with#special$chars",
			},
			shouldMatch:   []string{"v1"},
			shouldNotMatch: []string{"v2", "v3", "v4"},
		},
		{
			name: "Filter by unicode",
			filter: map[string]string{
				"unicode": "こんにちは",
			},
			shouldMatch:   []string{"v4"},
			shouldNotMatch: []string{"v1", "v2", "v3"},
		},
		{
			name: "Filter vectors with nil metadata",
			filter: map[string]string{
				"any_key": "any_value",
			},
			shouldMatch:   []string{},
			shouldNotMatch: []string{"v1", "v2", "v3", "v4"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			searchReq := SearchRequest{
				Query:  []float32{0.5, 0.5},
				TopK:   10,
				Filter: tc.filter,
			}

			reqBody, _ := json.Marshal(searchReq)
			req, _ := http.NewRequest("POST", "/collections/edge-test-collection/search", bytes.NewBuffer(reqBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()
			server.router.ServeHTTP(rr, req)

			var results []SearchResult
			json.Unmarshal(rr.Body.Bytes(), &results)

			resultIDs := make(map[string]bool)
			for _, result := range results {
				resultIDs[result.ID] = true
			}

			// Check expected matches
			for _, expectedID := range tc.shouldMatch {
				if !resultIDs[expectedID] {
					t.Errorf("Expected to match ID %s, but it was not found", expectedID)
				}
			}

			// Check unexpected matches
			for _, unexpectedID := range tc.shouldNotMatch {
				if resultIDs[unexpectedID] {
					t.Errorf("Should not match ID %s, but it was found", unexpectedID)
				}
			}
		})
	}
}