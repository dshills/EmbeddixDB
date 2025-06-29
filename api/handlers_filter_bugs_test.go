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

// TestSearchFilteringBehavior tests specific filtering behaviors and potential bugs
func TestSearchFilteringBehavior(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "filter-behavior-collection",
		Dimension: 2,
		IndexType: "flat",
		Distance:  "cosine",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add test vectors with various metadata edge cases
	testVectors := []core.Vector{
		{
			ID:     "v1",
			Values: []float32{1.0, 0.0},
			Metadata: map[string]string{
				"color": "Blue", // Capital B
				"size":  "10",
			},
		},
		{
			ID:     "v2",
			Values: []float32{0.9, 0.1},
			Metadata: map[string]string{
				"color": "blue", // lowercase b
				"size":  "10",
			},
		},
		{
			ID:     "v3",
			Values: []float32{0.8, 0.2},
			Metadata: map[string]string{
				"color": "BLUE", // All caps
				"size":  "20",
			},
		},
		{
			ID:     "v4",
			Values: []float32{0.7, 0.3},
			Metadata: map[string]string{
				"color": "blue ", // Trailing space
				"size":  "10",
			},
		},
		{
			ID:     "v5",
			Values: []float32{0.6, 0.4},
			Metadata: map[string]string{
				"color": " blue", // Leading space
				"size":  "10",
			},
		},
		{
			ID:     "v6",
			Values: []float32{0.5, 0.5},
			Metadata: map[string]string{
				"color": "light blue", // Partial match
				"size":  "10",
			},
		},
	}

	// Add vectors
	for _, vec := range testVectors {
		err := vectorStore.AddVector(ctx, "filter-behavior-collection", vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test cases to check filtering behavior
	testCases := []struct {
		name           string
		filter         map[string]string
		expectedIDs    []string
		notExpectedIDs []string
		description    string
	}{
		{
			name: "Case sensitivity - lowercase filter",
			filter: map[string]string{
				"color": "blue",
			},
			expectedIDs:    []string{"v2"}, // Only exact match
			notExpectedIDs: []string{"v1", "v3", "v4", "v5", "v6"},
			description:    "Filter should be case-sensitive and match only exact 'blue'",
		},
		{
			name: "Case sensitivity - uppercase filter",
			filter: map[string]string{
				"color": "Blue",
			},
			expectedIDs:    []string{"v1"}, // Only exact match
			notExpectedIDs: []string{"v2", "v3", "v4", "v5", "v6"},
			description:    "Filter should be case-sensitive and match only exact 'Blue'",
		},
		{
			name: "Whitespace handling - no spaces",
			filter: map[string]string{
				"color": "blue",
			},
			expectedIDs:    []string{"v2"}, // Should not match "blue " or " blue"
			notExpectedIDs: []string{"v1", "v3", "v4", "v5", "v6"},
			description:    "Filter should not match values with extra whitespace",
		},
		{
			name: "Partial matching - should not match",
			filter: map[string]string{
				"color": "blue",
			},
			expectedIDs:    []string{"v2"}, // Should not match "light blue"
			notExpectedIDs: []string{"v1", "v3", "v4", "v5", "v6"},
			description:    "Filter should not perform partial matching",
		},
		{
			name: "Multiple filters - all must match",
			filter: map[string]string{
				"color": "blue",
				"size":  "10",
			},
			expectedIDs:    []string{"v2"}, // Only v2 has both exact matches
			notExpectedIDs: []string{"v1", "v3", "v4", "v5", "v6"},
			description:    "All filter conditions must match exactly",
		},
		{
			name: "Numeric string comparison",
			filter: map[string]string{
				"size": "10",
			},
			expectedIDs:    []string{"v1", "v2", "v4", "v5", "v6"}, // All with size "10"
			notExpectedIDs: []string{"v3"},                         // Has size "20"
			description:    "Numeric strings should be compared as strings, not numbers",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			searchReq := SearchRequest{
				Query:  []float32{1.0, 0.0},
				TopK:   10,
				Filter: tc.filter,
			}

			reqBody, _ := json.Marshal(searchReq)
			req, _ := http.NewRequest("POST", "/collections/filter-behavior-collection/search", bytes.NewBuffer(reqBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()
			server.router.ServeHTTP(rr, req)

			if status := rr.Code; status != http.StatusOK {
				t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}

			var results []SearchResult
			json.Unmarshal(rr.Body.Bytes(), &results)

			// Check result count
			if len(results) != len(tc.expectedIDs) {
				t.Errorf("%s: Expected %d results, got %d", tc.description, len(tc.expectedIDs), len(results))
				t.Logf("Results: %+v", results)
			}

			// Create result ID map
			resultIDs := make(map[string]bool)
			for _, result := range results {
				resultIDs[result.ID] = true
			}

			// Check expected IDs
			for _, expectedID := range tc.expectedIDs {
				if !resultIDs[expectedID] {
					t.Errorf("%s: Expected to find ID %s", tc.description, expectedID)
				}
			}

			// Check unexpected IDs
			for _, unexpectedID := range tc.notExpectedIDs {
				if resultIDs[unexpectedID] {
					t.Errorf("%s: Should not find ID %s", tc.description, unexpectedID)
				}
			}
		})
	}
}

// TestFilteringWithMissingMetadata tests how filtering handles vectors with missing metadata keys
func TestFilteringWithMissingMetadata(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "missing-metadata-collection",
		Dimension: 2,
		IndexType: "hnsw",
		Distance:  "cosine",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add vectors with various metadata configurations
	testVectors := []core.Vector{
		{
			ID:     "v1",
			Values: []float32{1.0, 0.0},
			Metadata: map[string]string{
				"color": "red",
				"size":  "large",
				"brand": "nike",
			},
		},
		{
			ID:     "v2",
			Values: []float32{0.9, 0.1},
			Metadata: map[string]string{
				"color": "red",
				"size":  "small",
				// Missing "brand" key
			},
		},
		{
			ID:     "v3",
			Values: []float32{0.8, 0.2},
			Metadata: map[string]string{
				"color": "red",
				// Missing both "size" and "brand" keys
			},
		},
		{
			ID:       "v4",
			Values:   []float32{0.7, 0.3},
			Metadata: nil, // No metadata at all
		},
		{
			ID:       "v5",
			Values:   []float32{0.6, 0.4},
			Metadata: map[string]string{}, // Empty metadata map
		},
	}

	// Add vectors
	for _, vec := range testVectors {
		err := vectorStore.AddVector(ctx, "missing-metadata-collection", vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test filtering behavior with missing keys
	testCases := []struct {
		name        string
		filter      map[string]string
		expectedIDs []string
		description string
	}{
		{
			name: "Filter by existing key",
			filter: map[string]string{
				"color": "red",
			},
			expectedIDs: []string{"v1", "v2", "v3"},
			description: "Should match all vectors with color=red, regardless of other missing keys",
		},
		{
			name: "Filter by sometimes-missing key",
			filter: map[string]string{
				"brand": "nike",
			},
			expectedIDs: []string{"v1"},
			description: "Should only match vectors that have the brand key with value 'nike'",
		},
		{
			name: "Filter by multiple keys where some are missing",
			filter: map[string]string{
				"color": "red",
				"brand": "nike",
			},
			expectedIDs: []string{"v1"},
			description: "Should only match vectors that have ALL filter keys with matching values",
		},
		{
			name: "Filter that matches no vectors",
			filter: map[string]string{
				"nonexistent": "value",
			},
			expectedIDs: []string{},
			description: "Should return empty when filtering by a key that no vector has",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			searchReq := SearchRequest{
				Query:  []float32{1.0, 0.0},
				TopK:   10,
				Filter: tc.filter,
			}

			reqBody, _ := json.Marshal(searchReq)
			req, _ := http.NewRequest("POST", "/collections/missing-metadata-collection/search", bytes.NewBuffer(reqBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()
			server.router.ServeHTTP(rr, req)

			var results []SearchResult
			json.Unmarshal(rr.Body.Bytes(), &results)

			// Check result count
			if len(results) != len(tc.expectedIDs) {
				t.Errorf("%s: Expected %d results, got %d", tc.description, len(tc.expectedIDs), len(results))
				t.Logf("Filter: %+v", tc.filter)
				t.Logf("Results: %+v", results)
			}

			// Verify expected IDs
			resultIDs := make(map[string]bool)
			for _, result := range results {
				resultIDs[result.ID] = true
			}

			for _, expectedID := range tc.expectedIDs {
				if !resultIDs[expectedID] {
					t.Errorf("%s: Expected to find ID %s", tc.description, expectedID)
				}
			}
		})
	}
}