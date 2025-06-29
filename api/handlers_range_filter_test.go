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

// TestRangeSearchHandlerWithMetadataFilter tests metadata filtering in the range search handler
func TestRangeSearchHandlerWithMetadataFilter(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "range-test-collection",
		Dimension: 2,
		IndexType: "flat",
		Distance:  "l2", // Using L2 distance for range search
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add test vectors with metadata
	testVectors := []core.Vector{
		{
			ID:     "v1",
			Values: []float32{0.0, 0.0}, // Distance 0 from origin
			Metadata: map[string]string{
				"type": "A",
				"zone": "near",
			},
		},
		{
			ID:     "v2",
			Values: []float32{1.0, 0.0}, // Distance 1 from origin
			Metadata: map[string]string{
				"type": "A",
				"zone": "near",
			},
		},
		{
			ID:     "v3",
			Values: []float32{2.0, 0.0}, // Distance 2 from origin
			Metadata: map[string]string{
				"type": "B",
				"zone": "far",
			},
		},
		{
			ID:     "v4",
			Values: []float32{3.0, 0.0}, // Distance 3 from origin
			Metadata: map[string]string{
				"type": "A",
				"zone": "far",
			},
		},
		{
			ID:     "v5",
			Values: []float32{0.5, 0.0}, // Distance 0.5 from origin
			Metadata: map[string]string{
				"type": "B",
				"zone": "near",
			},
		},
	}

	// Add vectors to the collection
	for _, vec := range testVectors {
		err := vectorStore.AddVector(ctx, "range-test-collection", vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test cases for range search with metadata filtering
	testCases := []struct {
		name          string
		rangeRequest  RangeSearchRequest
		expectedIDs   []string
		shouldNotHave []string
		description   string
	}{
		{
			name: "Range search with single filter",
			rangeRequest: RangeSearchRequest{
				Query:  []float32{0.0, 0.0}, // Search from origin
				Radius: 1.5,                  // Should include v1, v2, v5
				Filter: map[string]string{
					"type": "A",
				},
			},
			expectedIDs:   []string{"v1", "v2"}, // Only type A within radius 1.5
			shouldNotHave: []string{"v3", "v4", "v5"},
			description:   "Should only return type A vectors within radius 1.5",
		},
		{
			name: "Range search with multiple filters",
			rangeRequest: RangeSearchRequest{
				Query:  []float32{0.0, 0.0},
				Radius: 3.5, // Should include all vectors
				Filter: map[string]string{
					"type": "A",
					"zone": "far",
				},
			},
			expectedIDs:   []string{"v4"}, // Only type A in far zone
			shouldNotHave: []string{"v1", "v2", "v3", "v5"},
			description:   "Should only return type A vectors in far zone",
		},
		{
			name: "Range search with no filter",
			rangeRequest: RangeSearchRequest{
				Query:  []float32{0.0, 0.0},
				Radius: 1.5,
				Filter: nil,
			},
			expectedIDs:   []string{"v1", "v2", "v5"}, // All within radius
			shouldNotHave: []string{"v3", "v4"},
			description:   "Should return all vectors within radius when no filter",
		},
		{
			name: "Range search with limit and filter",
			rangeRequest: RangeSearchRequest{
				Query:  []float32{0.0, 0.0},
				Radius: 3.5,
				Filter: map[string]string{
					"type": "A",
				},
				Limit: 2,
			},
			expectedIDs: []string{"v1", "v2"}, // Closest 2 type A vectors
			description: "Should return only 2 closest type A vectors",
		},
		{
			name: "Range search with filter no matches",
			rangeRequest: RangeSearchRequest{
				Query:  []float32{0.0, 0.0},
				Radius: 1.0,
				Filter: map[string]string{
					"type": "C", // Non-existent type
				},
			},
			expectedIDs:   []string{},
			shouldNotHave: []string{"v1", "v2", "v3", "v4", "v5"},
			description:   "Should return empty when filter matches nothing",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Prepare request body
			reqBody, err := json.Marshal(tc.rangeRequest)
			if err != nil {
				t.Fatalf("Failed to marshal request: %v", err)
			}

			// Create HTTP request
			req, err := http.NewRequest("POST", "/collections/range-test-collection/search/range", bytes.NewBuffer(reqBody))
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
			var response RangeSearchResponse
			if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
				t.Fatalf("Failed to unmarshal response: %v", err)
			}

			// Check result count
			if len(response.Results) != len(tc.expectedIDs) {
				t.Errorf("%s: Expected %d results, got %d", tc.description, len(tc.expectedIDs), len(response.Results))
				t.Logf("Results: %+v", response.Results)
			}

			// Check that expected IDs are present
			resultIDs := make(map[string]bool)
			for _, result := range response.Results {
				resultIDs[result.ID] = true
			}

			for _, expectedID := range tc.expectedIDs {
				if !resultIDs[expectedID] {
					t.Errorf("%s: Expected result to contain ID %s, but it was not found", tc.description, expectedID)
				}
			}

			// Check that unwanted IDs are not present
			for _, unexpectedID := range tc.shouldNotHave {
				if resultIDs[unexpectedID] {
					t.Errorf("%s: Result should not contain ID %s, but it was found", tc.description, unexpectedID)
				}
			}

			// Verify that all returned results are within radius
			for _, result := range response.Results {
				if result.Score > tc.rangeRequest.Radius {
					t.Errorf("%s: Result %s has distance %f which exceeds radius %f",
						tc.description, result.ID, result.Score, tc.rangeRequest.Radius)
				}
			}

			// Verify that all returned results match the filter
			if tc.rangeRequest.Filter != nil && len(tc.rangeRequest.Filter) > 0 {
				for _, result := range response.Results {
					for filterKey, filterValue := range tc.rangeRequest.Filter {
						if result.Metadata[filterKey] != filterValue {
							t.Errorf("%s: Result %s does not match filter: expected %s=%s, got %s=%s",
								tc.description, result.ID, filterKey, filterValue, filterKey, result.Metadata[filterKey])
						}
					}
				}
			}

			// Verify ordering (closest vectors should come first)
			if len(response.Results) > 1 {
				for i := 1; i < len(response.Results); i++ {
					if response.Results[i-1].Score > response.Results[i].Score {
						t.Errorf("%s: Results not properly ordered by distance: %f > %f",
							tc.description, response.Results[i-1].Score, response.Results[i].Score)
					}
				}
			}

			// Check the Limited flag
			if tc.rangeRequest.Limit > 0 && response.Count >= tc.rangeRequest.Limit {
				if !response.Limited {
					t.Errorf("%s: Expected Limited flag to be true when limit is reached", tc.description)
				}
			}
		})
	}
}