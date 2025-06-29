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

// TestSearchFilterSecurity tests that filter values are handled safely
func TestSearchFilterSecurity(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "security-test-collection",
		Dimension: 2,
		IndexType: "flat",
		Distance:  "cosine",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add test vectors with metadata
	testVectors := []core.Vector{
		{
			ID:     "v1",
			Values: []float32{1.0, 0.0},
			Metadata: map[string]string{
				"tag":      "admin",
				"query":    "SELECT * FROM users",
				"script":   "<script>alert('xss')</script>",
				"special":  "'; DROP TABLE vectors; --",
				"wildcard": "test*",
			},
		},
		{
			ID:     "v2",
			Values: []float32{0.9, 0.1},
			Metadata: map[string]string{
				"tag":      "user",
				"query":    "normal text",
				"script":   "safe text",
				"special":  "normal",
				"wildcard": "test",
			},
		},
	}

	// Add vectors
	for _, vec := range testVectors {
		err := vectorStore.AddVector(ctx, "security-test-collection", vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test cases for potentially malicious filter values
	testCases := []struct {
		name           string
		filter         map[string]string
		expectedCount  int
		expectedIDs    []string
		description    string
	}{
		{
			name: "SQL injection attempt in filter value",
			filter: map[string]string{
				"tag": "admin'; DROP TABLE vectors; --",
			},
			expectedCount: 0,
			expectedIDs:   []string{},
			description:   "Should not match anything - filter should be treated as literal string",
		},
		{
			name: "Exact match for SQL-like content",
			filter: map[string]string{
				"query": "SELECT * FROM users",
			},
			expectedCount: 1,
			expectedIDs:   []string{"v1"},
			description:   "Should match exact SQL string in metadata",
		},
		{
			name: "XSS attempt in filter",
			filter: map[string]string{
				"script": "<script>alert('xss')</script>",
			},
			expectedCount: 1,
			expectedIDs:   []string{"v1"},
			description:   "Should match exact script tag string",
		},
		{
			name: "Special characters in filter",
			filter: map[string]string{
				"special": "'; DROP TABLE vectors; --",
			},
			expectedCount: 1,
			expectedIDs:   []string{"v1"},
			description:   "Should match exact string with special characters",
		},
		{
			name: "Wildcard attempt",
			filter: map[string]string{
				"wildcard": "test*",
			},
			expectedCount: 1,
			expectedIDs:   []string{"v1"},
			description:   "Wildcard should not work - exact match only",
		},
		{
			name: "Regex attempt",
			filter: map[string]string{
				"tag": "admin|user",
			},
			expectedCount: 0,
			expectedIDs:   []string{},
			description:   "Regex should not work - exact match only",
		},
		{
			name: "Null byte injection",
			filter: map[string]string{
				"tag": "admin\x00extra",
			},
			expectedCount: 0,
			expectedIDs:   []string{},
			description:   "Null bytes should not cause issues",
		},
		{
			name: "Very long filter value",
			filter: map[string]string{
				"tag": string(make([]byte, 10000)), // 10KB string
			},
			expectedCount: 0,
			expectedIDs:   []string{},
			description:   "Large filter values should be handled safely",
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
			req, _ := http.NewRequest("POST", "/collections/security-test-collection/search", bytes.NewBuffer(reqBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()
			server.router.ServeHTTP(rr, req)

			// Should always return 200 OK regardless of filter content
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
				t.Logf("Response: %s", rr.Body.String())
			}

			var results []SearchResult
			if err := json.Unmarshal(rr.Body.Bytes(), &results); err != nil {
				t.Fatalf("Failed to unmarshal response: %v", err)
			}

			// Check result count
			if len(results) != tc.expectedCount {
				t.Errorf("%s: Expected %d results, got %d", tc.description, tc.expectedCount, len(results))
			}

			// Check expected IDs
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

// TestFilterKeyInjection tests that filter keys are handled safely
func TestFilterKeyInjection(t *testing.T) {
	// Create test server
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)

	config := DefaultServerConfig()
	server := NewServer(vectorStore, config)

	// Create a test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "key-injection-collection",
		Dimension: 2,
		IndexType: "flat",
		Distance:  "cosine",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add a test vector with various metadata keys
	vector := core.Vector{
		ID:     "v1",
		Values: []float32{1.0, 0.0},
		Metadata: map[string]string{
			"normal_key":                     "value1",
			"key.with.dots":                  "value2",
			"key[with]brackets":              "value3",
			"key with spaces":                "value4",
			"key\nwith\nnewlines":            "value5",
			"key\twith\ttabs":                "value6",
			"key'with'quotes":                "value7",
			`key"with"doublequotes`:          "value8",
			"key`with`backticks":             "value9",
			"key$with$special$chars":         "value10",
			"key\\with\\backslashes":         "value11",
			"key/with/slashes":               "value12",
			"key:with:colons":                "value13",
			"key;with;semicolons":            "value14",
			"key=with=equals":                "value15",
			"key?with?questions":             "value16",
			"key#with#hashes":                "value17",
			"key@with@ats":                   "value18",
			"key!with!exclamations":          "value19",
			"key%with%percents":              "value20",
			"key^with^carets":                "value21",
			"key&with&ampersands":            "value22",
			"key*with*asterisks":             "value23",
			"key(with)parentheses":           "value24",
			"key{with}braces":                "value25",
			"key|with|pipes":                 "value26",
			"key~with~tildes":                "value27",
			"key+with+plus":                  "value28",
			"key-with-dashes":                "value29",
			"key_with_underscores":           "value30",
			"key<with>angles":                "value31",
			"key,with,commas":                "value32",
			"🔑emoji":                        "value33",
			"кириллица":                      "value34",
			"中文":                            "value35",
			"العربية":                        "value36",
			"हिन्दी":                         "value37",
			string([]byte{0x00, 0x01, 0x02}): "binary_key", // Binary data in key
		},
	}

	err = vectorStore.AddVector(ctx, "key-injection-collection", vector)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Test that various special keys work correctly
	testCases := []struct {
		name        string
		filterKey   string
		filterValue string
		shouldMatch bool
		description string
	}{
		{
			name:        "Normal key",
			filterKey:   "normal_key",
			filterValue: "value1",
			shouldMatch: true,
			description: "Normal alphanumeric key should work",
		},
		{
			name:        "Key with dots",
			filterKey:   "key.with.dots",
			filterValue: "value2",
			shouldMatch: true,
			description: "Keys with dots should work",
		},
		{
			name:        "Key with brackets",
			filterKey:   "key[with]brackets",
			filterValue: "value3",
			shouldMatch: true,
			description: "Keys with brackets should work",
		},
		{
			name:        "Key with spaces",
			filterKey:   "key with spaces",
			filterValue: "value4",
			shouldMatch: true,
			description: "Keys with spaces should work",
		},
		{
			name:        "Key with newlines",
			filterKey:   "key\nwith\nnewlines",
			filterValue: "value5",
			shouldMatch: true,
			description: "Keys with newlines should work",
		},
		{
			name:        "Key with single quotes",
			filterKey:   "key'with'quotes",
			filterValue: "value7",
			shouldMatch: true,
			description: "Keys with single quotes should work",
		},
		{
			name:        "Key with double quotes",
			filterKey:   `key"with"doublequotes`,
			filterValue: "value8",
			shouldMatch: true,
			description: "Keys with double quotes should work",
		},
		{
			name:        "Key with special chars",
			filterKey:   "key$with$special$chars",
			filterValue: "value10",
			shouldMatch: true,
			description: "Keys with special characters should work",
		},
		{
			name:        "Unicode emoji key",
			filterKey:   "🔑emoji",
			filterValue: "value33",
			shouldMatch: true,
			description: "Unicode emoji keys should work",
		},
		{
			name:        "Cyrillic key",
			filterKey:   "кириллица",
			filterValue: "value34",
			shouldMatch: true,
			description: "Cyrillic keys should work",
		},
		{
			name:        "Chinese key",
			filterKey:   "中文",
			filterValue: "value35",
			shouldMatch: true,
			description: "Chinese keys should work",
		},
		{
			name:        "Non-existent key",
			filterKey:   "non_existent_key",
			filterValue: "any_value",
			shouldMatch: false,
			description: "Non-existent keys should not match",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			searchReq := SearchRequest{
				Query: []float32{1.0, 0.0},
				TopK:  10,
				Filter: map[string]string{
					tc.filterKey: tc.filterValue,
				},
			}

			reqBody, _ := json.Marshal(searchReq)
			req, _ := http.NewRequest("POST", "/collections/key-injection-collection/search", bytes.NewBuffer(reqBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()
			server.router.ServeHTTP(rr, req)

			if status := rr.Code; status != http.StatusOK {
				t.Errorf("Handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}

			var results []SearchResult
			json.Unmarshal(rr.Body.Bytes(), &results)

			if tc.shouldMatch {
				if len(results) != 1 {
					t.Errorf("%s: Expected 1 result, got %d", tc.description, len(results))
				}
			} else {
				if len(results) != 0 {
					t.Errorf("%s: Expected 0 results, got %d", tc.description, len(results))
				}
			}
		})
	}
}