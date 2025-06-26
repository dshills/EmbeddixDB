package index

import (
	"testing"

	"github.com/dshills/EmbeddixDB/core"
)

func TestRangeSearch(t *testing.T) {
	// Test with flat index
	t.Run("FlatIndex", func(t *testing.T) {
		testRangeSearchWithIndex(t, func(dim int, distance core.DistanceMetric) core.Index {
			return NewFlatIndex(dim, distance)
		})
	})

	// Test with HNSW index
	t.Run("HNSWIndex", func(t *testing.T) {
		testRangeSearchWithIndex(t, func(dim int, distance core.DistanceMetric) core.Index {
			config := HNSWConfig{
				M:              16,
				MMax:           16,
				ML:             1.0 / 1.442,
				EfConstruction: 200,
				EfSearch:       50,
				MaxLevels:      16,
				Seed:           42,
			}
			return NewHNSWIndex(dim, distance, config)
		})
	})
}

func testRangeSearchWithIndex(t *testing.T, createIndex func(int, core.DistanceMetric) core.Index) {
	dimension := 3
	index := createIndex(dimension, core.DistanceL2)

	// Add test vectors with known distances
	// For L2 distance: sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
	vectors := []core.Vector{
		{ID: "v1", Values: []float32{0, 0, 0}, Metadata: map[string]string{"type": "origin"}},
		{ID: "v2", Values: []float32{1, 0, 0}, Metadata: map[string]string{"type": "x-axis"}},   // Distance 1 from origin
		{ID: "v3", Values: []float32{0, 1, 0}, Metadata: map[string]string{"type": "y-axis"}},   // Distance 1 from origin
		{ID: "v4", Values: []float32{0, 0, 1}, Metadata: map[string]string{"type": "z-axis"}},   // Distance 1 from origin
		{ID: "v5", Values: []float32{1, 1, 0}, Metadata: map[string]string{"type": "xy-plane"}}, // Distance √2 ≈ 1.414 from origin
		{ID: "v6", Values: []float32{1, 1, 1}, Metadata: map[string]string{"type": "diagonal"}}, // Distance √3 ≈ 1.732 from origin
		{ID: "v7", Values: []float32{2, 0, 0}, Metadata: map[string]string{"type": "x-axis"}},   // Distance 2 from origin
		{ID: "v8", Values: []float32{3, 0, 0}, Metadata: map[string]string{"type": "x-axis"}},   // Distance 3 from origin
	}

	for _, vec := range vectors {
		err := index.Add(vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Test cases
	tests := []struct {
		name          string
		query         []float32
		radius        float32
		filter        map[string]string
		limit         int
		expectedCount int
		expectedIDs   []string
	}{
		{
			name:          "Find vectors within radius 1.0 from origin",
			query:         []float32{0, 0, 0},
			radius:        1.0,
			expectedCount: 4,
			expectedIDs:   []string{"v1", "v2", "v3", "v4"},
		},
		{
			name:          "Find vectors within radius 1.5 from origin",
			query:         []float32{0, 0, 0},
			radius:        1.5,
			expectedCount: 5,
			expectedIDs:   []string{"v1", "v2", "v3", "v4", "v5"},
		},
		{
			name:          "Find vectors within radius 2.0 from origin",
			query:         []float32{0, 0, 0},
			radius:        2.0,
			expectedCount: 7,
			expectedIDs:   []string{"v1", "v2", "v3", "v4", "v5", "v6", "v7"},
		},
		{
			name:          "Find vectors with filter",
			query:         []float32{0, 0, 0},
			radius:        3.0,
			filter:        map[string]string{"type": "x-axis"},
			expectedCount: 3,
			expectedIDs:   []string{"v2", "v7", "v8"},
		},
		{
			name:          "Find vectors with limit",
			query:         []float32{0, 0, 0},
			radius:        3.0,
			limit:         3,
			expectedCount: 3,
		},
		{
			name:          "Find vectors from different point",
			query:         []float32{1, 0, 0},
			radius:        1.0,
			expectedCount: 4,
			expectedIDs:   []string{"v1", "v2", "v5", "v7"}, // v1 is distance 1, v2 is distance 0, v5 is distance 1, v7 is distance 1
		},
		{
			name:          "No vectors within small radius",
			query:         []float32{10, 10, 10},
			radius:        0.5,
			expectedCount: 0,
			expectedIDs:   []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := index.RangeSearch(tt.query, tt.radius, tt.filter, tt.limit)
			if err != nil {
				t.Fatalf("Range search failed: %v", err)
			}

			if len(results) != tt.expectedCount {
				t.Errorf("Expected %d results, got %d", tt.expectedCount, len(results))
			}

			// Check specific IDs if provided
			if len(tt.expectedIDs) > 0 {
				foundIDs := make(map[string]bool)
				for _, res := range results {
					foundIDs[res.ID] = true
				}

				for _, expectedID := range tt.expectedIDs {
					if !foundIDs[expectedID] {
						t.Errorf("Expected to find ID %s in results", expectedID)
					}
				}
			}

			// Verify all results are within radius
			for _, res := range results {
				if res.Score > tt.radius {
					t.Errorf("Result %s has distance %f, exceeds radius %f",
						res.ID, res.Score, tt.radius)
				}
			}

			// Verify results are sorted by distance
			for i := 1; i < len(results); i++ {
				if results[i].Score < results[i-1].Score {
					t.Errorf("Results not sorted: %f < %f at position %d",
						results[i].Score, results[i-1].Score, i)
				}
			}
		})
	}
}

func TestRangeSearchWithDifferentDistanceMetrics(t *testing.T) {
	testCases := []struct {
		name      string
		distance  string
		dimension int
		vectors   []core.Vector
		query     []float32
		radius    float32
		expected  int
	}{
		{
			name:      "Cosine similarity",
			distance:  "cosine",
			dimension: 2,
			vectors: []core.Vector{
				{ID: "v1", Values: []float32{1, 0}},         // angle 0°, cosine distance 0
				{ID: "v2", Values: []float32{0.707, 0.707}}, // angle 45°, cosine distance ≈ 0.293
				{ID: "v3", Values: []float32{0, 1}},         // angle 90°, cosine distance 1
				{ID: "v4", Values: []float32{-1, 0}},        // angle 180°, cosine distance 2
			},
			query:    []float32{1, 0},
			radius:   0.5,
			expected: 2, // v1 and v2
		},
		{
			name:      "Dot product",
			distance:  "dot",
			dimension: 2,
			vectors: []core.Vector{
				{ID: "v1", Values: []float32{1, 1}},
				{ID: "v2", Values: []float32{2, 2}},
				{ID: "v3", Values: []float32{-1, -1}},
				{ID: "v4", Values: []float32{0.5, 0.5}},
			},
			query:    []float32{1, 1},
			radius:   -1.5, // For dot product, we use negative values (higher dot product = more similar)
			expected: 2,    // v1 (dot=-2) and v2 (dot=-4)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"_FlatIndex", func(t *testing.T) {
			var metric core.DistanceMetric
			switch tc.distance {
			case "cosine":
				metric = core.DistanceCosine
			case "dot":
				metric = core.DistanceDot
			case "l2":
				metric = core.DistanceL2
			}
			index := NewFlatIndex(tc.dimension, metric)
			testDistanceMetric(t, index, tc)
		})

		t.Run(tc.name+"_HNSWIndex", func(t *testing.T) {
			var metric core.DistanceMetric
			switch tc.distance {
			case "cosine":
				metric = core.DistanceCosine
			case "dot":
				metric = core.DistanceDot
			case "l2":
				metric = core.DistanceL2
			}
			config := HNSWConfig{
				M:              16,
				MMax:           16,
				ML:             1.0 / 1.442,
				EfConstruction: 200,
				EfSearch:       50,
				MaxLevels:      16,
				Seed:           42,
			}
			index := NewHNSWIndex(tc.dimension, metric, config)
			testDistanceMetric(t, index, tc)
		})
	}
}

func testDistanceMetric(t *testing.T, index core.Index, tc struct {
	name      string
	distance  string
	dimension int
	vectors   []core.Vector
	query     []float32
	radius    float32
	expected  int
}) {
	// Add vectors
	for _, vec := range tc.vectors {
		err := index.Add(vec)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}

	// Perform range search
	results, err := index.RangeSearch(tc.query, tc.radius, nil, 0)
	if err != nil {
		t.Fatalf("Range search failed: %v", err)
	}

	if len(results) != tc.expected {
		t.Errorf("Expected %d results for %s distance, got %d",
			tc.expected, tc.distance, len(results))
	}
}

func TestRangeSearchEdgeCases(t *testing.T) {
	t.Run("EmptyIndex", func(t *testing.T) {
		index := NewFlatIndex(2, core.DistanceL2)

		results, err := index.RangeSearch([]float32{0, 0}, 1.0, nil, 0)
		if err != nil {
			t.Fatalf("Range search failed: %v", err)
		}

		if len(results) != 0 {
			t.Errorf("Expected 0 results from empty index, got %d", len(results))
		}
	})

	t.Run("InvalidDimension", func(t *testing.T) {
		index := NewFlatIndex(2, core.DistanceL2)

		// Add a vector
		index.Add(core.Vector{ID: "v1", Values: []float32{1, 0}})

		// Search with wrong dimension
		_, err := index.RangeSearch([]float32{0}, 1.0, nil, 0)
		if err == nil {
			t.Error("Expected error for wrong dimension query")
		}
	})

	t.Run("ZeroRadius", func(t *testing.T) {
		index := NewFlatIndex(2, core.DistanceL2)

		// Add vectors
		index.Add(core.Vector{ID: "v1", Values: []float32{1, 0}})
		index.Add(core.Vector{ID: "v2", Values: []float32{0, 1}})

		results, err := index.RangeSearch([]float32{1, 0}, 0.0, nil, 0)
		if err != nil {
			t.Fatalf("Range search failed: %v", err)
		}

		// Should only find exact match if any
		if len(results) > 1 {
			t.Errorf("Expected at most 1 result with radius 0, got %d", len(results))
		}
	})
}
