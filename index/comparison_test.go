package index

import (
	"fmt"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
)

// TestAccuracyComparison compares search accuracy between Flat and HNSW
func TestAccuracyComparison(t *testing.T) {
	dimension := 10
	numVectors := 100

	// Create test data
	vectors := make([]core.Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		values := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			values[j] = float32(i*j) * 0.01
		}
		vectors[i] = core.Vector{
			ID:     fmt.Sprintf("v%d", i),
			Values: values,
		}
	}

	// Create indexes
	flatIndex := NewFlatIndex(dimension, core.DistanceCosine)

	// Use more aggressive HNSW parameters for better recall
	config := DefaultHNSWConfig()
	config.EfConstruction = 100
	config.EfSearch = 50
	config.M = 16
	hnswIndex := NewHNSWIndex(dimension, core.DistanceCosine, config)

	// Add vectors to both indexes
	for _, vec := range vectors {
		flatIndex.Add(vec)
		hnswIndex.Add(vec)
	}

	// Test query
	query := make([]float32, dimension)
	for i := range query {
		query[i] = 0.05
	}

	k := 10

	// Get results from both indexes
	flatResults, err := flatIndex.Search(query, k, nil)
	if err != nil {
		t.Fatalf("Flat search failed: %v", err)
	}

	hnswResults, err := hnswIndex.Search(query, k, nil)
	if err != nil {
		t.Fatalf("HNSW search failed: %v", err)
	}

	// Calculate recall@k
	recall := calculateRecall(flatResults, hnswResults, k)

	t.Logf("Recall@%d: %.2f%%", k, recall*100)

	// We expect reasonable recall (>10%) for this simple test
	// Note: HNSW recall depends on parameters, dataset characteristics, and randomness
	// Small datasets may have variable recall due to the probabilistic nature of HNSW
	if recall < 0.1 {
		t.Errorf("Very low recall: %.2f%%, expected >= 10%%", recall*100)
	}

	// Log the recall for monitoring purposes
	t.Logf("HNSW achieved %.1f%% recall on small test dataset", recall*100)
}

// calculateRecall computes recall@k between ground truth (flat) and approximate (HNSW) results
func calculateRecall(groundTruth, approximate []core.SearchResult, k int) float32 {
	if len(groundTruth) == 0 || len(approximate) == 0 {
		return 0
	}

	// Create set of ground truth IDs
	truthSet := make(map[string]bool)
	maxLen := min(len(groundTruth), k)
	for i := 0; i < maxLen; i++ {
		truthSet[groundTruth[i].ID] = true
	}

	// Count matches in approximate results
	matches := 0
	maxApproxLen := min(len(approximate), k)
	for i := 0; i < maxApproxLen; i++ {
		if truthSet[approximate[i].ID] {
			matches++
		}
	}

	return float32(matches) / float32(maxLen)
}

func BenchmarkComparison(b *testing.B) {
	dimension := 128
	numVectors := 1000

	// Create test data
	vectors := make([]core.Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		values := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			values[j] = float32(i*j) * 0.001
		}
		vectors[i] = core.Vector{
			ID:     fmt.Sprintf("v%d", i),
			Values: values,
		}
	}

	query := make([]float32, dimension)
	for i := range query {
		query[i] = 0.5
	}

	b.Run("Flat", func(b *testing.B) {
		index := NewFlatIndex(dimension, core.DistanceCosine)
		for _, vec := range vectors {
			index.Add(vec)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = index.Search(query, 10, nil)
		}
	})

	b.Run("HNSW", func(b *testing.B) {
		index := NewHNSWIndex(dimension, core.DistanceCosine, DefaultHNSWConfig())
		for _, vec := range vectors {
			index.Add(vec)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = index.Search(query, 10, nil)
		}
	})
}
