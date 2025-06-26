package index

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

func TestHierarchicalHNSW_Basic(t *testing.T) {
	dimension := 128
	config := DefaultHierarchicalConfig(dimension)

	// Create hierarchical index
	index := NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Test basic properties
	if index.Size() != 0 {
		t.Errorf("Expected empty index to have size 0, got %d", index.Size())
	}

	if index.Type() != "hierarchical_hnsw" {
		t.Errorf("Expected type 'hierarchical_hnsw', got %s", index.Type())
	}
}

func TestHierarchicalHNSW_AddAndSearch(t *testing.T) {
	dimension := 64
	config := DefaultHierarchicalConfig(dimension)
	config.NumFineClusters = 4 // Small number for testing

	index := NewHierarchicalHNSW(dimension, core.DistanceL2, config)

	// Add test vectors
	numVectors := 100
	vectors := generateTestVectors(numVectors, dimension)

	for _, vector := range vectors {
		if err := index.Add(vector); err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Verify size
	if index.Size() != numVectors {
		t.Errorf("Expected size %d, got %d", numVectors, index.Size())
	}

	// Test search
	query := vectors[0].Values
	k := 5

	results, err := index.Search(query, k, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != k {
		t.Errorf("Expected %d results, got %d", k, len(results))
	}

	// First result should be the query vector itself (distance 0)
	if results[0].Score > 0.01 { // Small tolerance for floating point
		t.Errorf("Expected first result to have distance ~0, got %f", results[0].Score)
	}
}

func TestHierarchicalHNSW_RangeSearch(t *testing.T) {
	dimension := 32
	config := DefaultHierarchicalConfig(dimension)
	config.NumFineClusters = 2

	index := NewHierarchicalHNSW(dimension, core.DistanceL2, config)

	// Add test vectors
	vectors := generateTestVectors(50, dimension)
	for _, vector := range vectors {
		index.Add(vector)
	}

	// Range search
	query := vectors[0].Values
	radius := float32(1.0)
	limit := 10

	results, err := index.RangeSearch(query, radius, nil, limit)
	if err != nil {
		t.Fatalf("Range search failed: %v", err)
	}

	// Verify all results are within radius
	for _, result := range results {
		if result.Score > radius {
			t.Errorf("Result distance %f exceeds radius %f", result.Score, radius)
		}
	}

	// Verify limit is respected
	if len(results) > limit {
		t.Errorf("Expected at most %d results, got %d", limit, len(results))
	}
}

func TestHierarchicalHNSW_Delete(t *testing.T) {
	dimension := 32
	config := DefaultHierarchicalConfig(dimension)

	index := NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Add vectors
	vectors := generateTestVectors(20, dimension)
	for _, vector := range vectors {
		index.Add(vector)
	}

	initialSize := index.Size()

	// Delete a vector
	targetID := vectors[5].ID
	err := index.Delete(targetID)
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	// Verify size decreased
	if index.Size() != initialSize-1 {
		t.Errorf("Expected size %d after deletion, got %d", initialSize-1, index.Size())
	}

	// Verify vector is not found in search
	query := vectors[5].Values
	results, err := index.Search(query, 10, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Check that deleted vector is not in results
	for _, result := range results {
		if result.ID == targetID {
			t.Errorf("Deleted vector %s found in search results", targetID)
		}
	}

	// Test deleting non-existent vector
	err = index.Delete("non-existent-id")
	if err == nil {
		t.Error("Expected error when deleting non-existent vector")
	}
}

func TestHierarchicalHNSW_AdaptiveRouting(t *testing.T) {
	dimension := 32
	config := DefaultHierarchicalConfig(dimension)
	config.AdaptiveRouting = true
	config.NumFineClusters = 3

	index := NewHierarchicalHNSW(dimension, core.DistanceL2, config)

	// Add many vectors to test load balancing
	vectors := generateTestVectors(60, dimension)
	for _, vector := range vectors {
		index.Add(vector)
	}

	// Check cluster distribution
	clusterSizes := make(map[int]int)
	for _, clusterID := range index.vectorRouting {
		clusterSizes[clusterID]++
	}

	// Verify all clusters are used
	if len(clusterSizes) != config.NumFineClusters {
		t.Errorf("Expected %d clusters to be used, got %d", config.NumFineClusters, len(clusterSizes))
	}

	// Check for reasonable load balancing (no cluster should be completely empty)
	for clusterID, size := range clusterSizes {
		if size == 0 {
			t.Errorf("Cluster %d is empty", clusterID)
		}
	}
}

func TestHierarchicalHNSW_QualityMonitoring(t *testing.T) {
	dimension := 32
	config := DefaultHierarchicalConfig(dimension)
	config.EnableQualityMonitoring = true

	index := NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Add vectors
	vectors := generateTestVectors(50, dimension)
	for _, vector := range vectors {
		index.Add(vector)
	}

	// Perform searches to generate quality metrics
	for i := 0; i < 10; i++ {
		query := vectors[i].Values
		_, err := index.Search(query, 5, nil)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}

	// Check that quality metrics were recorded
	if len(index.qualityMonitor.searchQueries) == 0 {
		t.Error("Expected quality metrics to be recorded")
	}

	// Verify quality metrics structure
	for _, metric := range index.qualityMonitor.searchQueries {
		if metric.Timestamp == 0 {
			t.Error("Quality metric should have timestamp")
		}
		if metric.ClusterHits < 0 || metric.ClusterHits > config.NumFineClusters {
			t.Errorf("Invalid cluster hits: %d", metric.ClusterHits)
		}
	}
}

func TestHierarchicalHNSW_CoarseRepresentation(t *testing.T) {
	dimension := 64
	config := DefaultHierarchicalConfig(dimension)
	config.CoarseDimension = 16

	index := NewHierarchicalHNSW(dimension, core.DistanceL2, config)

	// Test coarse representation creation
	vector := core.Vector{
		ID:     "test-vector",
		Values: make([]float32, dimension),
	}

	// Fill with random values
	for i := range vector.Values {
		vector.Values[i] = rand.Float32()
	}

	coarseVector, err := index.createCoarseRepresentation(vector)
	if err != nil {
		t.Fatalf("Failed to create coarse representation: %v", err)
	}

	// Verify dimensions
	if len(coarseVector.Values) != config.CoarseDimension {
		t.Errorf("Expected coarse dimension %d, got %d", config.CoarseDimension, len(coarseVector.Values))
	}

	// Verify values are preserved (first N dimensions)
	for i := 0; i < config.CoarseDimension; i++ {
		if coarseVector.Values[i] != vector.Values[i] {
			t.Errorf("Coarse value mismatch at index %d: expected %f, got %f",
				i, vector.Values[i], coarseVector.Values[i])
		}
	}
}

func TestHierarchicalHNSW_Configuration(t *testing.T) {
	dimension := 128

	// Test default configuration
	config := DefaultHierarchicalConfig(dimension)

	if config.CoarseDimension <= 0 {
		t.Error("Coarse dimension should be positive")
	}

	if config.NumFineClusters <= 0 {
		t.Error("Number of fine clusters should be positive")
	}

	if config.RoutingOverlap < 0 || config.RoutingOverlap > 1 {
		t.Error("Routing overlap should be between 0 and 1")
	}

	if config.QualityThreshold < 0 || config.QualityThreshold > 1 {
		t.Error("Quality threshold should be between 0 and 1")
	}

	// Test custom configuration
	customConfig := HierarchicalConfig{
		CoarseDimension:         32,
		NumFineClusters:         8,
		RoutingOverlap:          0.2,
		AdaptiveRouting:         false,
		EnableQualityMonitoring: false,
	}

	index := NewHierarchicalHNSW(dimension, core.DistanceDot, customConfig)

	if len(index.fineIndexes) != customConfig.NumFineClusters {
		t.Errorf("Expected %d fine indexes, got %d", customConfig.NumFineClusters, len(index.fineIndexes))
	}
}

// Benchmark tests

func BenchmarkHierarchicalHNSW_Add(b *testing.B) {
	dimension := 128
	config := DefaultHierarchicalConfig(dimension)
	index := NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	vectors := generateTestVectors(b.N, dimension)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index.Add(vectors[i])
	}
}

func BenchmarkHierarchicalHNSW_Search(b *testing.B) {
	dimension := 128
	config := DefaultHierarchicalConfig(dimension)
	index := NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

	// Pre-populate index
	vectors := generateTestVectors(10000, dimension)
	for _, vector := range vectors {
		index.Add(vector)
	}

	// Prepare queries
	queries := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		queries[i] = vectors[i%len(vectors)].Values
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index.Search(queries[i], 10, nil)
	}
}

func BenchmarkHierarchicalHNSW_vs_HNSW(b *testing.B) {
	dimension := 128
	numVectors := 10000

	// Generate test data
	vectors := generateTestVectors(numVectors, dimension)
	query := vectors[0].Values

	b.Run("HierarchicalHNSW", func(b *testing.B) {
		config := DefaultHierarchicalConfig(dimension)
		index := NewHierarchicalHNSW(dimension, core.DistanceCosine, config)

		// Add vectors
		for _, vector := range vectors {
			index.Add(vector)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			index.Search(query, 10, nil)
		}
	})

	b.Run("StandardHNSW", func(b *testing.B) {
		config := DefaultHNSWConfig()
		index := NewHNSWIndex(dimension, core.DistanceCosine, config)

		// Add vectors
		for _, vector := range vectors {
			index.Add(vector)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			index.Search(query, 10, nil)
		}
	})
}

// Helper functions

func generateTestVectors(count, dimension int) []core.Vector {
	rand.Seed(time.Now().UnixNano())
	vectors := make([]core.Vector, count)

	for i := 0; i < count; i++ {
		values := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			values[j] = rand.Float32()*2 - 1 // Random values between -1 and 1
		}

		vectors[i] = core.Vector{
			ID:     generateID(i),
			Values: values,
			Metadata: map[string]string{
				"index":     string(rune(i)),
				"test_type": "hierarchical_hnsw",
			},
		}
	}

	return vectors
}

func generateID(index int) string {
	return fmt.Sprintf("vector-%d", index)
}
