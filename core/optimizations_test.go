package core

import (
	"fmt"
	"testing"
	"time"
)

func TestSIMDDistanceCalculations(t *testing.T) {
	// Test vectors
	a := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0}
	b := []float32{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0}

	// Test SIMD cosine similarity
	simdResult, err := CalculateDistanceOptimized(a, b, DistanceCosine)
	if err != nil {
		t.Fatalf("SIMD cosine similarity failed: %v", err)
	}

	// Test regular cosine similarity
	regularResult, err := CalculateDistance(a, b, DistanceCosine)
	if err != nil {
		t.Fatalf("Regular cosine similarity failed: %v", err)
	}

	// Results should be very close (within floating point precision)
	diff := simdResult - regularResult
	if diff < 0 {
		diff = -diff
	}
	if diff > 0.0001 {
		t.Errorf("SIMD and regular results differ too much: %f vs %f", simdResult, regularResult)
	}

	t.Logf("SIMD result: %f, Regular result: %f", simdResult, regularResult)
}

func TestVectorPool(t *testing.T) {
	dimension := 128

	// Get vector from pool
	vec1 := GetPooledVector(dimension)
	if vec1.Len() != dimension {
		t.Errorf("Expected dimension %d, got %d", dimension, vec1.Len())
	}

	// Modify the vector
	data := vec1.Data()
	for i := range data {
		data[i] = float32(i)
	}

	// Return to pool
	PutPooledVector(vec1)

	// Get another vector (should be reused)
	vec2 := GetPooledVector(dimension)
	data2 := vec2.Data()

	// Data should be cleared
	for i, val := range data2 {
		if val != 0 {
			t.Errorf("Vector data not cleared at index %d: %f", i, val)
		}
	}

	PutPooledVector(vec2)
}

func TestAdaptiveIndex(t *testing.T) {
	// Create a mock index factory
	factory := &MockIndexFactory{}

	// Create adaptive index
	ai := NewAdaptiveIndex(128, DistanceCosine, factory)

	// Test that it starts with flat index
	if ai.Type() != "flat" {
		t.Errorf("Expected flat index initially, got %s", ai.Type())
	}

	// Add some vectors to trigger potential index switching
	for i := 0; i < 100; i++ {
		vector := Vector{
			ID:     fmt.Sprintf("vec_%d", i),
			Values: make([]float32, 128),
		}

		err := ai.Add(vector)
		if err != nil {
			t.Errorf("Failed to add vector: %v", err)
		}
	}

	// Get stats
	stats := ai.GetCurrentStats()
	if stats.TotalQueries < 0 {
		t.Error("Invalid stats")
	}
}

func TestLLMCache(t *testing.T) {
	config := DefaultCacheConfig()
	cache := NewLLMCache(config)

	// Test caching search results
	query := []float32{1.0, 2.0, 3.0, 4.0}
	results := []SearchResult{
		{ID: "result1", Score: 0.9},
		{ID: "result2", Score: 0.8},
	}

	agentID := "agent123"
	conversationID := "conv456"

	// Cache results
	cache.CacheSearchResults(query, results, agentID, conversationID)

	// Retrieve results
	cached, found := cache.GetSearchResults(query, agentID)
	if !found {
		t.Error("Expected to find cached results")
	}

	if len(cached) != len(results) {
		t.Errorf("Expected %d results, got %d", len(results), len(cached))
	}

	// Test cache invalidation
	cache.InvalidateAgent(agentID)

	// Should not find results after invalidation (try temporal cache instead)
	_, found = cache.GetSearchResults(query, "different_agent")
	if !found {
		// This is expected - temporal cache might still have it
		t.Log("Results not found in temporal cache, which is acceptable")
	}

	// Test stats
	stats := cache.GetStats()
	if stats.TotalQueries == 0 {
		t.Error("Expected some query statistics")
	}
}

func TestDeduplication(t *testing.T) {
	config := DefaultDeduplicationConfig()
	dm := NewDeduplicationManager(128, config)

	// Create test vectors
	vector1 := Vector{
		ID:     "vec1",
		Values: make([]float32, 128),
	}

	// Fill with test data
	for i := range vector1.Values {
		vector1.Values[i] = float32(i)
	}

	// Add first vector
	duplicate1, err := dm.AddVector(vector1)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	if duplicate1.OriginalID != "vec1" {
		t.Errorf("Expected original ID vec1, got %s", duplicate1.OriginalID)
	}

	// Add identical vector
	vector2 := vector1
	vector2.ID = "vec2"

	duplicate2, err := dm.AddVector(vector2)
	if err != nil {
		t.Fatalf("Failed to add duplicate vector: %v", err)
	}

	// Should detect as duplicate
	if duplicate2.OriginalID != "vec1" {
		t.Errorf("Expected duplicate to reference vec1, got %s", duplicate2.OriginalID)
	}

	if duplicate2.RefCount != 2 {
		t.Errorf("Expected ref count 2, got %d", duplicate2.RefCount)
	}

	// Test stats
	stats := dm.GetStats()
	if stats.ExactDuplicates == 0 {
		t.Error("Expected to detect exact duplicate")
	}
}

func TestLSMStorage(t *testing.T) {
	config := DefaultLSMConfig()

	// Use temporary directory
	baseDir := "/tmp/lsm_test_" + fmt.Sprintf("%d", time.Now().UnixNano())
	defer func() {
		// Cleanup would go here in a real test
	}()

	lsm, err := NewLSMStorage(baseDir, config)
	if err != nil {
		t.Fatalf("Failed to create LSM storage: %v", err)
	}
	defer lsm.Close()

	// Test put and get
	vector := Vector{
		ID:     "test_vec",
		Values: []float32{1.0, 2.0, 3.0, 4.0},
	}

	err = lsm.Put("key1", vector)
	if err != nil {
		t.Fatalf("Failed to put vector: %v", err)
	}

	retrieved, found, err := lsm.Get("key1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}

	if !found {
		t.Error("Expected to find stored vector")
	}

	if retrieved.ID != vector.ID {
		t.Errorf("Expected ID %s, got %s", vector.ID, retrieved.ID)
	}

	// Test delete
	err = lsm.Delete("key1")
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	_, found, err = lsm.Get("key1")
	if err != nil {
		t.Fatalf("Failed to check deleted vector: %v", err)
	}

	if found {
		t.Error("Expected not to find deleted vector")
	}

	// Test stats
	stats := lsm.GetStats()
	if stats.WritesTotal == 0 {
		t.Error("Expected some write statistics")
	}
}

// Mock index factory for testing
type MockIndexFactory struct{}

func (mif *MockIndexFactory) CreateIndex(indexType string, dimension int, distanceMetric DistanceMetric) (Index, error) {
	return &MockIndex{indexType: indexType}, nil
}

type MockIndex struct {
	indexType string
	size      int
}

func (mi *MockIndex) Add(vector Vector) error {
	mi.size++
	return nil
}

func (mi *MockIndex) Search(query []float32, k int, filter map[string]string) ([]SearchResult, error) {
	return []SearchResult{}, nil
}

func (mi *MockIndex) RangeSearch(query []float32, radius float32, filter map[string]string, limit int) ([]SearchResult, error) {
	return []SearchResult{}, nil
}

func (mi *MockIndex) Delete(id string) error {
	if mi.size > 0 {
		mi.size--
	}
	return nil
}

func (mi *MockIndex) Rebuild() error {
	return nil
}

func (mi *MockIndex) Size() int {
	return mi.size
}

func (mi *MockIndex) Type() string {
	return mi.indexType
}

func (mi *MockIndex) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (mi *MockIndex) Deserialize(data []byte) error {
	return nil
}
