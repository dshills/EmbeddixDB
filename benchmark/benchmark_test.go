package benchmark

import (
	"context"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func TestBenchmark(t *testing.T) {
	// Create in-memory vector store
	persist := persistence.NewMemoryPersistence()
	indexFactory := index.NewDefaultFactory()
	vectorStore := core.NewVectorStore(persist, indexFactory)
	defer vectorStore.Close()

	// Create test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "bench_test",
		Dimension: 64,
		IndexType: "flat",
		Distance:  "cosine",
	}

	err := vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Configure small benchmark for testing
	config := BenchmarkConfig{
		VectorDimension: 64,
		NumVectors:      100,
		NumQueries:      10,
		TopK:            5,
		Parallelism:     2,
		BatchSize:       10,
	}

	// Create benchmark
	bench := NewBenchmark(vectorStore, config)

	// Run benchmarks
	results, err := bench.RunAll(ctx, "bench_test")
	if err != nil {
		t.Fatalf("Benchmark failed: %v", err)
	}

	// Verify we got results
	if len(results) == 0 {
		t.Error("No benchmark results returned")
	}

	// Check each result
	expectedOps := map[string]int{
		"Individual Insert": 100,
		"Batch Insert":      100, // Total vectors inserted
		"Search":            10,
		"Concurrent Search": 10,
		"Get Vector":        100,
		"Update Vector":     100,
	}

	for _, result := range results {
		t.Logf("%s: %d ops in %v (%.2f ops/sec)",
			result.Operation,
			result.OperationCount,
			result.TotalTime,
			result.Throughput)

		if result.OperationCount == 0 {
			t.Errorf("%s: no operations completed", result.Operation)
		}

		if result.AvgLatency == 0 {
			t.Errorf("%s: average latency is zero", result.Operation)
		}

		if expected, ok := expectedOps[result.Operation]; ok {
			if result.OperationCount != expected {
				t.Errorf("%s: expected %d operations, got %d",
					result.Operation, expected, result.OperationCount)
			}
		}
	}
}

func TestGenerateVectors(t *testing.T) {
	config := BenchmarkConfig{
		VectorDimension: 128,
		NumVectors:      10,
	}

	bench := &Benchmark{config: config}
	vectors := bench.generateVectors(10)

	if len(vectors) != 10 {
		t.Errorf("Expected 10 vectors, got %d", len(vectors))
	}

	for i, vec := range vectors {
		if len(vec.Values) != 128 {
			t.Errorf("Vector %d has wrong dimension: expected 128, got %d", i, len(vec.Values))
		}

		if vec.ID == "" {
			t.Errorf("Vector %d has empty ID", i)
		}

		// Check values are in reasonable range
		for j, val := range vec.Values {
			if val < 0 || val > 1 {
				t.Errorf("Vector %d value %d out of range: %f", i, j, val)
			}
		}
	}
}

func TestSortDurations(t *testing.T) {
	durations := []time.Duration{
		5 * time.Millisecond,
		1 * time.Millisecond,
		10 * time.Millisecond,
		3 * time.Millisecond,
	}

	sortDurations(durations)

	// Check sorted order
	for i := 1; i < len(durations); i++ {
		if durations[i] < durations[i-1] {
			t.Errorf("Durations not sorted at index %d: %v < %v",
				i, durations[i], durations[i-1])
		}
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		duration time.Duration
		expected string
	}{
		{500 * time.Nanosecond, "500ns"},
		{1500 * time.Nanosecond, "1.5µs"},
		{1500 * time.Microsecond, "1.5ms"},
		{1500 * time.Millisecond, "1.50s"},
	}

	for _, test := range tests {
		result := formatDuration(test.duration)
		if result != test.expected {
			t.Errorf("formatDuration(%v) = %s, expected %s",
				test.duration, result, test.expected)
		}
	}
}
