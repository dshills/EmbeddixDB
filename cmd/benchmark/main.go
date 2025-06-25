package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/dshills/EmbeddixDB/benchmark"
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	// Parse command line flags
	var (
		dbType          = flag.String("db", "memory", "Database type: memory, bolt, badger")
		dbPath          = flag.String("path", "data/benchmark.db", "Database path")
		vectorDim       = flag.Int("dim", 128, "Vector dimension")
		numVectors      = flag.Int("vectors", 10000, "Number of vectors to test")
		numQueries      = flag.Int("queries", 1000, "Number of search queries")
		topK            = flag.Int("topk", 10, "Top K results for search")
		parallelism     = flag.Int("parallel", 10, "Parallelism for concurrent tests")
		batchSize       = flag.Int("batch", 100, "Batch size for batch operations")
		indexType       = flag.String("index", "flat", "Index type: flat, hnsw")
		hnswComparison  = flag.Bool("compare", false, "Compare flat vs HNSW performance")
	)
	flag.Parse()

	fmt.Println("=== EmbeddixDB Performance Benchmark ===")
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Database: %s\n", *dbType)
	fmt.Printf("  Index: %s\n", *indexType)
	fmt.Printf("  Vectors: %d x %d dimensions\n", *numVectors, *vectorDim)
	fmt.Printf("  Queries: %d\n", *numQueries)
	fmt.Printf("  Parallelism: %d\n", *parallelism)
	fmt.Println()

	if *dbType == "memory" {
		// For benchmarking, we'll use memory persistence
		runBenchmark("memory", *indexType, *vectorDim, *numVectors, *numQueries, *topK, *parallelism, *batchSize)
	} else {
		// Clean up any existing data
		os.RemoveAll(*dbPath)
		runBenchmark(*dbType, *indexType, *vectorDim, *numVectors, *numQueries, *topK, *parallelism, *batchSize)
	}

	// If comparison mode, run both flat and HNSW
	if *hnswComparison && *indexType != "hnsw" {
		fmt.Println("\n\n=== Running HNSW comparison ===")
		runBenchmark(*dbType, "hnsw", *vectorDim, *numVectors, *numQueries, *topK, *parallelism, *batchSize)
	}
}

func runBenchmark(dbType, indexType string, vectorDim, numVectors, numQueries, topK, parallelism, batchSize int) {
	// Create persistence
	var persist core.Persistence
	var err error

	switch dbType {
	case "memory":
		persist = persistence.NewMemoryPersistence()
	case "bolt":
		config := persistence.PersistenceConfig{
			Type: persistence.PersistenceBolt,
			Path: "data/benchmark_bolt.db",
		}
		factory := persistence.NewDefaultFactory()
		persist, err = factory.CreatePersistence(config)
		if err != nil {
			log.Fatalf("Failed to create BoltDB persistence: %v", err)
		}
		defer persist.Close()
		defer os.RemoveAll("data/benchmark_bolt.db")
	case "badger":
		config := persistence.PersistenceConfig{
			Type: persistence.PersistenceBadger,
			Path: "data/benchmark_badger",
		}
		factory := persistence.NewDefaultFactory()
		persist, err = factory.CreatePersistence(config)
		if err != nil {
			log.Fatalf("Failed to create BadgerDB persistence: %v", err)
		}
		defer persist.Close()
		defer os.RemoveAll("data/benchmark_badger")
	default:
		log.Fatalf("Unknown database type: %s", dbType)
	}

	// Create index factory
	indexFactory := index.NewDefaultFactory()

	// Create vector store
	vectorStore := core.NewVectorStore(persist, indexFactory)
	defer vectorStore.Close()

	// Configure benchmark
	benchConfig := benchmark.BenchmarkConfig{
		VectorDimension: vectorDim,
		NumVectors:      numVectors,
		NumQueries:      numQueries,
		TopK:            topK,
		Parallelism:     parallelism,
		BatchSize:       batchSize,
	}

	// Create benchmark runner
	bench := benchmark.NewBenchmark(vectorStore, benchConfig)

	// Create test collection with specified index type
	ctx := context.Background()
	collectionName := fmt.Sprintf("bench_%s_%s", dbType, indexType)

	// Override the collection creation in benchmark to use our index type
	collection := core.Collection{
		Name:      collectionName,
		Dimension: vectorDim,
		IndexType: indexType,
		Distance:  "cosine",
	}

	// Delete any existing collection
	vectorStore.DeleteCollection(ctx, collectionName)

	// Create new collection
	if err := vectorStore.CreateCollection(ctx, collection); err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Run benchmarks
	fmt.Printf("Running benchmarks for %s database with %s index...\n", dbType, indexType)
	results, err := bench.RunAll(ctx, collectionName)
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	// Print results
	benchmark.PrintResults(results)

	// Additional statistics
	fmt.Println("\n=== Summary ===")
	var totalOps int
	var totalTime time.Duration
	for _, r := range results {
		if r.Operation != "Batch Insert" { // Don't double count batch inserts
			totalOps += r.OperationCount
			totalTime += r.TotalTime
		}
	}

	fmt.Printf("Total operations: %d\n", totalOps)
	fmt.Printf("Total time: %v\n", totalTime)
	fmt.Printf("Database type: %s\n", dbType)
	fmt.Printf("Index type: %s\n", indexType)

	// Memory usage (approximate)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory usage: %.2f MB\n", float64(m.Alloc)/1024/1024)
}

