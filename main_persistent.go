package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"
	
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	ctx := context.Background()
	
	// Test both BoltDB and BadgerDB
	fmt.Println("=== Testing BoltDB Persistence ===")
	testPersistent(ctx, "bolt", "data/bolt/test.db")
	
	fmt.Println("\n=== Testing BadgerDB Persistence ===")
	testPersistent(ctx, "badger", "data/badger")
	
	fmt.Println("\n=== Testing Data Persistence Across Restarts ===")
	testDataPersistence(ctx)
}

func testPersistent(ctx context.Context, persistenceType, dbPath string) {
	// Clean up any existing data
	os.RemoveAll(dbPath)
	
	// Create persistence layer
	var persist core.Persistence
	var err error
	
	switch persistenceType {
	case "bolt":
		persist, err = persistence.NewBoltPersistence(dbPath)
	case "badger":
		persist, err = persistence.NewBadgerPersistence(dbPath)
	default:
		log.Fatalf("Unknown persistence type: %s", persistenceType)
	}
	
	if err != nil {
		log.Fatalf("Failed to create %s persistence: %v", persistenceType, err)
	}
	defer persist.Close()
	
	// Create index factory
	indexFactory := index.NewDefaultFactory()
	
	// Create vector store with persistent storage
	store := core.NewVectorStore(persist, indexFactory)
	defer store.Close()
	
	// Create a collection
	collection := core.Collection{
		Name:      "persistent_docs",
		Dimension: 4,
		IndexType: "hnsw",
		Distance:  "cosine",
		CreatedAt: time.Now(),
	}
	
	err = store.CreateCollection(ctx, collection)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	
	fmt.Printf("%s Collection created successfully!\n", persistenceType)
	
	// Add test vectors
	vectors := []core.Vector{
		{ID: "doc1", Values: []float32{1.0, 0.0, 0.0, 0.0}, Metadata: map[string]string{"type": "text", "category": "A"}},
		{ID: "doc2", Values: []float32{0.0, 1.0, 0.0, 0.0}, Metadata: map[string]string{"type": "text", "category": "B"}},
		{ID: "doc3", Values: []float32{0.0, 0.0, 1.0, 0.0}, Metadata: map[string]string{"type": "image", "category": "A"}},
		{ID: "doc4", Values: []float32{0.0, 0.0, 0.0, 1.0}, Metadata: map[string]string{"type": "audio", "category": "C"}},
		{ID: "doc5", Values: []float32{0.7, 0.7, 0.0, 0.0}, Metadata: map[string]string{"type": "text", "category": "A"}},
	}
	
	for _, vec := range vectors {
		err := store.AddVector(ctx, "persistent_docs", vec)
		if err != nil {
			log.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}
	
	fmt.Printf("Added %d vectors to %s persistence!\n", len(vectors), persistenceType)
	
	// Perform search
	searchReq := core.SearchRequest{
		Query:          []float32{1.0, 0.1, 0.0, 0.0},
		TopK:           3,
		Filter:         map[string]string{"category": "A"},
		IncludeVectors: false,
	}
	
	results, err := store.Search(ctx, "persistent_docs", searchReq)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	
	fmt.Printf("%s Search results (%d found):\n", persistenceType, len(results))
	for i, result := range results {
		fmt.Printf("  %d. ID: %s, Score: %.4f, Metadata: %v\n", 
			i+1, result.ID, result.Score, result.Metadata)
	}
	
	// Show stats if available
	if statsPersist, ok := persist.(interface{ Stats() (map[string]interface{}, error) }); ok {
		stats, err := statsPersist.Stats()
		if err == nil {
			fmt.Printf("%s Database stats: %+v\n", persistenceType, stats)
		}
	}
}

func testDataPersistence(ctx context.Context) {
	dbPath := "data/persistence_test.db"
	
	// Phase 1: Create data and close
	{
		persist, err := persistence.NewBoltPersistence(dbPath)
		if err != nil {
			log.Fatalf("Failed to create BoltDB persistence: %v", err)
		}
		
		indexFactory := index.NewDefaultFactory()
		store := core.NewVectorStore(persist, indexFactory)
		
		// Create collection and add data
		collection := core.Collection{
			Name:      "restart_test",
			Dimension: 2,
			IndexType: "flat",
			Distance:  "l2",
			CreatedAt: time.Now(),
		}
		
		store.CreateCollection(ctx, collection)
		
		vectors := []core.Vector{
			{ID: "persistent1", Values: []float32{1.0, 2.0}, Metadata: map[string]string{"session": "1"}},
			{ID: "persistent2", Values: []float32{3.0, 4.0}, Metadata: map[string]string{"session": "1"}},
		}
		
		for _, vec := range vectors {
			store.AddVector(ctx, "restart_test", vec)
		}
		
		fmt.Println("Phase 1: Created data and closing...")
		store.Close()
		persist.Close()
	}
	
	// Phase 2: Reopen and verify data persisted
	{
		persist, err := persistence.NewBoltPersistence(dbPath)
		if err != nil {
			log.Fatalf("Failed to reopen BoltDB persistence: %v", err)
		}
		defer persist.Close()
		
		indexFactory := index.NewDefaultFactory()
		store, err := core.NewVectorStoreWithRecovery(persist, indexFactory)
		if err != nil {
			log.Fatalf("Failed to create vector store with recovery: %v", err)
		}
		defer store.Close()
		
		// Verify collection exists
		collections, err := store.ListCollections(ctx)
		if err != nil {
			log.Fatalf("Failed to list collections: %v", err)
		}
		
		fmt.Printf("Phase 2: Found %d collections after restart\n", len(collections))
		
		// Verify vectors exist and can be searched
		vec, err := store.GetVector(ctx, "restart_test", "persistent1")
		if err != nil {
			log.Fatalf("Failed to retrieve vector after restart: %v", err)
		}
		
		fmt.Printf("Successfully retrieved vector after restart: %s with values %v\n", vec.ID, vec.Values)
		
		// Verify search still works
		searchReq := core.SearchRequest{
			Query:          []float32{1.1, 2.1},
			TopK:           2,
			IncludeVectors: true,
		}
		
		results, err := store.Search(ctx, "restart_test", searchReq)
		if err != nil {
			log.Fatalf("Search failed after restart: %v", err)
		}
		
		fmt.Printf("Search after restart found %d results\n", len(results))
		for _, result := range results {
			fmt.Printf("  ID: %s, Score: %.4f\n", result.ID, result.Score)
		}
	}
	
	fmt.Println("✅ Data persistence across restarts verified!")
}