package main

import (
	"context"
	"fmt"
	"log"
	"time"
	
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	ctx := context.Background()
	
	// Create persistence layer
	memPersistence := persistence.NewMemoryPersistence()
	
	// Create index factory
	indexFactory := index.NewDefaultFactory()
	
	// Create vector store
	store := core.NewVectorStore(memPersistence, indexFactory)
	defer store.Close()
	
	// Create a collection
	collection := core.Collection{
		Name:      "documents",
		Dimension: 3,
		IndexType: "flat",
		Distance:  "cosine",
		CreatedAt: time.Now(),
	}
	
	err := store.CreateCollection(ctx, collection)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	
	fmt.Println("Collection created successfully!")
	
	// Add some test vectors
	vectors := []core.Vector{
		{ID: "doc1", Values: []float32{1.0, 0.0, 0.0}, Metadata: map[string]string{"type": "text"}},
		{ID: "doc2", Values: []float32{0.0, 1.0, 0.0}, Metadata: map[string]string{"type": "text"}},
		{ID: "doc3", Values: []float32{0.0, 0.0, 1.0}, Metadata: map[string]string{"type": "image"}},
	}
	
	for _, vec := range vectors {
		err := store.AddVector(ctx, "documents", vec)
		if err != nil {
			log.Fatalf("Failed to add vector %s: %v", vec.ID, err)
		}
	}
	
	fmt.Printf("Added %d vectors successfully!\n", len(vectors))
	
	// Perform a search
	searchReq := core.SearchRequest{
		Query:          []float32{1.0, 0.1, 0.0},
		TopK:           2,
		Filter:         map[string]string{"type": "text"},
		IncludeVectors: true,
	}
	
	results, err := store.Search(ctx, "documents", searchReq)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	
	fmt.Printf("Search results (%d found):\n", len(results))
	for i, result := range results {
		fmt.Printf("  %d. ID: %s, Score: %.4f, Metadata: %v\n", 
			i+1, result.ID, result.Score, result.Metadata)
	}
}