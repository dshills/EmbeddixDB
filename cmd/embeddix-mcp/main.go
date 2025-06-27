package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/mcp"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	// Command line flags
	var (
		persistenceType = flag.String("persistence", "memory", "Persistence type: memory, bolt, badger")
		dataPath        = flag.String("data", "./data", "Data directory path")
		verbose         = flag.Bool("verbose", false, "Enable verbose logging")
	)
	
	flag.Parse()
	
	// Set up logging
	if !*verbose {
		// Suppress standard logging in non-verbose mode
		log.SetOutput(os.Stderr)
		log.SetFlags(0)
	}
	
	// Create persistence configuration
	var persistenceTypeEnum persistence.PersistenceType
	switch *persistenceType {
	case "memory":
		persistenceTypeEnum = persistence.PersistenceMemory
	case "bolt":
		persistenceTypeEnum = persistence.PersistenceBolt
	case "badger":
		persistenceTypeEnum = persistence.PersistenceBadger
	default:
		log.Fatalf("Unknown persistence type: %s", *persistenceType)
	}
	
	// Create persistence configuration
	persistenceConfig := persistence.DefaultPersistenceConfig(persistenceTypeEnum, *dataPath)
	
	// Create persistence factory and persistence backend
	persistenceFactory := persistence.NewDefaultFactory()
	persistenceBackend, err := persistenceFactory.CreatePersistence(persistenceConfig)
	if err != nil {
		log.Fatalf("Failed to create persistence backend: %v", err)
	}
	defer persistenceBackend.Close()
	
	// Create index factory
	indexFactory := index.NewDefaultFactory()
	
	// Create vector store
	store := core.NewVectorStore(persistenceBackend, indexFactory)
	
	// Create MCP server
	server := mcp.NewServer(store)
	
	// Set up signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		if *verbose {
			log.Println("Shutting down...")
		}
		cancel()
	}()
	
	// Log startup message to stderr (MCP uses stdout for communication)
	if *verbose {
		fmt.Fprintf(os.Stderr, "EmbeddixDB MCP Server starting...\n")
		fmt.Fprintf(os.Stderr, "Persistence: %s\n", *persistenceType)
		fmt.Fprintf(os.Stderr, "Data path: %s\n", *dataPath)
	}
	
	// Start serving
	if err := server.Serve(ctx); err != nil && err != context.Canceled {
		log.Fatalf("Server error: %v", err)
	}
	
	// Clean shutdown
	if err := server.Close(); err != nil {
		log.Printf("Error closing server: %v", err)
	}
	
	if *verbose {
		fmt.Fprintf(os.Stderr, "Server stopped\n")
	}
}