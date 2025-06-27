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
	"github.com/dshills/EmbeddixDB/core/ai"
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
		enableEmbedding = flag.Bool("enable-embedding", false, "Enable text embedding support")
		modelPath       = flag.String("model", "", "Path to ONNX embedding model")
		modelName       = flag.String("model-name", "all-MiniLM-L6-v2", "Name of the embedding model")
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
	var store core.VectorStore
	store = core.NewVectorStore(persistenceBackend, indexFactory)

	// Wrap with embedding support if enabled
	if *enableEmbedding {
		// Validate that model path is provided
		if *modelPath == "" {
			log.Fatalf("--model flag is required when --enable-embedding is set")
		}

		modelConfig := ai.ModelConfig{
			Name:                *modelName,
			MaxTokens:           512,
			BatchSize:           32,
			NormalizeEmbeddings: true,
			PoolingStrategy:     "mean",
		}

		embeddingStore, err := mcp.CreateEmbeddingStore(store, *modelPath, modelConfig)
		if err != nil {
			log.Fatalf("Failed to create embedding store: %v", err)
		}
		store = embeddingStore

		if *verbose {
			fmt.Fprintf(os.Stderr, "Embedding support enabled with model: %s\n", *modelName)
		}
	}

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
