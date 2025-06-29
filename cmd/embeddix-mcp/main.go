package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/dshills/EmbeddixDB/config"
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/ai/embedding"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/mcp"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	// Command line flags
	var (
		configPath      = flag.String("config", "", "Path to configuration file (default: ~/.embeddixdb.yml)")
		persistenceType = flag.String("persistence", "", "Persistence type: memory, bolt, badger (overrides config)")
		dataPath        = flag.String("data", "", "Data directory path (overrides config)")
		verbose         = flag.Bool("verbose", false, "Enable verbose logging")
		enableEmbedding = flag.Bool("enable-embedding", false, "Enable text embedding support")
		modelPath       = flag.String("model", "", "Path to ONNX embedding model (overrides config)")
		modelName       = flag.String("model-name", "", "Name of the embedding model (overrides config)")
	)

	flag.Parse()

	// Set up logging
	if !*verbose {
		// Suppress standard logging in non-verbose mode
		log.SetOutput(os.Stderr)
		log.SetFlags(0)
	}

	// Load configuration
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Override with command-line flags
	if *persistenceType != "" {
		cfg.Persistence.Type = persistence.PersistenceType(*persistenceType)
	}
	if *dataPath != "" {
		cfg.Persistence.Path = *dataPath
		cfg.Persistence.Options["path"] = *dataPath
	}
	if *modelPath != "" {
		cfg.AI.Embedding.ONNX.ModelPath = *modelPath
	}
	if *modelName != "" {
		cfg.AI.Embedding.Model = *modelName
	}

	// Create persistence configuration
	persistenceConfig := cfg.Persistence

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

	// Wrap with embedding support if enabled or configured
	if *enableEmbedding || cfg.AI.Embedding.Engine != "" {
		// Use configuration if no explicit model path provided
		var modelConfig ai.ModelConfig
		if cfg.AI.Embedding.Engine != "" {
			modelConfig = cfg.AI.Embedding.ToModelConfig()
		} else {
			// Legacy support for command-line model specification
			if *modelPath == "" {
				log.Fatalf("--model flag is required when --enable-embedding is set")
			}
			modelConfig = ai.ModelConfig{
				Type:                ai.ModelTypeONNX,
				Path:                *modelPath,
				Name:                cfg.AI.Embedding.Model,
				MaxTokens:           512,
				BatchSize:           32,
				NormalizeEmbeddings: true,
				PoolingStrategy:     "mean",
			}
		}

		// Create embedding engine
		embedder, err := embedding.CreateEngine(modelConfig)
		if err != nil {
			log.Fatalf("Failed to create embedding engine: %v", err)
		}

		// Warm up the engine (required for Ollama and other engines)
		warmCtx := context.Background()
		fmt.Fprintf(os.Stderr, "Warming up embedding engine...\n")
		if err := embedder.Warm(warmCtx); err != nil {
			log.Fatalf("Failed to warm up embedding engine: %v", err)
		}
		fmt.Fprintf(os.Stderr, "Embedding engine warmed up successfully\n")

		embeddingStore, err := mcp.CreateEmbeddingStoreWithEngine(store, embedder)
		if err != nil {
			log.Fatalf("Failed to create embedding store: %v", err)
		}
		store = embeddingStore

		fmt.Fprintf(os.Stderr, "Embedding support enabled with engine: %s, model: %s\n",
			cfg.AI.Embedding.Engine, cfg.AI.Embedding.Model)
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

	// Log startup message and configuration to stderr (MCP uses stdout for communication)
	fmt.Fprintf(os.Stderr, "=== EmbeddixDB MCP Server Configuration ===\n")
	fmt.Fprintf(os.Stderr, "\n[Server Settings]\n")
	fmt.Fprintf(os.Stderr, "  Host: %s\n", cfg.Server.Host)
	fmt.Fprintf(os.Stderr, "  Port: %d\n", cfg.Server.Port)
	fmt.Fprintf(os.Stderr, "  Read Timeout: %v\n", cfg.Server.ReadTimeout)
	fmt.Fprintf(os.Stderr, "  Write Timeout: %v\n", cfg.Server.WriteTimeout)
	fmt.Fprintf(os.Stderr, "  Shutdown Timeout: %v\n", cfg.Server.ShutdownTimeout)
	
	fmt.Fprintf(os.Stderr, "\n[Persistence]\n")
	fmt.Fprintf(os.Stderr, "  Type: %s\n", cfg.Persistence.Type)
	fmt.Fprintf(os.Stderr, "  Path: %s\n", cfg.Persistence.Path)
	if len(cfg.Persistence.Options) > 0 {
		fmt.Fprintf(os.Stderr, "  Options:\n")
		for k, v := range cfg.Persistence.Options {
			fmt.Fprintf(os.Stderr, "    %s: %v\n", k, v)
		}
	}
	
	fmt.Fprintf(os.Stderr, "\n[Vector Store]\n")
	fmt.Fprintf(os.Stderr, "  Default Index Type: %s\n", cfg.VectorStore.DefaultIndexType)
	fmt.Fprintf(os.Stderr, "  Default Distance Metric: %s\n", cfg.VectorStore.DefaultDistanceMetric)
	
	if cfg.AI.Embedding.Engine != "" || *enableEmbedding {
		fmt.Fprintf(os.Stderr, "\n[AI/Embedding]\n")
		fmt.Fprintf(os.Stderr, "  Engine: %s\n", cfg.AI.Embedding.Engine)
		fmt.Fprintf(os.Stderr, "  Model: %s\n", cfg.AI.Embedding.Model)
		fmt.Fprintf(os.Stderr, "  Batch Size: %d\n", cfg.AI.Embedding.BatchSize)
		fmt.Fprintf(os.Stderr, "  Max Queue Size: %d\n", cfg.AI.Embedding.MaxQueueSize)
		
		if cfg.AI.Embedding.Engine == "ollama" {
			fmt.Fprintf(os.Stderr, "  Ollama Endpoint: %s\n", cfg.AI.Embedding.Ollama.Endpoint)
			fmt.Fprintf(os.Stderr, "  Ollama Timeout: %v\n", cfg.AI.Embedding.Ollama.Timeout)
		} else if cfg.AI.Embedding.Engine == "onnx" || *modelPath != "" {
			modelPathStr := cfg.AI.Embedding.ONNX.ModelPath
			if *modelPath != "" {
				modelPathStr = *modelPath
			}
			fmt.Fprintf(os.Stderr, "  ONNX Model Path: %s\n", modelPathStr)
			fmt.Fprintf(os.Stderr, "  ONNX Use GPU: %v\n", cfg.AI.Embedding.ONNX.UseGPU)
			fmt.Fprintf(os.Stderr, "  ONNX Threads: %d\n", cfg.AI.Embedding.ONNX.NumThreads)
		}
	}
	
	fmt.Fprintf(os.Stderr, "\n[Runtime Settings]\n")
	fmt.Fprintf(os.Stderr, "  Config Path: %s\n", func() string {
		if *configPath != "" {
			return *configPath
		}
		return "~/.embeddixdb.yml (default)"
	}())
	fmt.Fprintf(os.Stderr, "  Verbose Mode: %v\n", *verbose)
	fmt.Fprintf(os.Stderr, "  Embedding Enabled: %v\n", *enableEmbedding || cfg.AI.Embedding.Engine != "")
	
	fmt.Fprintf(os.Stderr, "\n==========================================\n")
	fmt.Fprintf(os.Stderr, "Starting MCP server...\n\n")

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
