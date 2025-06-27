package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/dshills/EmbeddixDB/api"
	"github.com/dshills/EmbeddixDB/config"
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai/embedding"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	// Parse command line flags
	var (
		configPath = flag.String("config", "", "Path to configuration file (default: ~/.embeddixdb.yml)")
		host       = flag.String("host", "", "Host to listen on (overrides config)")
		port       = flag.Int("port", 0, "Port to listen on (overrides config)")
		dbType     = flag.String("db", "", "Database type: memory, bolt, badger (overrides config)")
		dbPath     = flag.String("path", "", "Database path (overrides config)")
		enableWAL  = flag.Bool("wal", false, "Enable Write-Ahead Logging")
		walPath    = flag.String("wal-path", "data/wal", "WAL directory path")
	)
	flag.Parse()

	// Load configuration
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Override with command-line flags
	if *host != "" {
		cfg.Server.Host = *host
	}
	if *port != 0 {
		cfg.Server.Port = *port
	}
	if *dbType != "" {
		cfg.Persistence.Type = persistence.PersistenceType(*dbType)
	}
	if *dbPath != "" {
		cfg.Persistence.Options["path"] = *dbPath
	}

	// Add WAL configuration if enabled
	if *enableWAL {
		cfg.Persistence.Options["wal_enabled"] = true
		cfg.Persistence.Options["wal_path"] = *walPath
	}

	fmt.Println("=== EmbeddixDB Server ===")
	fmt.Printf("Version: 2.2.0\n")
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Host: %s:%d\n", cfg.Server.Host, cfg.Server.Port)
	fmt.Printf("  Database: %s\n", cfg.Persistence.Type)
	if path, ok := cfg.Persistence.Options["path"]; ok {
		fmt.Printf("  Path: %v\n", path)
	}
	fmt.Printf("  AI Engine: %s\n", cfg.AI.Embedding.Engine)
	if cfg.AI.Embedding.Engine == "ollama" {
		fmt.Printf("  Ollama Endpoint: %s\n", cfg.AI.Embedding.Ollama.Endpoint)
	}
	fmt.Println()

	// Create persistence layer
	factory := persistence.NewDefaultFactory()
	persist, err := factory.CreatePersistence(cfg.Persistence)
	if err != nil {
		log.Fatalf("Failed to create persistence: %v", err)
	}
	defer persist.Close()

	// Create index factory
	indexFactory := index.NewDefaultFactory()

	// Create vector store with recovery
	vectorStore, err := core.NewVectorStoreWithRecovery(persist, indexFactory)
	if err != nil {
		log.Fatalf("Failed to create vector store: %v", err)
	}
	defer vectorStore.Close()

	// Initialize AI components if enabled
	if cfg.AI.Embedding.Engine != "" {
		modelConfig := cfg.AI.Embedding.ToModelConfig()
		embedder, err := embedding.CreateEngine(modelConfig)
		if err != nil {
			log.Printf("Warning: Failed to create embedding engine: %v", err)
		} else {
			fmt.Printf("  Embedding Engine: %s initialized\n", cfg.AI.Embedding.Engine)
			// The embedder can be passed to handlers that need it
			_ = embedder // Placeholder for future use
		}
	}

	// Create API server
	serverConfig := cfg.Server.ToServerConfig()
	server := api.NewServer(vectorStore, serverConfig)

	// Start server in a goroutine
	go func() {
		if err := server.Start(); err != nil {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	fmt.Println("\nShutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	fmt.Println("Server stopped gracefully")
}
