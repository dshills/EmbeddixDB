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
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func main() {
	// Parse command line flags
	var (
		host         = flag.String("host", "0.0.0.0", "Host to listen on")
		port         = flag.Int("port", 8080, "Port to listen on")
		dbType       = flag.String("db", "bolt", "Database type: memory, bolt, badger")
		dbPath       = flag.String("path", "data/embeddix.db", "Database path")
		enableWAL    = flag.Bool("wal", false, "Enable Write-Ahead Logging")
		walPath      = flag.String("wal-path", "data/wal", "WAL directory path")
	)
	flag.Parse()

	fmt.Println("=== EmbeddixDB Server ===")
	fmt.Printf("Version: 1.0.0\n")
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Host: %s:%d\n", *host, *port)
	fmt.Printf("  Database: %s\n", *dbType)
	fmt.Printf("  Path: %s\n", *dbPath)
	fmt.Printf("  WAL: %v\n", *enableWAL)
	fmt.Println()

	// Create persistence configuration
	config := persistence.PersistenceConfig{
		Type: persistence.PersistenceType(*dbType),
		Path: *dbPath,
	}

	// Add WAL configuration if enabled
	if *enableWAL {
		config.WAL = &persistence.WALConfig{
			Path:     *walPath,
			MaxSize:  10 * 1024 * 1024, // 10MB
			SyncMode: true,
		}
	}

	// Validate configuration
	if err := persistence.ValidateConfig(config); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	// Create persistence layer
	factory := persistence.NewDefaultFactory()
	persist, err := factory.CreatePersistence(config)
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

	// Create API server
	serverConfig := api.ServerConfig{
		Host:         *host,
		Port:         *port,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

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