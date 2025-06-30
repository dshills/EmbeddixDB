package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/dshills/EmbeddixDB/config"
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/ai/embedding"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/mcp"
	"github.com/dshills/EmbeddixDB/persistence"
)

// resolveDatabasePath determines the database path using the following priority:
// 1. EMBEDDIXDB_DATA_DIR environment variable (explicit override)
// 2. CLAUDE_WORKING_DIR environment variable + /.embeddixdb/
// 3. Current working directory + /.embeddixdb/
// 4. ~/.embeddixdb/default/ (global fallback)
//
// Safety considerations:
// - Creates a dedicated .embeddixdb/ directory to avoid conflicts
// - Uses safe file permissions (0755 for directories, 0600 for BoltDB files)
// - Avoids overwriting existing user files by using dedicated subdirectory
func resolveDatabasePath() (string, error) {
	// Priority 1: Explicit override via environment variable
	if dataDir := os.Getenv("EMBEDDIXDB_DATA_DIR"); dataDir != "" {
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			return "", fmt.Errorf("failed to create data directory %s: %v", dataDir, err)
		}
		return dataDir, nil
	}

	// Priority 2: Claude working directory
	if claudeDir := os.Getenv("CLAUDE_WORKING_DIR"); claudeDir != "" {
		dbPath := filepath.Join(claudeDir, ".embeddixdb")
		if err := os.MkdirAll(dbPath, 0755); err != nil {
			return "", fmt.Errorf("failed to create database directory %s: %v", dbPath, err)
		}
		return dbPath, nil
	}

	// Priority 3: Current working directory
	if cwd, err := os.Getwd(); err == nil {
		dbPath := filepath.Join(cwd, ".embeddixdb")
		if err := os.MkdirAll(dbPath, 0755); err == nil {
			return dbPath, nil
		}
		// If we can't create in current directory, continue to fallback
	}

	// Priority 4: Global fallback in user home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %v", err)
	}
	
	globalPath := filepath.Join(homeDir, ".embeddixdb", "default")
	if err := os.MkdirAll(globalPath, 0755); err != nil {
		return "", fmt.Errorf("failed to create global database directory %s: %v", globalPath, err)
	}
	
	return globalPath, nil
}

// checkDatabaseSafety performs safety checks on the resolved database path
// to warn about potential conflicts with existing files
func checkDatabaseSafety(dbPath string, persistenceType persistence.PersistenceType) {
	if persistenceType == persistence.PersistenceMemory {
		return // Memory backend creates no files
	}
	
	// For BoltDB, show what file will be created/used
	if persistenceType == persistence.PersistenceBolt {
		// dbPath is already the full file path for BoltDB
		if _, err := os.Stat(dbPath); err == nil {
			fmt.Fprintf(os.Stderr, "ℹ️  Using existing BoltDB file: %s\n", dbPath)
		} else {
			fmt.Fprintf(os.Stderr, "ℹ️  Will create new BoltDB file: %s\n", dbPath)
		}
		
		// Check the directory for potential conflicts
		dbDir := filepath.Dir(dbPath)
		conflictFiles := []string{
			"embeddixdb.db",      // Alternative naming
			"database.db",        // Common database file
			"data.db",           // Generic data file
		}
		
		for _, filename := range conflictFiles {
			if filename == filepath.Base(dbPath) {
				continue // Skip the actual database file
			}
			fullPath := filepath.Join(dbDir, filename)
			if _, err := os.Stat(fullPath); err == nil {
				fmt.Fprintf(os.Stderr, "⚠️  WARNING: Other database file found: %s\n", fullPath)
				fmt.Fprintf(os.Stderr, "   Multiple database files in same directory may cause confusion.\n")
			}
		}
	}
	
	// For BadgerDB, show directory usage
	if persistenceType == persistence.PersistenceBadger {
		fmt.Fprintf(os.Stderr, "ℹ️  BadgerDB will use directory: %s\n", dbPath)
		if entries, err := os.ReadDir(dbPath); err == nil && len(entries) > 0 {
			fmt.Fprintf(os.Stderr, "   Directory contains %d existing files/subdirectories\n", len(entries))
		}
	}
}

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
	} else {
		// Use dynamic path resolution when no explicit path is provided
		resolvedPath, err := resolveDatabasePath()
		if err != nil {
			log.Fatalf("Failed to resolve database path: %v", err)
		}
		
		// Adjust path based on persistence type
		finalPath := resolvedPath
		if cfg.Persistence.Type == persistence.PersistenceBolt {
			// BoltDB needs a file path, not directory
			finalPath = filepath.Join(resolvedPath, "embeddix.db")
		}
		// BadgerDB and Memory use directory path as-is
		
		cfg.Persistence.Path = finalPath
		cfg.Persistence.Options["path"] = finalPath
	}
	if *modelPath != "" {
		cfg.AI.Embedding.ONNX.ModelPath = *modelPath
	}
	if *modelName != "" {
		cfg.AI.Embedding.Model = *modelName
	}

	// Create persistence configuration
	persistenceConfig := cfg.Persistence
	
	// Perform safety checks for database path
	checkDatabaseSafety(cfg.Persistence.Path, cfg.Persistence.Type)

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
	
	// Show how the path was resolved
	if *dataPath != "" {
		fmt.Fprintf(os.Stderr, "  Path Source: Command line flag --data\n")
	} else {
		var pathSource string
		if os.Getenv("EMBEDDIXDB_DATA_DIR") != "" {
			pathSource = "Environment variable EMBEDDIXDB_DATA_DIR"
		} else if os.Getenv("CLAUDE_WORKING_DIR") != "" {
			pathSource = "Claude working directory + .embeddixdb"
		} else if cwd, err := os.Getwd(); err == nil {
			// Check if the path starts with current directory
			expectedDir := filepath.Join(cwd, ".embeddixdb")
			if strings.HasPrefix(cfg.Persistence.Path, expectedDir) {
				pathSource = "Current working directory + .embeddixdb"
			} else {
				pathSource = "Global fallback (~/.embeddixdb/default)"
			}
		} else {
			pathSource = "Global fallback (~/.embeddixdb/default)"
		}
		fmt.Fprintf(os.Stderr, "  Path Source: %s\n", pathSource)
	}
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
