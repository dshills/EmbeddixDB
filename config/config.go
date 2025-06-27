package config

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/dshills/EmbeddixDB/api"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/persistence"
	"gopkg.in/yaml.v3"
)

// Config represents the complete EmbeddixDB configuration
type Config struct {
	// Server configuration
	Server ServerConfig `yaml:"server" json:"server"`

	// Persistence configuration
	Persistence persistence.PersistenceConfig `yaml:"persistence" json:"persistence"`

	// AI/Embedding configuration
	AI AIConfig `yaml:"ai" json:"ai"`

	// Vector store configuration
	VectorStore VectorStoreConfig `yaml:"vector_store" json:"vector_store"`

	// Logging configuration
	Logging LoggingConfig `yaml:"logging" json:"logging"`
}

// ServerConfig contains server-related configuration
type ServerConfig struct {
	Host            string        `yaml:"host" json:"host"`
	Port            int           `yaml:"port" json:"port"`
	ReadTimeout     time.Duration `yaml:"read_timeout" json:"read_timeout"`
	WriteTimeout    time.Duration `yaml:"write_timeout" json:"write_timeout"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout" json:"shutdown_timeout"`
}

// AIConfig contains AI and embedding-related configuration
type AIConfig struct {
	// Embedding configuration
	Embedding EmbeddingConfig `yaml:"embedding" json:"embedding"`

	// Content analysis configuration
	ContentAnalysis ContentAnalysisConfig `yaml:"content_analysis" json:"content_analysis"`

	// Query understanding configuration
	QueryUnderstanding QueryUnderstandingConfig `yaml:"query_understanding" json:"query_understanding"`
}

// EmbeddingConfig contains embedding engine configuration
type EmbeddingConfig struct {
	// Engine type: "onnx", "ollama", etc.
	Engine string `yaml:"engine" json:"engine"`

	// Model name
	Model string `yaml:"model" json:"model"`

	// Ollama-specific configuration
	Ollama OllamaConfig `yaml:"ollama" json:"ollama"`

	// ONNX-specific configuration
	ONNX ONNXConfig `yaml:"onnx" json:"onnx"`

	// Batch size for embedding generation
	BatchSize int `yaml:"batch_size" json:"batch_size"`

	// Maximum queue size
	MaxQueueSize int `yaml:"max_queue_size" json:"max_queue_size"`
}

// OllamaConfig contains Ollama-specific configuration
type OllamaConfig struct {
	// Endpoint URL for Ollama API
	Endpoint string `yaml:"endpoint" json:"endpoint"`

	// Request timeout
	Timeout time.Duration `yaml:"timeout" json:"timeout"`

	// API key (if required)
	APIKey string `yaml:"api_key" json:"api_key"`
}

// ONNXConfig contains ONNX-specific configuration
type ONNXConfig struct {
	// Path to ONNX model file
	ModelPath string `yaml:"model_path" json:"model_path"`

	// Use GPU acceleration
	UseGPU bool `yaml:"use_gpu" json:"use_gpu"`

	// Number of threads
	NumThreads int `yaml:"num_threads" json:"num_threads"`
}

// ContentAnalysisConfig contains content analysis configuration
type ContentAnalysisConfig struct {
	// Enable sentiment analysis
	EnableSentiment bool `yaml:"enable_sentiment" json:"enable_sentiment"`

	// Enable entity extraction
	EnableEntities bool `yaml:"enable_entities" json:"enable_entities"`

	// Enable key phrase extraction
	EnableKeyPhrases bool `yaml:"enable_key_phrases" json:"enable_key_phrases"`

	// Enable language detection
	EnableLanguageDetection bool `yaml:"enable_language_detection" json:"enable_language_detection"`
}

// QueryUnderstandingConfig contains query understanding configuration
type QueryUnderstandingConfig struct {
	// Enable semantic query understanding
	Enabled bool `yaml:"enabled" json:"enabled"`

	// Confidence threshold
	ConfidenceThreshold float32 `yaml:"confidence_threshold" json:"confidence_threshold"`
}

// VectorStoreConfig contains vector store configuration
type VectorStoreConfig struct {
	// Default distance metric
	DefaultDistanceMetric string `yaml:"default_distance_metric" json:"default_distance_metric"`

	// Default index type
	DefaultIndexType string `yaml:"default_index_type" json:"default_index_type"`

	// HNSW configuration
	HNSW HNSWConfig `yaml:"hnsw" json:"hnsw"`

	// Cache configuration
	Cache CacheConfig `yaml:"cache" json:"cache"`
}

// HNSWConfig contains HNSW index configuration
type HNSWConfig struct {
	M              int `yaml:"m" json:"m"`
	EfConstruction int `yaml:"ef_construction" json:"ef_construction"`
	EfSearch       int `yaml:"ef_search" json:"ef_search"`
	MaxConnections int `yaml:"max_connections" json:"max_connections"`
}

// CacheConfig contains cache configuration
type CacheConfig struct {
	// Enable caching
	Enabled bool `yaml:"enabled" json:"enabled"`

	// Maximum cache size in MB
	MaxSizeMB int `yaml:"max_size_mb" json:"max_size_mb"`

	// Cache TTL
	TTL time.Duration `yaml:"ttl" json:"ttl"`
}

// LoggingConfig contains logging configuration
type LoggingConfig struct {
	Level  string `yaml:"level" json:"level"`
	Format string `yaml:"format" json:"format"`
	Output string `yaml:"output" json:"output"`
}

// LoadConfig loads configuration from various sources with the following precedence:
// 1. Command-line flags (if provided)
// 2. Environment variables
// 3. Configuration file (~/.embeddixdb.yml or specified path)
// 4. Default values
func LoadConfig(configPath string) (*Config, error) {
	config := DefaultConfig()

	// If no config path specified, try default location
	if configPath == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			configPath = filepath.Join(homeDir, ".embeddixdb.yml")
		}
	}

	// Load from file if it exists
	if configPath != "" {
		if err := loadConfigFromFile(configPath, config); err != nil {
			// Only return error if file exists but can't be read
			if !os.IsNotExist(err) {
				return nil, fmt.Errorf("failed to load config from %s: %w", configPath, err)
			}
		}
	}

	// Override with environment variables
	loadConfigFromEnv(config)

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return config, nil
}

// loadConfigFromFile loads configuration from a YAML file
func loadConfigFromFile(path string, config *Config) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	return yaml.Unmarshal(data, config)
}

// loadConfigFromEnv loads configuration from environment variables
func loadConfigFromEnv(config *Config) {
	// Server configuration
	if host := os.Getenv("EMBEDDIXDB_HOST"); host != "" {
		config.Server.Host = host
	}
	if port := os.Getenv("EMBEDDIXDB_PORT"); port != "" {
		if p, err := parsePort(port); err == nil {
			config.Server.Port = p
		}
	}

	// Ollama configuration
	if endpoint := os.Getenv("EMBEDDIXDB_OLLAMA_ENDPOINT"); endpoint != "" {
		config.AI.Embedding.Ollama.Endpoint = endpoint
	}
	if apiKey := os.Getenv("EMBEDDIXDB_OLLAMA_API_KEY"); apiKey != "" {
		config.AI.Embedding.Ollama.APIKey = apiKey
	}

	// Persistence configuration
	if backend := os.Getenv("EMBEDDIXDB_PERSISTENCE_BACKEND"); backend != "" {
		config.Persistence.Type = persistence.PersistenceType(backend)
	}
	if path := os.Getenv("EMBEDDIXDB_PERSISTENCE_PATH"); path != "" {
		config.Persistence.Options["path"] = path
	}
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:            "0.0.0.0",
			Port:            8080,
			ReadTimeout:     30 * time.Second,
			WriteTimeout:    30 * time.Second,
			ShutdownTimeout: 10 * time.Second,
		},
		Persistence: persistence.PersistenceConfig{
			Type:    persistence.PersistenceMemory,
			Path:    "data/embeddix.db",
			Options: map[string]interface{}{},
		},
		AI: AIConfig{
			Embedding: EmbeddingConfig{
				Engine:       "onnx",
				Model:        "all-MiniLM-L6-v2",
				BatchSize:    32,
				MaxQueueSize: 1000,
				Ollama: OllamaConfig{
					Endpoint: "http://localhost:11434",
					Timeout:  30 * time.Second,
				},
				ONNX: ONNXConfig{
					ModelPath:  "models/all-MiniLM-L6-v2.onnx",
					UseGPU:     false,
					NumThreads: 4,
				},
			},
			ContentAnalysis: ContentAnalysisConfig{
				EnableSentiment:         true,
				EnableEntities:          true,
				EnableKeyPhrases:        true,
				EnableLanguageDetection: true,
			},
			QueryUnderstanding: QueryUnderstandingConfig{
				Enabled:             true,
				ConfidenceThreshold: 0.7,
			},
		},
		VectorStore: VectorStoreConfig{
			DefaultDistanceMetric: "cosine",
			DefaultIndexType:      "hnsw",
			HNSW: HNSWConfig{
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				MaxConnections: 32,
			},
			Cache: CacheConfig{
				Enabled:   true,
				MaxSizeMB: 512,
				TTL:       1 * time.Hour,
			},
		},
		Logging: LoggingConfig{
			Level:  "info",
			Format: "json",
			Output: "stdout",
		},
	}
}

// Validate validates the configuration
func (c *Config) Validate() error {
	// Validate server config
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port number: %d", c.Server.Port)
	}

	// Validate AI config
	if c.AI.Embedding.Engine == "ollama" {
		if c.AI.Embedding.Ollama.Endpoint == "" {
			return fmt.Errorf("ollama endpoint is required when using ollama engine")
		}
	}

	// Validate persistence config
	if err := persistence.ValidateConfig(c.Persistence); err != nil {
		return fmt.Errorf("persistence config validation failed: %w", err)
	}

	// Validate vector store config
	validMetrics := map[string]bool{
		"cosine": true,
		"l2":     true,
		"dot":    true,
	}
	if !validMetrics[c.VectorStore.DefaultDistanceMetric] {
		return fmt.Errorf("invalid distance metric: %s", c.VectorStore.DefaultDistanceMetric)
	}

	return nil
}

// ToModelConfig converts the embedding configuration to ai.ModelConfig
func (e *EmbeddingConfig) ToModelConfig() ai.ModelConfig {
	config := ai.ModelConfig{
		Type:                ai.ModelTypeONNX,
		Name:                e.Model,
		BatchSize:           e.BatchSize,
		MaxConcurrency:      10, // Default
		NormalizeEmbeddings: true,
	}

	switch e.Engine {
	case "ollama":
		config.Type = ai.ModelTypeOllama
		config.Path = e.Model // Ollama expects model name in Path field
		config.OllamaEndpoint = e.Ollama.Endpoint
		if e.Ollama.Timeout > 0 {
			config.TimeoutDuration = e.Ollama.Timeout
		}
	case "onnx":
		config.Type = ai.ModelTypeONNX
		config.Path = e.ONNX.ModelPath
		config.EnableGPU = e.ONNX.UseGPU
		config.NumThreads = e.ONNX.NumThreads
	}

	return config
}

// ToServerConfig converts to api.ServerConfig
func (s *ServerConfig) ToServerConfig() api.ServerConfig {
	return api.ServerConfig{
		Host:            s.Host,
		Port:            s.Port,
		ReadTimeout:     s.ReadTimeout,
		WriteTimeout:    s.WriteTimeout,
		IdleTimeout:     60 * time.Second, // Default idle timeout
		ShutdownTimeout: s.ShutdownTimeout,
	}
}

// parsePort parses a port string to int
func parsePort(s string) (int, error) {
	var port int
	_, err := fmt.Sscanf(s, "%d", &port)
	return port, err
}
