package ai

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ModelRegistry manages available embedding models and their metadata
type ModelRegistry struct {
	models       map[string]*ModelEntry
	downloadPath string
	mutex        sync.RWMutex
}

// ModelEntry represents a model in the registry
type ModelEntry struct {
	Info         ModelInfo         `json:"info"`
	Config       ModelConfig       `json:"config"`
	Source       ModelSource       `json:"source"`
	Dependencies []string          `json:"dependencies"`
	Tags         []string          `json:"tags"`
	Examples     []ModelExample    `json:"examples"`
	Performance  ModelPerformance  `json:"performance"`
	Verification ModelVerification `json:"verification"`
}

// ModelSource defines where and how to obtain a model
type ModelSource struct {
	Type       string            `json:"type"`       // huggingface, url, local
	Repository string            `json:"repository"` // For HuggingFace models
	URL        string            `json:"url"`        // Direct download URL
	Path       string            `json:"path"`       // Local file path
	Revision   string            `json:"revision"`   // Git revision/branch
	AuthToken  string            `json:"auth_token"` // Authentication token
	Headers    map[string]string `json:"headers"`    // HTTP headers
	Checksum   string            `json:"checksum"`   // SHA256 checksum
}

// ModelExample shows how to use the model
type ModelExample struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Input       []string `json:"input"`
	UseCase     string   `json:"use_case"`
}

// ModelPerformance contains benchmark results
type ModelPerformance struct {
	Latency     map[string]float64 `json:"latency_ms"`     // By batch size
	Throughput  map[string]float64 `json:"throughput_tps"` // By batch size
	MemoryUsage int64              `json:"memory_usage_mb"`
	Accuracy    float64            `json:"accuracy_score"`
	Benchmarks  []BenchmarkResult  `json:"benchmarks"`
}

// BenchmarkResult represents a specific benchmark result
type BenchmarkResult struct {
	Dataset    string  `json:"dataset"`
	Metric     string  `json:"metric"`
	Score      float64 `json:"score"`
	Rank       int     `json:"rank"`
	ComparedTo int     `json:"compared_to"`
}

// ModelVerification contains integrity and security information
type ModelVerification struct {
	Verified    bool        `json:"verified"`
	Signatures  []string    `json:"signatures"`
	Publisher   string      `json:"publisher"`
	ScanResults ScanResults `json:"scan_results"`
}

// ScanResults contains security scan results
type ScanResults struct {
	Scanned        bool     `json:"scanned"`
	Clean          bool     `json:"clean"`
	Threats        []string `json:"threats"`
	LastScanned    string   `json:"last_scanned"`
	ScannerVersion string   `json:"scanner_version"`
}

// NewModelRegistry creates a new model registry
func NewModelRegistry() *ModelRegistry {
	homeDir, _ := os.UserHomeDir()
	downloadPath := filepath.Join(homeDir, ".embeddixdb", "models")

	registry := &ModelRegistry{
		models:       make(map[string]*ModelEntry),
		downloadPath: downloadPath,
	}

	// Create download directory
	os.MkdirAll(downloadPath, 0755)

	// Load default models
	registry.loadDefaultModels()

	return registry
}

// RegisterModel adds a model to the registry
func (r *ModelRegistry) RegisterModel(name string, entry *ModelEntry) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	// Validate model entry
	if err := r.validateModelEntry(entry); err != nil {
		return fmt.Errorf("invalid model entry: %w", err)
	}

	r.models[name] = entry
	return nil
}

// GetModel retrieves a model entry by name
func (r *ModelRegistry) GetModel(name string) (*ModelEntry, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	entry, exists := r.models[name]
	if !exists {
		return nil, fmt.Errorf("model %s not found in registry", name)
	}

	return entry, nil
}

// ListModels returns all registered models
func (r *ModelRegistry) ListModels() map[string]*ModelEntry {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	// Return a copy to prevent external modification
	models := make(map[string]*ModelEntry)
	for name, entry := range r.models {
		models[name] = entry
	}

	return models
}

// SearchModels finds models matching criteria
func (r *ModelRegistry) SearchModels(criteria SearchCriteria) []*ModelSearchResult {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	var results []*ModelSearchResult

	for name, entry := range r.models {
		if r.matchesCriteria(entry, criteria) {
			result := &ModelSearchResult{
				Name:      name,
				Entry:     entry,
				Relevance: r.calculateRelevance(entry, criteria),
			}
			results = append(results, result)
		}
	}

	// Sort by relevance (simplified sorting)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Relevance < results[j].Relevance {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results
}

// DownloadModel downloads a model from its source
func (r *ModelRegistry) DownloadModel(name string) (string, error) {
	entry, err := r.GetModel(name)
	if err != nil {
		return "", err
	}

	// Check if already downloaded
	localPath := r.getLocalPath(name)
	if r.isModelDownloaded(localPath) {
		return localPath, nil
	}

	// Download based on source type
	switch entry.Source.Type {
	case "url":
		return r.downloadFromURL(name, entry.Source.URL, entry.Source.Checksum)
	case "huggingface":
		return r.downloadFromHuggingFace(name, entry.Source)
	case "local":
		return entry.Source.Path, nil
	default:
		return "", fmt.Errorf("unsupported source type: %s", entry.Source.Type)
	}
}

// GetLocalPath returns the local path for a model
func (r *ModelRegistry) GetLocalPath(name string) string {
	return r.getLocalPath(name)
}

// IsModelDownloaded checks if a model is available locally
func (r *ModelRegistry) IsModelDownloaded(name string) bool {
	localPath := r.getLocalPath(name)
	return r.isModelDownloaded(localPath)
}

// loadDefaultModels loads a set of popular pre-configured models
// createSentenceTransformerModel creates a model entry for sentence transformer models
func createSentenceTransformerModel(name string, dimension int, size int64, accuracy, speed float64, tags []string) *ModelEntry {
	return &ModelEntry{
		Info: ModelInfo{
			Name:       name,
			Version:    "1.0",
			Dimension:  dimension,
			MaxTokens:  512,
			Languages:  []string{"en"},
			Modalities: []string{"text"},
			License:    "Apache-2.0",
			Size:       size,
			Accuracy:   accuracy,
			Speed:      int(speed),
		},
		Config: ModelConfig{
			Name:                name,
			Type:                "onnx",
			BatchSize:           16,
			MaxConcurrency:      2,
			EnableGPU:           true,
			OptimizationLevel:   1,
			NumThreads:          4,
			NormalizeEmbeddings: true,
		},
		Source: ModelSource{
			Type:       "huggingface",
			Repository: "sentence-transformers/" + name,
			Revision:   "main",
		},
		Tags: tags,
		Examples: []ModelExample{
			{
				Name:        "Semantic Search",
				Description: "Find similar documents based on meaning",
				Input:       []string{"What is machine learning?", "How does AI work?"},
				UseCase:     "knowledge_base",
			},
		},
		Performance: ModelPerformance{
			Latency: map[string]float64{
				"1":  50,
				"8":  80,
				"32": 150,
			},
			Throughput: map[string]float64{
				"1":  20,
				"8":  100,
				"32": 200,
			},
			MemoryUsage: size / (1024 * 1024), // Convert to MB
			Accuracy:    accuracy,
		},
	}
}

func (r *ModelRegistry) loadDefaultModels() {
	// Add popular sentence transformers models
	defaultModels := map[string]*ModelEntry{
		"all-MiniLM-L6-v2": createSentenceTransformerModel(
			"all-MiniLM-L6-v2",
			384,
			90*1024*1024,
			0.85,
			1000,
			[]string{"general", "fast", "small", "english"},
		),
		"all-mpnet-base-v2": createSentenceTransformerModel(
			"all-mpnet-base-v2",
			768,
			420*1024*1024,
			0.92,
			500,
			[]string{"general", "high-quality", "large", "english"},
		),
	}

	for name, entry := range defaultModels {
		r.models[name] = entry
	}
}

// Helper methods

func (r *ModelRegistry) validateModelEntry(entry *ModelEntry) error {
	if entry.Info.Name == "" {
		return fmt.Errorf("model name is required")
	}
	if entry.Info.Dimension <= 0 {
		return fmt.Errorf("model dimension must be positive")
	}
	if entry.Source.Type == "" {
		return fmt.Errorf("source type is required")
	}
	return nil
}

func (r *ModelRegistry) getLocalPath(name string) string {
	safeName := strings.ReplaceAll(name, "/", "_")
	return filepath.Join(r.downloadPath, safeName)
}

func (r *ModelRegistry) isModelDownloaded(localPath string) bool {
	_, err := os.Stat(localPath)
	return err == nil
}

func (r *ModelRegistry) downloadFromURL(name, url, checksum string) (string, error) {
	localPath := r.getLocalPath(name)

	// Create directory if needed
	if err := os.MkdirAll(filepath.Dir(localPath), 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	// Download the file
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed with status: %s", resp.Status)
	}

	// Create output file
	out, err := os.Create(localPath)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	// Copy data
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	// TODO: Verify checksum if provided

	return localPath, nil
}

func (r *ModelRegistry) downloadFromHuggingFace(name string, source ModelSource) (string, error) {
	// For now, return a mock path - real implementation would use HuggingFace Hub API
	localPath := r.getLocalPath(name)

	// Create a mock ONNX file to simulate download
	if err := os.MkdirAll(filepath.Dir(localPath), 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	// Create empty file to simulate model
	file, err := os.Create(localPath + ".onnx")
	if err != nil {
		return "", fmt.Errorf("failed to create mock model file: %w", err)
	}
	file.Close()

	return localPath + ".onnx", nil
}

func (r *ModelRegistry) matchesCriteria(entry *ModelEntry, criteria SearchCriteria) bool {
	// Check language
	if len(criteria.Languages) > 0 {
		found := false
		for _, lang := range criteria.Languages {
			for _, modelLang := range entry.Info.Languages {
				if lang == modelLang {
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check dimension range
	if criteria.MinDimension > 0 && entry.Info.Dimension < criteria.MinDimension {
		return false
	}
	if criteria.MaxDimension > 0 && entry.Info.Dimension > criteria.MaxDimension {
		return false
	}

	// Check tags
	if len(criteria.Tags) > 0 {
		found := false
		for _, tag := range criteria.Tags {
			for _, modelTag := range entry.Tags {
				if tag == modelTag {
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

func (r *ModelRegistry) calculateRelevance(entry *ModelEntry, criteria SearchCriteria) float64 {
	relevance := 0.0

	// Base relevance from accuracy
	relevance += entry.Info.Accuracy * 0.4

	// Bonus for matching tags
	if len(criteria.Tags) > 0 {
		matches := 0
		for _, tag := range criteria.Tags {
			for _, modelTag := range entry.Tags {
				if tag == modelTag {
					matches++
				}
			}
		}
		relevance += float64(matches) / float64(len(criteria.Tags)) * 0.3
	}

	// Performance factor
	if throughput, exists := entry.Performance.Throughput["32"]; exists {
		relevance += (throughput / 1000.0) * 0.2 // Normalize by expected max throughput
	}

	// Size penalty (smaller is better for mobile/edge)
	sizeMB := float64(entry.Info.Size) / (1024 * 1024)
	if sizeMB > 0 {
		relevance += (1000.0 / sizeMB) * 0.1 // Inverse size factor
	}

	return relevance
}

// SearchCriteria defines search parameters for finding models
type SearchCriteria struct {
	Languages    []string `json:"languages"`
	MinDimension int      `json:"min_dimension"`
	MaxDimension int      `json:"max_dimension"`
	Tags         []string `json:"tags"`
	UseCase      string   `json:"use_case"`
	MaxSize      int64    `json:"max_size_mb"`
	MinAccuracy  float64  `json:"min_accuracy"`
}

// ModelSearchResult represents a search result
type ModelSearchResult struct {
	Name      string      `json:"name"`
	Entry     *ModelEntry `json:"entry"`
	Relevance float64     `json:"relevance"`
}
