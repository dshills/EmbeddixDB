package ai

import (
	"context"
	"time"
)

// EmbeddingEngine provides the core interface for embedding generation
type EmbeddingEngine interface {
	// Embed generates embeddings for a list of content strings
	Embed(ctx context.Context, content []string) ([][]float32, error)

	// EmbedBatch processes content in batches for optimal performance
	EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error)

	// GetModelInfo returns metadata about the loaded model
	GetModelInfo() ModelInfo

	// Warm preloads the model for faster inference
	Warm(ctx context.Context) error

	// Close releases model resources
	Close() error
}

// ModelInfo contains metadata about embedding models
type ModelInfo struct {
	Name       string   `json:"name"`
	Version    string   `json:"version"`
	Dimension  int      `json:"dimension"`
	MaxTokens  int      `json:"max_tokens"`
	Languages  []string `json:"languages"`
	Modalities []string `json:"modalities"` // text, image, audio
	License    string   `json:"license"`
	Size       int64    `json:"size_bytes"`
	Accuracy   float64  `json:"accuracy_score"`
	Speed      int      `json:"tokens_per_second"`
}

// ModelConfig defines model loading configuration
type ModelConfig struct {
	Name                string        `json:"name"`
	Type                string        `json:"type"` // onnx, huggingface, custom
	Path                string        `json:"path"`
	BatchSize           int           `json:"batch_size"`
	MaxConcurrency      int           `json:"max_concurrency"`
	TimeoutDuration     time.Duration `json:"timeout_duration"`
	EnableGPU           bool          `json:"enable_gpu"`
	OptimizationLevel   int           `json:"optimization_level"`
	NumThreads          int           `json:"num_threads"`
	NormalizeEmbeddings bool          `json:"normalize_embeddings"`
	MaxTokens           int           `json:"max_tokens"`
	PoolingStrategy     string        `json:"pooling_strategy"` // cls, mean, max
}

// HybridSearchEngine combines vector and text search capabilities
type HybridSearchEngine interface {
	// Search performs hybrid search combining vector and text results
	Search(ctx context.Context, query HybridSearchRequest) (HybridSearchResult, error)

	// AddDocuments adds documents to both vector and text indices
	AddDocuments(ctx context.Context, docs []Document) error

	// UpdateFusionWeights adjusts how vector and text results are combined
	UpdateFusionWeights(weights SearchWeights) error

	// GetStats returns search performance statistics
	GetStats() SearchStats
}

// HybridSearchRequest defines a hybrid search query
type HybridSearchRequest struct {
	Query      string                 `json:"query"`
	Collection string                 `json:"collection"`
	Limit      int                    `json:"limit"`
	Filters    map[string]interface{} `json:"filters"`
	Weights    SearchWeights          `json:"weights"`
	Options    SearchOptions          `json:"options"`
}

// SearchWeights controls fusion of different search methods
type SearchWeights struct {
	Vector    float64 `json:"vector"`    // Vector similarity weight
	Text      float64 `json:"text"`      // Text relevance weight
	Freshness float64 `json:"freshness"` // Temporal relevance weight
	Authority float64 `json:"authority"` // Source authority weight
}

// SearchOptions configures search behavior
type SearchOptions struct {
	IncludeExplanation bool    `json:"include_explanation"`
	ExpandQuery        bool    `json:"expand_query"`
	Rerank             bool    `json:"rerank"`
	MinScore           float64 `json:"min_score"`
	FusionAlgorithm    string  `json:"fusion_algorithm"` // rrf, linear, learned
}

// HybridSearchResult contains search results and metadata
type HybridSearchResult struct {
	Results     []SearchResult `json:"results"`
	QueryInfo   QueryInfo      `json:"query_info"`
	Performance SearchPerf     `json:"performance"`
	DebugInfo   *DebugInfo     `json:"debug_info,omitempty"`
}

// SearchResult represents a single search result
type SearchResult struct {
	ID          string                 `json:"id"`
	Score       float64                `json:"score"`
	Content     string                 `json:"content"`
	Metadata    map[string]interface{} `json:"metadata"`
	Explanation *SearchExplanation     `json:"explanation,omitempty"`
}

// SearchExplanation provides details about result scoring
type SearchExplanation struct {
	VectorScore  float64  `json:"vector_score"`
	TextScore    float64  `json:"text_score"`
	FusionScore  float64  `json:"fusion_score"`
	MatchedTerms []string `json:"matched_terms"`
	SemanticSim  float64  `json:"semantic_similarity"`
}

// QueryInfo contains processed query information
type QueryInfo struct {
	Original  string         `json:"original"`
	Processed string         `json:"processed"`
	Intent    QueryIntent    `json:"intent"`
	Entities  []Entity       `json:"entities"`
	Expansion QueryExpansion `json:"expansion"`
}

// QueryIntent represents classified query intent
type QueryIntent struct {
	Type       string  `json:"type"` // factual, exploratory, etc.
	Confidence float64 `json:"confidence"`
	Subtype    string  `json:"subtype"`
}

// Entity represents extracted named entities
type Entity struct {
	Text       string  `json:"text"`
	Label      string  `json:"label"` // PERSON, ORG, LOC, etc.
	Confidence float64 `json:"confidence"`
	StartPos   int     `json:"start_pos"`
	EndPos     int     `json:"end_pos"`
}

// QueryExpansion contains expanded query terms
type QueryExpansion struct {
	Synonyms      []string           `json:"synonyms"`
	Concepts      []string           `json:"concepts"`
	RelatedTerms  []string           `json:"related_terms"`
	WeightedTerms map[string]float64 `json:"weighted_terms"`
}

// SearchPerf tracks search performance metrics
type SearchPerf struct {
	TotalTimeMs    int64 `json:"total_time_ms"`
	VectorSearchMs int64 `json:"vector_search_ms"`
	TextSearchMs   int64 `json:"text_search_ms"`
	FusionMs       int64 `json:"fusion_ms"`
	RerankMs       int64 `json:"rerank_ms"`
}

// DebugInfo provides debugging information for search
type DebugInfo struct {
	VectorCandidates int `json:"vector_candidates"`
	TextCandidates   int `json:"text_candidates"`
	FusionCandidates int `json:"fusion_candidates"`
	FinalResults     int `json:"final_results"`
}

// SearchStats tracks overall search statistics
type SearchStats struct {
	TotalQueries   int64         `json:"total_queries"`
	AverageLatency time.Duration `json:"average_latency"`
	P95Latency     time.Duration `json:"p95_latency"`
	SuccessRate    float64       `json:"success_rate"`
	CacheHitRate   float64       `json:"cache_hit_rate"`
}

// ContentAnalyzer provides automatic content understanding
type ContentAnalyzer interface {
	// AnalyzeContent extracts insights from content
	AnalyzeContent(ctx context.Context, content string) (ContentInsights, error)

	// AnalyzeBatch processes multiple content items
	AnalyzeBatch(ctx context.Context, content []string) ([]ContentInsights, error)

	// ExtractEntities finds named entities in content
	ExtractEntities(ctx context.Context, content string) ([]Entity, error)

	// DetectLanguage identifies content language
	DetectLanguage(ctx context.Context, content string) (LanguageInfo, error)
}

// ContentInsights represents analyzed content metadata
type ContentInsights struct {
	Language    LanguageInfo   `json:"language"`
	Topics      []Topic        `json:"topics"`
	Entities    []Entity       `json:"entities"`
	Sentiment   SentimentScore `json:"sentiment"`
	Complexity  float64        `json:"complexity_score"`
	Readability float64        `json:"readability_score"`
	KeyPhrases  []string       `json:"key_phrases"`
	Summary     string         `json:"summary"`
	WordCount   int            `json:"word_count"`
}

// LanguageInfo contains language detection results
type LanguageInfo struct {
	Code       string  `json:"code"`       // ISO 639-1 code (en, es, etc.)
	Name       string  `json:"name"`       // English name
	Confidence float64 `json:"confidence"` // Detection confidence
}

// Topic represents discovered content topics
type Topic struct {
	ID         string   `json:"id"`
	Label      string   `json:"label"`
	Keywords   []string `json:"keywords"`
	Confidence float64  `json:"confidence"`
	Weight     float64  `json:"weight"`
}

// SentimentScore represents sentiment analysis results
type SentimentScore struct {
	Polarity   float64 `json:"polarity"`   // -1.0 to 1.0
	Confidence float64 `json:"confidence"` // 0.0 to 1.0
	Label      string  `json:"label"`      // positive, negative, neutral
}

// Document represents a document for embedding and indexing
type Document struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
	Type     string                 `json:"type"` // text, markdown, html, etc.
}

// ModelManager handles embedding model lifecycle
type ModelManager interface {
	// LoadModel loads a model for inference
	LoadModel(ctx context.Context, modelName string, config ModelConfig) error

	// UnloadModel releases model resources
	UnloadModel(modelName string) error

	// GetEngine returns a loaded embedding engine
	GetEngine(modelName string) (EmbeddingEngine, error)

	// ListModels returns available models
	ListModels() []ModelInfo

	// GetModelHealth returns model health status
	GetModelHealth(modelName string) (ModelHealth, error)
}

// ModelHealth represents model health status
type ModelHealth struct {
	ModelName    string        `json:"model_name"`
	Status       string        `json:"status"` // healthy, unhealthy, loading
	LoadedAt     time.Time     `json:"loaded_at"`
	LastCheck    time.Time     `json:"last_check"`
	Latency      time.Duration `json:"latency"`
	ErrorRate    float64       `json:"error_rate"`
	MemoryUsage  int64         `json:"memory_usage_mb"`
	CPUUsage     float64       `json:"cpu_usage"`
	GPUUsage     float64       `json:"gpu_usage"`
	ErrorMessage string        `json:"error_message,omitempty"`
}

// FusionAlgorithm defines how to combine different search results
type FusionAlgorithm interface {
	// Fuse combines vector and text results
	Fuse(vectorResults, textResults []SearchResult, weights SearchWeights) []SearchResult

	// GetName returns the algorithm name
	GetName() string

	// GetParameters returns algorithm parameters
	GetParameters() map[string]interface{}
}

// TextIndex provides full-text search capabilities
type TextIndex interface {
	// Index adds documents to the text index
	Index(ctx context.Context, docs []Document) error

	// Search performs text-based search
	Search(ctx context.Context, query string, limit int) ([]SearchResult, error)

	// Delete removes a document from the index
	Delete(ctx context.Context, docID string) error

	// GetStats returns index statistics
	GetStats() TextIndexStats
}

// TextIndexStats provides text index statistics
type TextIndexStats struct {
	DocumentCount int64     `json:"document_count"`
	TermCount     int64     `json:"term_count"`
	IndexSize     int64     `json:"index_size_bytes"`
	AverageDocLen float64   `json:"average_doc_length"`
	LastUpdate    time.Time `json:"last_update"`
}
