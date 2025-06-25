package ai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// APIHandler provides HTTP handlers for AI-powered endpoints
type APIHandler struct {
	modelManager ModelManager
	vectorStore  core.VectorStore
	analyzer     ContentAnalyzer
}

// NewAPIHandler creates a new API handler
func NewAPIHandler(modelManager ModelManager, vectorStore core.VectorStore, analyzer ContentAnalyzer) *APIHandler {
	return &APIHandler{
		modelManager: modelManager,
		vectorStore:  vectorStore,
		analyzer:     analyzer,
	}
}

// EmbedRequest represents a request to embed content
type EmbedRequest struct {
	Content      []string                 `json:"content"`
	ModelName    string                   `json:"model_name,omitempty"`
	Collection   string                   `json:"collection"`
	Metadata     []map[string]interface{} `json:"metadata,omitempty"`
	ChunkSize    int                      `json:"chunk_size,omitempty"`
	ChunkOverlap int                      `json:"chunk_overlap,omitempty"`
	Analyze      bool                     `json:"analyze,omitempty"`
	Options      EmbedOptions             `json:"options,omitempty"`
}

// EmbedOptions provides configuration for embedding operations
type EmbedOptions struct {
	BatchSize     int    `json:"batch_size,omitempty"`
	Normalize     bool   `json:"normalize,omitempty"`
	Format        string `json:"format,omitempty"`         // text, markdown, html
	Language      string `json:"language,omitempty"`       // auto-detect if empty
	IncludeVector bool   `json:"include_vector,omitempty"` // return vectors in response
}

// EmbedResponse represents the response from embedding content
type EmbedResponse struct {
	Success    bool              `json:"success"`
	Message    string            `json:"message,omitempty"`
	Vectors    []core.Vector     `json:"vectors,omitempty"`
	Embeddings [][]float32       `json:"embeddings,omitempty"`
	Analysis   []ContentInsights `json:"analysis,omitempty"`
	Stats      EmbedStats        `json:"stats"`
	JobID      string            `json:"job_id,omitempty"`
}

// EmbedStats provides statistics about the embedding operation
type EmbedStats struct {
	ProcessedCount  int           `json:"processed_count"`
	ChunksCreated   int           `json:"chunks_created"`
	ProcessingTime  time.Duration `json:"processing_time_ms"`
	EmbeddingTime   time.Duration `json:"embedding_time_ms"`
	StorageTime     time.Duration `json:"storage_time_ms"`
	TokensProcessed int           `json:"tokens_processed"`
	ModelUsed       string        `json:"model_used"`
}

// BatchEmbedRequest represents a batch embedding request
type BatchEmbedRequest struct {
	Documents  []DocumentRequest `json:"documents"`
	ModelName  string            `json:"model_name,omitempty"`
	Collection string            `json:"collection"`
	Options    EmbedOptions      `json:"options,omitempty"`
}

// DocumentRequest represents a single document in a batch
type DocumentRequest struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Type     string                 `json:"type,omitempty"`
}

// HandleEmbed processes individual embedding requests
func (h *APIHandler) HandleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeErrorResponse(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate request
	if len(req.Content) == 0 {
		h.writeErrorResponse(w, "Content is required", http.StatusBadRequest)
		return
	}

	if req.Collection == "" {
		h.writeErrorResponse(w, "Collection is required", http.StatusBadRequest)
		return
	}

	startTime := time.Now()
	ctx := r.Context()

	// Get or set default model
	modelName := req.ModelName
	if modelName == "" {
		modelName = "all-MiniLM-L6-v2" // Default model
	}

	// Get embedding engine
	engine, err := h.modelManager.GetEngine(modelName)
	if err != nil {
		h.writeErrorResponse(w, fmt.Sprintf("Failed to get model: %v", err), http.StatusBadRequest)
		return
	}

	// Process content (chunking if needed)
	chunks, chunkMetadata := h.processContent(req.Content, req.Metadata, req.ChunkSize, req.ChunkOverlap)

	embeddingStart := time.Now()

	// Generate embeddings
	embeddings, err := engine.Embed(ctx, chunks)
	if err != nil {
		h.writeErrorResponse(w, fmt.Sprintf("Embedding failed: %v", err), http.StatusInternalServerError)
		return
	}

	embeddingTime := time.Since(embeddingStart)

	// Create vectors
	vectors := make([]core.Vector, len(embeddings))
	for i, embedding := range embeddings {
		// Convert metadata from map[string]interface{} to map[string]string
		stringMetadata := make(map[string]string)
		for k, v := range chunkMetadata[i] {
			stringMetadata[k] = fmt.Sprintf("%v", v)
		}

		vectors[i] = core.Vector{
			ID:       h.generateVectorID(req.Collection, i),
			Values:   embedding,
			Metadata: stringMetadata,
		}
	}

	storageStart := time.Now()

	// Store in vector database
	if err := h.vectorStore.AddVectorsBatch(r.Context(), req.Collection, vectors); err != nil {
		h.writeErrorResponse(w, fmt.Sprintf("Storage failed: %v", err), http.StatusInternalServerError)
		return
	}

	storageTime := time.Since(storageStart)

	// Perform content analysis if requested
	var analysis []ContentInsights
	if req.Analyze && h.analyzer != nil {
		analysis, err = h.analyzer.AnalyzeBatch(ctx, chunks)
		if err != nil {
			// Log error but don't fail the request
			fmt.Printf("Content analysis failed: %v\n", err)
		}
	}

	// Calculate stats
	stats := EmbedStats{
		ProcessedCount:  len(req.Content),
		ChunksCreated:   len(chunks),
		ProcessingTime:  time.Since(startTime),
		EmbeddingTime:   embeddingTime,
		StorageTime:     storageTime,
		TokensProcessed: h.estimateTokens(chunks),
		ModelUsed:       modelName,
	}

	// Prepare response
	response := EmbedResponse{
		Success: true,
		Stats:   stats,
	}

	// Include vectors in response if requested
	if req.Options.IncludeVector {
		response.Vectors = vectors
	}

	// Include embeddings if specifically requested
	if req.Options.IncludeVector {
		response.Embeddings = embeddings
	}

	if len(analysis) > 0 {
		response.Analysis = analysis
	}

	h.writeJSONResponse(w, response, http.StatusOK)
}

// HandleBatchEmbed processes batch embedding requests
func (h *APIHandler) HandleBatchEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req BatchEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeErrorResponse(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate request
	if len(req.Documents) == 0 {
		h.writeErrorResponse(w, "Documents are required", http.StatusBadRequest)
		return
	}

	if req.Collection == "" {
		h.writeErrorResponse(w, "Collection is required", http.StatusBadRequest)
		return
	}

	startTime := time.Now()
	ctx := r.Context()

	// Get or set default model
	modelName := req.ModelName
	if modelName == "" {
		modelName = "all-MiniLM-L6-v2"
	}

	// Get embedding engine
	engine, err := h.modelManager.GetEngine(modelName)
	if err != nil {
		h.writeErrorResponse(w, fmt.Sprintf("Failed to get model: %v", err), http.StatusBadRequest)
		return
	}

	// Extract content for embedding
	content := make([]string, len(req.Documents))
	for i, doc := range req.Documents {
		content[i] = doc.Content
	}

	embeddingStart := time.Now()

	// Generate embeddings using batch processing
	batchSize := req.Options.BatchSize
	if batchSize <= 0 {
		batchSize = 32 // Default batch size
	}

	embeddings, err := engine.EmbedBatch(ctx, content, batchSize)
	if err != nil {
		h.writeErrorResponse(w, fmt.Sprintf("Batch embedding failed: %v", err), http.StatusInternalServerError)
		return
	}

	embeddingTime := time.Since(embeddingStart)

	// Create vectors with document metadata
	vectors := make([]core.Vector, len(embeddings))
	for i, embedding := range embeddings {
		doc := req.Documents[i]

		// Use provided ID or generate one
		vectorID := doc.ID
		if vectorID == "" {
			vectorID = h.generateVectorID(req.Collection, i)
		}

		// Merge document metadata with type information
		metadata := doc.Metadata
		if metadata == nil {
			metadata = make(map[string]interface{})
		}
		if doc.Type != "" {
			metadata["content_type"] = doc.Type
		}
		metadata["document_id"] = doc.ID

		// Convert to string metadata
		stringMetadata := make(map[string]string)
		for k, v := range metadata {
			stringMetadata[k] = fmt.Sprintf("%v", v)
		}

		vectors[i] = core.Vector{
			ID:       vectorID,
			Values:   embedding,
			Metadata: stringMetadata,
		}
	}

	storageStart := time.Now()

	// Store in vector database
	if err := h.vectorStore.AddVectorsBatch(r.Context(), req.Collection, vectors); err != nil {
		h.writeErrorResponse(w, fmt.Sprintf("Storage failed: %v", err), http.StatusInternalServerError)
		return
	}

	storageTime := time.Since(storageStart)

	// Calculate stats
	stats := EmbedStats{
		ProcessedCount:  len(req.Documents),
		ChunksCreated:   len(vectors),
		ProcessingTime:  time.Since(startTime),
		EmbeddingTime:   embeddingTime,
		StorageTime:     storageTime,
		TokensProcessed: h.estimateTokens(content),
		ModelUsed:       modelName,
	}

	// Prepare response
	response := EmbedResponse{
		Success: true,
		Message: fmt.Sprintf("Successfully processed %d documents", len(req.Documents)),
		Stats:   stats,
	}

	// Include vectors in response if requested
	if req.Options.IncludeVector {
		response.Vectors = vectors
		response.Embeddings = embeddings
	}

	h.writeJSONResponse(w, response, http.StatusOK)
}

// HandleModelList returns available models
func (h *APIHandler) HandleModelList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	models := h.modelManager.ListModels()

	response := map[string]interface{}{
		"success": true,
		"models":  models,
		"count":   len(models),
	}

	h.writeJSONResponse(w, response, http.StatusOK)
}

// HandleModelHealth returns health status for all models
func (h *APIHandler) HandleModelHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get model name from query parameter if specified
	modelName := r.URL.Query().Get("model")

	if modelName != "" {
		// Return health for specific model
		health, err := h.modelManager.GetModelHealth(modelName)
		if err != nil {
			h.writeErrorResponse(w, fmt.Sprintf("Model not found: %v", err), http.StatusNotFound)
			return
		}

		response := map[string]interface{}{
			"success": true,
			"health":  health,
		}
		h.writeJSONResponse(w, response, http.StatusOK)
	} else {
		// Return health for all models
		models := h.modelManager.ListModels()
		healthMap := make(map[string]interface{})

		for _, model := range models {
			health, err := h.modelManager.GetModelHealth(model.Name)
			if err == nil {
				healthMap[model.Name] = health
			}
		}

		response := map[string]interface{}{
			"success": true,
			"health":  healthMap,
		}
		h.writeJSONResponse(w, response, http.StatusOK)
	}
}

// Helper methods

func (h *APIHandler) processContent(content []string, metadata []map[string]interface{}, chunkSize, chunkOverlap int) ([]string, []map[string]interface{}) {
	// If no chunking specified, return as-is
	if chunkSize <= 0 {
		// Ensure metadata slice matches content length
		if len(metadata) == 0 {
			metadata = make([]map[string]interface{}, len(content))
			for i := range metadata {
				metadata[i] = make(map[string]interface{})
			}
		}
		return content, metadata
	}

	// Implement text chunking
	var chunks []string
	var chunkMetadata []map[string]interface{}

	for i, text := range content {
		// Get metadata for this content item
		var baseMetadata map[string]interface{}
		if i < len(metadata) && metadata[i] != nil {
			baseMetadata = metadata[i]
		} else {
			baseMetadata = make(map[string]interface{})
		}

		// Simple word-based chunking
		words := h.splitIntoWords(text)

		for j := 0; j < len(words); j += chunkSize - chunkOverlap {
			end := j + chunkSize
			if end > len(words) {
				end = len(words)
			}

			chunk := h.joinWords(words[j:end])
			chunks = append(chunks, chunk)

			// Create metadata for chunk
			chunkMeta := make(map[string]interface{})
			for k, v := range baseMetadata {
				chunkMeta[k] = v
			}
			chunkMeta["chunk_index"] = len(chunks) - 1
			chunkMeta["source_index"] = i
			chunkMeta["chunk_start"] = j
			chunkMeta["chunk_end"] = end

			chunkMetadata = append(chunkMetadata, chunkMeta)

			// Break if we've reached the end
			if end >= len(words) {
				break
			}
		}
	}

	return chunks, chunkMetadata
}

func (h *APIHandler) splitIntoWords(text string) []string {
	// Simple whitespace-based splitting
	// In a real implementation, this would be more sophisticated
	words := []string{}
	current := ""

	for _, char := range text {
		if char == ' ' || char == '\n' || char == '\t' {
			if current != "" {
				words = append(words, current)
				current = ""
			}
		} else {
			current += string(char)
		}
	}

	if current != "" {
		words = append(words, current)
	}

	return words
}

func (h *APIHandler) joinWords(words []string) string {
	result := ""
	for i, word := range words {
		if i > 0 {
			result += " "
		}
		result += word
	}
	return result
}

func (h *APIHandler) generateVectorID(collection string, index int) string {
	timestamp := time.Now().UnixNano()
	return fmt.Sprintf("%s_%d_%d", collection, timestamp, index)
}

func (h *APIHandler) estimateTokens(content []string) int {
	// Rough estimation: ~4 characters per token
	totalChars := 0
	for _, text := range content {
		totalChars += len(text)
	}
	return totalChars / 4
}

func (h *APIHandler) writeJSONResponse(w http.ResponseWriter, data interface{}, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(data)
}

func (h *APIHandler) writeErrorResponse(w http.ResponseWriter, message string, statusCode int) {
	response := map[string]interface{}{
		"success": false,
		"error":   message,
	}
	h.writeJSONResponse(w, response, statusCode)
}

// RegisterRoutes registers all AI API routes with an HTTP mux
func (h *APIHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/embed", h.HandleEmbed)
	mux.HandleFunc("/v1/embed/batch", h.HandleBatchEmbed)
	mux.HandleFunc("/v1/models", h.HandleModelList)
	mux.HandleFunc("/v1/models/health", h.HandleModelHealth)
}
