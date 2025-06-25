package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/dshills/EmbeddixDB/core"
)

// Health check response
type HealthResponse struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Version   string    `json:"version"`
}

// handleHealth returns server health status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	response := HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now(),
		Version:   "1.0.0",
	}
	s.respondWithJSON(w, http.StatusOK, response)
}

// Collection request/response types
type CreateCollectionRequest struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	IndexType string `json:"index_type"`
	Distance  string `json:"distance"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type CollectionResponse struct {
	Name      string                 `json:"name"`
	Dimension int                    `json:"dimension"`
	IndexType string                 `json:"index_type"`
	Distance  string                 `json:"distance"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// handleListCollections returns all collections
func (s *Server) handleListCollections(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	
	collections, err := s.vectorStore.ListCollections(ctx)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := make([]CollectionResponse, len(collections))
	for i, col := range collections {
		response[i] = CollectionResponse{
			Name:      col.Name,
			Dimension: col.Dimension,
			IndexType: col.IndexType,
			Distance:  string(col.Distance),
			Metadata:  col.Metadata,
			CreatedAt: col.CreatedAt,
			UpdatedAt: col.UpdatedAt,
		}
	}

	s.respondWithJSON(w, http.StatusOK, response)
}

// handleCreateCollection creates a new collection
func (s *Server) handleCreateCollection(w http.ResponseWriter, r *http.Request) {
	var req CreateCollectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Validate request
	if req.Name == "" {
		s.respondWithError(w, http.StatusBadRequest, "Collection name is required")
		return
	}
	if req.Dimension <= 0 {
		s.respondWithError(w, http.StatusBadRequest, "Dimension must be positive")
		return
	}

	// Set defaults
	if req.IndexType == "" {
		req.IndexType = "flat"
	}
	if req.Distance == "" {
		req.Distance = "cosine"
	}

	collection := core.Collection{
		Name:      req.Name,
		Dimension: req.Dimension,
		IndexType: req.IndexType,
		Distance:  req.Distance,
		Metadata:  req.Metadata,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	ctx := context.Background()
	err := s.vectorStore.CreateCollection(ctx, collection)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := CollectionResponse{
		Name:      collection.Name,
		Dimension: collection.Dimension,
		IndexType: collection.IndexType,
		Distance:  string(collection.Distance),
		Metadata:  collection.Metadata,
		CreatedAt: collection.CreatedAt,
		UpdatedAt: collection.UpdatedAt,
	}

	s.respondWithJSON(w, http.StatusCreated, response)
}

// handleGetCollection returns a specific collection
func (s *Server) handleGetCollection(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	ctx := context.Background()
	collections, err := s.vectorStore.ListCollections(ctx)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	for _, col := range collections {
		if col.Name == collectionName {
			response := CollectionResponse{
				Name:      col.Name,
				Dimension: col.Dimension,
				IndexType: col.IndexType,
				Distance:  string(col.Distance),
				Metadata:  col.Metadata,
				CreatedAt: col.CreatedAt,
				UpdatedAt: col.UpdatedAt,
			}
			s.respondWithJSON(w, http.StatusOK, response)
			return
		}
	}

	s.respondWithError(w, http.StatusNotFound, "Collection not found")
}

// handleDeleteCollection deletes a collection
func (s *Server) handleDeleteCollection(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	ctx := context.Background()
	err := s.vectorStore.DeleteCollection(ctx, collectionName)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, map[string]string{"message": "Collection deleted successfully"})
}

// Vector request/response types
type AddVectorRequest struct {
	ID       string                 `json:"id"`
	Values   []float32              `json:"values"`
	Metadata map[string]string      `json:"metadata,omitempty"`
}

type VectorResponse struct {
	ID       string                 `json:"id"`
	Values   []float32              `json:"values"`
	Metadata map[string]string      `json:"metadata,omitempty"`
}

// handleAddVector adds a single vector
func (s *Server) handleAddVector(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	var req AddVectorRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.ID == "" {
		s.respondWithError(w, http.StatusBadRequest, "Vector ID is required")
		return
	}
	if len(req.Values) == 0 {
		s.respondWithError(w, http.StatusBadRequest, "Vector values are required")
		return
	}

	vector := core.Vector{
		ID:       req.ID,
		Values:   req.Values,
		Metadata: req.Metadata,
	}

	ctx := context.Background()
	err := s.vectorStore.AddVector(ctx, collectionName, vector)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := VectorResponse{
		ID:       vector.ID,
		Values:   vector.Values,
		Metadata: vector.Metadata,
	}

	s.respondWithJSON(w, http.StatusCreated, response)
}

// handleAddVectorsBatch adds multiple vectors in batch
func (s *Server) handleAddVectorsBatch(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	var req []AddVectorRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if len(req) == 0 {
		s.respondWithError(w, http.StatusBadRequest, "At least one vector is required")
		return
	}

	vectors := make([]core.Vector, len(req))
	for i, v := range req {
		if v.ID == "" {
			s.respondWithError(w, http.StatusBadRequest, "All vectors must have an ID")
			return
		}
		if len(v.Values) == 0 {
			s.respondWithError(w, http.StatusBadRequest, "All vectors must have values")
			return
		}

		vectors[i] = core.Vector{
			ID:       v.ID,
			Values:   v.Values,
			Metadata: v.Metadata,
		}
	}

	ctx := context.Background()
	err := s.vectorStore.AddVectorsBatch(ctx, collectionName, vectors)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusCreated, map[string]interface{}{
		"message": "Vectors added successfully",
		"count":   len(vectors),
	})
}

// handleGetVector retrieves a specific vector
func (s *Server) handleGetVector(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]
	vectorID := vars["id"]

	ctx := context.Background()
	vector, err := s.vectorStore.GetVector(ctx, collectionName, vectorID)
	if err != nil {
		s.respondWithError(w, http.StatusNotFound, err.Error())
		return
	}

	response := VectorResponse{
		ID:       vector.ID,
		Values:   vector.Values,
		Metadata: vector.Metadata,
	}

	s.respondWithJSON(w, http.StatusOK, response)
}

// handleUpdateVector updates a vector
func (s *Server) handleUpdateVector(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]
	vectorID := vars["id"]

	var req AddVectorRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Override ID from URL
	req.ID = vectorID

	vector := core.Vector{
		ID:       req.ID,
		Values:   req.Values,
		Metadata: req.Metadata,
	}

	ctx := context.Background()
	err := s.vectorStore.UpdateVector(ctx, collectionName, vector)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := VectorResponse{
		ID:       vector.ID,
		Values:   vector.Values,
		Metadata: vector.Metadata,
	}

	s.respondWithJSON(w, http.StatusOK, response)
}

// handleDeleteVector deletes a vector
func (s *Server) handleDeleteVector(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]
	vectorID := vars["id"]

	ctx := context.Background()
	err := s.vectorStore.DeleteVector(ctx, collectionName, vectorID)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, map[string]string{"message": "Vector deleted successfully"})
}

// Search request/response types
type SearchRequest struct {
	Query          []float32         `json:"query"`
	TopK           int               `json:"top_k"`
	Filter         map[string]string `json:"filter,omitempty"`
	IncludeVectors bool              `json:"include_vectors"`
}

type SearchResult struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"`
	Vector   []float32         `json:"vector,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// handleSearch performs vector search
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if len(req.Query) == 0 {
		s.respondWithError(w, http.StatusBadRequest, "Query vector is required")
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10 // Default
	}

	searchReq := core.SearchRequest{
		Query:          req.Query,
		TopK:           req.TopK,
		Filter:         req.Filter,
		IncludeVectors: req.IncludeVectors,
	}

	ctx := context.Background()
	results, err := s.vectorStore.Search(ctx, collectionName, searchReq)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := make([]SearchResult, len(results))
	for i, res := range results {
		response[i] = SearchResult{
			ID:       res.ID,
			Score:    res.Score,
			Metadata: res.Metadata,
		}
		if req.IncludeVectors && res.Vector != nil {
			response[i].Vector = res.Vector.Values
		}
	}

	s.respondWithJSON(w, http.StatusOK, response)
}

// handleBatchSearch performs multiple searches
func (s *Server) handleBatchSearch(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	var requests []SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&requests); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if len(requests) == 0 {
		s.respondWithError(w, http.StatusBadRequest, "At least one search request is required")
		return
	}

	ctx := context.Background()
	allResults := make([][]SearchResult, len(requests))

	for i, req := range requests {
		if len(req.Query) == 0 {
			s.respondWithError(w, http.StatusBadRequest, "All queries must have a vector")
			return
		}
		if req.TopK <= 0 {
			req.TopK = 10
		}

		searchReq := core.SearchRequest{
			Query:          req.Query,
			TopK:           req.TopK,
			Filter:         req.Filter,
			IncludeVectors: req.IncludeVectors,
		}

		results, err := s.vectorStore.Search(ctx, collectionName, searchReq)
		if err != nil {
			s.respondWithError(w, http.StatusInternalServerError, err.Error())
			return
		}

		searchResults := make([]SearchResult, len(results))
		for j, res := range results {
			searchResults[j] = SearchResult{
				ID:       res.ID,
				Score:    res.Score,
				Metadata: res.Metadata,
			}
			if req.IncludeVectors && res.Vector != nil {
				searchResults[j].Vector = res.Vector.Values
			}
		}
		allResults[i] = searchResults
	}

	s.respondWithJSON(w, http.StatusOK, allResults)
}

// Stats response types
type StatsResponse struct {
	TotalCollections int                       `json:"total_collections"`
	TotalVectors     int                       `json:"total_vectors"`
	Collections      map[string]CollectionStats `json:"collections"`
}

type CollectionStats struct {
	VectorCount int    `json:"vector_count"`
	IndexType   string `json:"index_type"`
	Dimension   int    `json:"dimension"`
}

// handleStats returns overall statistics
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	
	collections, err := s.vectorStore.ListCollections(ctx)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	stats := StatsResponse{
		TotalCollections: len(collections),
		TotalVectors:     0,
		Collections:      make(map[string]CollectionStats),
	}

	for _, col := range collections {
		count, _ := s.vectorStore.GetCollectionSize(ctx, col.Name)
		stats.TotalVectors += count
		stats.Collections[col.Name] = CollectionStats{
			VectorCount: count,
			IndexType:   col.IndexType,
			Dimension:   col.Dimension,
		}
	}

	s.respondWithJSON(w, http.StatusOK, stats)
}

// handleCollectionStats returns statistics for a specific collection
func (s *Server) handleCollectionStats(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	ctx := context.Background()
	
	collections, err := s.vectorStore.ListCollections(ctx)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	for _, col := range collections {
		if col.Name == collectionName {
			count, _ := s.vectorStore.GetCollectionSize(ctx, collectionName)
			stats := CollectionStats{
				VectorCount: count,
				IndexType:   col.IndexType,
				Dimension:   col.Dimension,
			}
			s.respondWithJSON(w, http.StatusOK, stats)
			return
		}
	}

	s.respondWithError(w, http.StatusNotFound, "Collection not found")
}

// Range search request/response types
type RangeSearchRequest struct {
	Query          []float32         `json:"query"`
	Radius         float32           `json:"radius"`
	Filter         map[string]string `json:"filter,omitempty"`
	IncludeVectors bool              `json:"include_vectors"`
	Limit          int               `json:"limit,omitempty"`
}

type RangeSearchResponse struct {
	Results []SearchResult `json:"results"`
	Count   int            `json:"count"`
	Limited bool           `json:"limited"`
}

// handleRangeSearch performs range query to find vectors within a distance threshold
func (s *Server) handleRangeSearch(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	collectionName := vars["collection"]

	var req RangeSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Validate request
	if len(req.Query) == 0 {
		s.respondWithError(w, http.StatusBadRequest, "Query vector is required")
		return
	}
	if req.Radius < 0 {
		s.respondWithError(w, http.StatusBadRequest, "Radius must be non-negative")
		return
	}

	// Convert to core request
	rangeReq := core.RangeSearchRequest{
		Query:          req.Query,
		Radius:         req.Radius,
		Filter:         req.Filter,
		IncludeVectors: req.IncludeVectors,
		Limit:          req.Limit,
	}

	ctx := context.Background()
	result, err := s.vectorStore.RangeSearch(ctx, collectionName, rangeReq)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Convert results
	response := RangeSearchResponse{
		Results: make([]SearchResult, len(result.Results)),
		Count:   result.Count,
		Limited: result.Limited,
	}

	for i, res := range result.Results {
		response.Results[i] = SearchResult{
			ID:       res.ID,
			Score:    res.Score,
			Metadata: res.Metadata,
		}
		if req.IncludeVectors && res.Vector != nil {
			response.Results[i].Vector = res.Vector.Values
		}
	}

	s.respondWithJSON(w, http.StatusOK, response)
}