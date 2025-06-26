package core

import (
	"time"
)

// Vector represents a high-dimensional vector with metadata
type Vector struct {
	ID       string            `json:"id"`
	Values   []float32         `json:"values"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Collection represents a named grouping of vectors with configuration
type Collection struct {
	Name      string                 `json:"name"`
	Dimension int                    `json:"dimension"`
	IndexType string                 `json:"index_type"` // "flat", "hnsw"
	Distance  string                 `json:"distance"`   // "cosine", "l2", "dot"
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// SearchRequest represents a vector search query
type SearchRequest struct {
	Query          []float32         `json:"query"`
	TopK           int               `json:"top_k"`
	Filter         map[string]string `json:"filter,omitempty"`
	IncludeVectors bool              `json:"include_vectors,omitempty"`
}

// SearchResult represents a single search result
type SearchResult struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"` // Distance score - lower values indicate higher similarity, except for dot product which uses negative values
	Vector   *Vector           `json:"vector,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// RangeSearchRequest represents a range query to find all vectors within a distance threshold
type RangeSearchRequest struct {
	Query          []float32         `json:"query"`
	Radius         float32           `json:"radius"`           // Maximum distance threshold
	Filter         map[string]string `json:"filter,omitempty"` // Optional metadata filters
	IncludeVectors bool              `json:"include_vectors,omitempty"`
	Limit          int               `json:"limit,omitempty"` // Optional limit on results (0 = no limit)
}

// RangeSearchResult contains vectors found within the specified range
type RangeSearchResult struct {
	Results []SearchResult `json:"results"`
	Count   int            `json:"count"`   // Total number of results found
	Limited bool           `json:"limited"` // True if results were limited
}
