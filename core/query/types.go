package query

// SearchRequest represents a vector search request
// This is a copy of core.SearchRequest to avoid import cycles
type SearchRequest struct {
	Query          []float32
	TopK           int
	Filter         map[string]string
	IncludeVectors bool
}

// SearchResult represents a search result
// This is a copy of core.SearchResult to avoid import cycles
type SearchResult struct {
	ID       string
	Score    float32
	Metadata map[string]string
	Vector   []float32
}
