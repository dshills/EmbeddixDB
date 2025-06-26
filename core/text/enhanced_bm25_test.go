package text

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/dshills/EmbeddixDB/core/ai"
)

func TestEnhancedBM25Index_PhraseSearch(t *testing.T) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents with phrases
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "The quick brown fox jumps over the lazy dog",
		},
		{
			ID:      "doc2",
			Content: "Machine learning is a subset of artificial intelligence",
		},
		{
			ID:      "doc3",
			Content: "Deep learning neural networks are powerful tools",
		},
		{
			ID:      "doc4",
			Content: "The brown fox is quick and clever",
		},
	}

	// Use the enhanced indexing method
	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	// Test phrase searches
	testCases := []struct {
		query       string
		expectedIDs []string
		description string
	}{
		{
			query:       `"quick brown fox"`,
			expectedIDs: []string{"doc1"},
			description: "Exact phrase match",
		},
		{
			query:       `"machine learning"`,
			expectedIDs: []string{"doc2"},
			description: "Two-word phrase",
		},
		{
			query:       `"brown fox"`,
			expectedIDs: []string{"doc1", "doc4"},
			description: "Phrase in multiple docs",
		},
		{
			query:       `"deep learning neural"`,
			expectedIDs: []string{"doc3"},
			description: "Three-word phrase",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			req := EnhancedSearchRequest{
				Query: tc.query,
				Limit: 10,
			}

			results, err := idx.SearchWithOptions(ctx, req)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Check if expected documents are found
			foundDocs := make(map[string]bool)
			for _, result := range results {
				foundDocs[result.ID] = true
			}

			for _, expectedID := range tc.expectedIDs {
				if !foundDocs[expectedID] {
					t.Errorf("Expected document %s not found for query '%s'", expectedID, tc.query)
				}
			}

			// Phrase matches should have high scores
			if len(results) > 0 && results[0].Score < 5.0 {
				t.Errorf("Phrase match score too low: %f", results[0].Score)
			}
		})
	}
}

func TestEnhancedBM25Index_QueryExpansion(t *testing.T) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "I need to find some information quickly",
		},
		{
			ID:      "doc2",
			Content: "Searching for data in the database",
		},
		{
			ID:      "doc3",
			Content: "Query the system for results",
		},
		{
			ID:      "doc4",
			Content: "Looking up records in storage",
		},
	}

	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	// Search with query expansion enabled
	req := EnhancedSearchRequest{
		Query:       "search",
		Limit:       10,
		ExpandQuery: true,
	}

	results, err := idx.SearchWithOptions(ctx, req)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// With query expansion, should find documents with synonyms
	// "search" -> "find", "query", "lookup"
	if len(results) < 2 {
		t.Errorf("Expected at least 2 results with query expansion, got %d", len(results))
	}

	// Check if synonym matches are found
	foundDocs := make(map[string]bool)
	for _, result := range results {
		foundDocs[result.ID] = true
	}

	// doc1 has "find", doc3 has "query" - both synonyms of "search"
	if !foundDocs["doc1"] && !foundDocs["doc3"] {
		t.Error("Query expansion did not find synonym matches")
	}
}

func TestEnhancedBM25Index_FieldSearch(t *testing.T) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents with fields
	fieldDocs := []FieldDocument{
		{
			ID: "doc1",
			Fields: map[string]string{
				"title":       "Introduction to Machine Learning",
				"content":     "Machine learning is a powerful technology",
				"description": "A comprehensive guide to ML basics",
			},
		},
		{
			ID: "doc2",
			Fields: map[string]string{
				"title":       "Deep Learning Fundamentals",
				"content":     "Neural networks and deep architectures",
				"description": "Understanding deep learning concepts",
			},
		},
		{
			ID: "doc3",
			Fields: map[string]string{
				"title":       "Natural Language Processing",
				"content":     "Processing text with machine learning",
				"description": "NLP techniques and applications",
			},
		},
	}

	err := idx.IndexWithFields(ctx, fieldDocs)
	if err != nil {
		t.Fatalf("Failed to index field documents: %v", err)
	}

	// Search in specific fields
	req := EnhancedSearchRequest{
		Query:        "machine learning",
		Limit:        10,
		SearchFields: []string{"title", "content"},
		FieldWeights: map[string]float64{
			"title":   2.0,
			"content": 1.0,
		},
	}

	results, err := idx.SearchWithOptions(ctx, req)
	if err != nil {
		t.Fatalf("Field search failed: %v", err)
	}

	// doc1 should rank highest (has "machine learning" in title)
	if len(results) > 0 && results[0].ID != "doc1" {
		t.Errorf("Expected doc1 to rank first with title boost, got %s", results[0].ID)
	}
}

func TestEnhancedBM25Index_MustTerms(t *testing.T) {
	idx := NewEnhancedBM25Index()

	// Index documents
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "Python programming for data science",
		},
		{
			ID:      "doc2",
			Content: "Java programming for enterprise applications",
		},
		{
			ID:      "doc3",
			Content: "Python and Java are popular languages",
		},
		{
			ID:      "doc4",
			Content: "Data science with R programming",
		},
	}

	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	// Search with must terms (+python)
	req := EnhancedSearchRequest{
		Query: "+python programming",
		Limit: 10,
	}

	parsedQuery := idx.parseQuery(req.Query)

	// Check parsed query
	if len(parsedQuery.MustTerms) != 1 || parsedQuery.MustTerms[0] != "python" {
		t.Errorf("Failed to parse must term: %v", parsedQuery.MustTerms)
	}

	// In a full implementation, must terms would filter results
	// Here we just verify parsing works correctly
}

func TestPorterStemmer(t *testing.T) {
	stemmer := NewPorterStemmer()

	testCases := []struct {
		input    string
		expected string
	}{
		{"running", "runn"},
		{"runs", "run"},
		{"runner", "runner"}, // Our simple implementation doesn't handle -er
		{"quickly", "quick"},
		{"jumped", "jump"},
		{"jumping", "jump"},
		{"goes", "goe"},
		{"went", "went"}, // Irregular verb not handled
	}

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			result := stemmer.Stem(tc.input)
			if result != tc.expected {
				t.Errorf("Stem(%s) = %s, expected %s", tc.input, result, tc.expected)
			}
		})
	}
}

func TestEditDistance(t *testing.T) {
	testCases := []struct {
		s1       string
		s2       string
		expected int
	}{
		{"", "", 0},
		{"a", "", 1},
		{"", "a", 1},
		{"cat", "cat", 0},
		{"cat", "cut", 1},
		{"cat", "dog", 3},
		{"sitting", "kitten", 3},
		{"saturday", "sunday", 3},
		{"book", "back", 2},
	}

	for _, tc := range testCases {
		t.Run(tc.s1+"_"+tc.s2, func(t *testing.T) {
			result := editDistance(tc.s1, tc.s2)
			if result != tc.expected {
				t.Errorf("editDistance(%s, %s) = %d, expected %d", tc.s1, tc.s2, result, tc.expected)
			}
		})
	}
}

func TestEnhancedBM25Index_FuzzySearch(t *testing.T) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "color colours coloring colored",
		},
		{
			ID:      "doc2",
			Content: "organize organisation organizing",
		},
		{
			ID:      "doc3",
			Content: "analyze analysis analytical",
		},
	}

	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	// Enable fuzzy search
	req := EnhancedSearchRequest{
		Query:         "colour", // British spelling
		Limit:         10,
		EnableFuzzy:   true,
		FuzzyDistance: 2,
	}

	results, err := idx.SearchWithOptions(ctx, req)
	if err != nil {
		t.Fatalf("Fuzzy search failed: %v", err)
	}

	// Should find doc1 with fuzzy matching
	foundDoc1 := false
	for _, result := range results {
		if result.ID == "doc1" {
			foundDoc1 = true
			break
		}
	}

	if !foundDoc1 {
		t.Error("Fuzzy search did not find similar terms")
	}
}

func TestEnhancedBM25Index_MinScore(t *testing.T) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "The quick brown fox jumps over the lazy dog",
		},
		{
			ID:      "doc2",
			Content: "A completely unrelated document about something else",
		},
		{
			ID:      "doc3",
			Content: "The fox is quick and brown",
		},
	}

	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	// Search with minimum score threshold
	req := EnhancedSearchRequest{
		Query:    "quick brown fox",
		Limit:    10,
		MinScore: 1.0,
	}

	results, err := idx.SearchWithOptions(ctx, req)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// All results should have score >= 1.0
	for _, result := range results {
		if result.Score < req.MinScore {
			t.Errorf("Result %s has score %f below minimum %f", result.ID, result.Score, req.MinScore)
		}
	}

	// doc2 should be filtered out due to low score
	for _, result := range results {
		if result.ID == "doc2" {
			t.Error("Low-scoring document doc2 was not filtered out")
		}
	}
}

func TestEnhancedBM25Index_LongPhrase(t *testing.T) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents with long phrases
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "The quick brown fox jumps over the lazy dog in the morning",
		},
		{
			ID:      "doc2",
			Content: "A quick brown fox jumps high but the lazy dog sleeps",
		},
		{
			ID:      "doc3",
			Content: "The lazy dog sleeps while the quick brown fox hunts",
		},
	}

	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	// Search for a 4-word phrase
	req := EnhancedSearchRequest{
		Query: `"quick brown fox jumps"`,
		Limit: 10,
	}

	results, err := idx.SearchWithOptions(ctx, req)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should find exact matches
	if len(results) == 0 {
		t.Error("Long phrase search returned no results")
	}

	// doc1 should be found (has exact phrase)
	foundDoc1 := false
	for _, result := range results {
		if result.ID == "doc1" {
			foundDoc1 = true
			if result.Score < 10.0 {
				t.Errorf("Phrase match score too low: %f", result.Score)
			}
			break
		}
	}

	if !foundDoc1 {
		t.Error("Failed to find document with exact long phrase")
	}
}

func BenchmarkEnhancedBM25Index_PhraseSearch(b *testing.B) {
	ctx := context.Background()
	idx := NewEnhancedBM25Index()

	// Index documents
	docs := make([]ai.Document, 1000)
	for i := 0; i < 1000; i++ {
		docs[i] = ai.Document{
			ID:      fmt.Sprintf("doc_%d", i),
			Content: fmt.Sprintf("Document %d contains machine learning and artificial intelligence concepts", i),
		}
	}

	for _, doc := range docs {
		idx.indexSingleDocument(doc)
	}
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()

	req := EnhancedSearchRequest{
		Query: `"machine learning"`,
		Limit: 10,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.SearchWithOptions(ctx, req)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

func TestFieldDocument_GetAllContent(t *testing.T) {
	doc := FieldDocument{
		ID: "test",
		Fields: map[string]string{
			"title":   "Test Title",
			"content": "Test Content",
			"tags":    "tag1 tag2",
		},
	}

	allContent := doc.GetAllContent()

	// Should contain all field values
	if !strings.Contains(allContent, "Test Title") {
		t.Error("All content should contain title")
	}

	if !strings.Contains(allContent, "Test Content") {
		t.Error("All content should contain content")
	}

	if !strings.Contains(allContent, "tag1 tag2") {
		t.Error("All content should contain tags")
	}
}
