package text

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/dshills/EmbeddixDB/core/ai"
)

func TestBM25Index_Basic(t *testing.T) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Test documents
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "The quick brown fox jumps over the lazy dog",
			Metadata: map[string]interface{}{
				"category": "animals",
			},
		},
		{
			ID:      "doc2",
			Content: "A fast brown fox leaps over a sleepy canine",
			Metadata: map[string]interface{}{
				"category": "animals",
			},
		},
		{
			ID:      "doc3",
			Content: "Machine learning algorithms for natural language processing",
			Metadata: map[string]interface{}{
				"category": "technology",
			},
		},
		{
			ID:      "doc4",
			Content: "Deep learning neural networks in computer vision applications",
			Metadata: map[string]interface{}{
				"category": "technology",
			},
		},
	}

	// Index documents
	err := idx.Index(ctx, docs)
	if err != nil {
		t.Fatalf("Failed to index documents: %v", err)
	}

	// Test search
	testCases := []struct {
		query       string
		expectedIDs []string
		minResults  int
	}{
		{
			query:       "brown fox",
			expectedIDs: []string{"doc1", "doc2"},
			minResults:  2,
		},
		{
			query:       "machine learning",
			expectedIDs: []string{"doc3"},
			minResults:  1,
		},
		{
			query:       "learning",
			expectedIDs: []string{"doc3", "doc4"},
			minResults:  2,
		},
		{
			query:       "quick lazy",
			expectedIDs: []string{"doc1"},
			minResults:  1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.query, func(t *testing.T) {
			results, err := idx.Search(ctx, tc.query, 10)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			if len(results) < tc.minResults {
				t.Errorf("Expected at least %d results, got %d", tc.minResults, len(results))
			}

			// Check if expected documents are in top results
			foundDocs := make(map[string]bool)
			for i, result := range results {
				foundDocs[result.ID] = true

				// Verify result structure
				if result.Score <= 0 {
					t.Errorf("Result %d has non-positive score: %f", i, result.Score)
				}

				if result.Content == "" {
					t.Errorf("Result %d has empty content", i)
				}

				if result.Explanation == nil || len(result.Explanation.MatchedTerms) == 0 {
					t.Errorf("Result %d has no matched terms", i)
				}
			}

			// Verify expected documents were found
			for _, expectedID := range tc.expectedIDs {
				if !foundDocs[expectedID] {
					t.Errorf("Expected document %s not found in results", expectedID)
				}
			}
		})
	}
}

func TestBM25Index_Scoring(t *testing.T) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Documents with varying term frequencies
	docs := []ai.Document{
		{
			ID:      "doc1",
			Content: "cat cat cat dog",
		},
		{
			ID:      "doc2",
			Content: "cat dog dog dog",
		},
		{
			ID:      "doc3",
			Content: "cat dog bird fish",
		},
	}

	err := idx.Index(ctx, docs)
	if err != nil {
		t.Fatalf("Failed to index documents: %v", err)
	}

	// Search for "cat" - doc1 should score highest
	results, err := idx.Search(ctx, "cat", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	if results[0].ID != "doc1" {
		t.Errorf("Expected doc1 to rank first for 'cat', got %s", results[0].ID)
	}

	// Search for "dog" - doc2 should score highest
	results, err = idx.Search(ctx, "dog", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if results[0].ID != "doc2" {
		t.Errorf("Expected doc2 to rank first for 'dog', got %s", results[0].ID)
	}
}

func TestBM25Index_Delete(t *testing.T) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Index documents
	docs := []ai.Document{
		{ID: "doc1", Content: "apple banana cherry"},
		{ID: "doc2", Content: "banana cherry date"},
		{ID: "doc3", Content: "cherry date elderberry"},
	}

	err := idx.Index(ctx, docs)
	if err != nil {
		t.Fatalf("Failed to index documents: %v", err)
	}

	// Delete doc2
	err = idx.Delete(ctx, "doc2")
	if err != nil {
		t.Fatalf("Failed to delete document: %v", err)
	}

	// Search for "banana" - should only find doc1
	results, err := idx.Search(ctx, "banana", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result after deletion, got %d", len(results))
	}

	if len(results) > 0 && results[0].ID != "doc1" {
		t.Errorf("Expected doc1, got %s", results[0].ID)
	}

	// Verify stats
	stats := idx.GetStats()
	if stats.DocumentCount != 2 {
		t.Errorf("Expected document count 2 after deletion, got %d", stats.DocumentCount)
	}
}

func TestBM25Index_EmptyQuery(t *testing.T) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Index a document
	docs := []ai.Document{
		{ID: "doc1", Content: "test content"},
	}
	idx.Index(ctx, docs)

	// Search with empty query
	results, err := idx.Search(ctx, "", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results for empty query, got %d", len(results))
	}
}

func TestBM25Index_Parameters(t *testing.T) {
	idx := NewBM25Index()

	// Test parameter setting
	idx.SetParameters(2.0, 0.5)

	if idx.k1 != 2.0 {
		t.Errorf("Expected k1=2.0, got %f", idx.k1)
	}

	if idx.b != 0.5 {
		t.Errorf("Expected b=0.5, got %f", idx.b)
	}
}

func TestSimpleTokenizer(t *testing.T) {
	tokenizer := NewSimpleTokenizer()

	testCases := []struct {
		input    string
		expected []string
	}{
		{
			input:    "Hello, world!",
			expected: []string{"Hello", "world"},
		},
		{
			input:    "test@email.com",
			expected: []string{"test", "email", "com"},
		},
		{
			input:    "multi-word-phrase",
			expected: []string{"multi", "word", "phrase"},
		},
		{
			input:    "Numbers123and456text",
			expected: []string{"Numbers123and456text"},
		},
		{
			input:    "   spaces   between   words   ",
			expected: []string{"spaces", "between", "words"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			tokens := tokenizer.Tokenize(tc.input)

			if len(tokens) != len(tc.expected) {
				t.Errorf("Expected %d tokens, got %d: %v", len(tc.expected), len(tokens), tokens)
				return
			}

			for i, token := range tokens {
				if token != tc.expected[i] {
					t.Errorf("Token %d: expected '%s', got '%s'", i, tc.expected[i], token)
				}
			}
		})
	}
}

func TestBM25Index_LargeDocument(t *testing.T) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Create a large document
	words := []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"}
	var content strings.Builder
	for i := 0; i < 1000; i++ {
		content.WriteString(words[i%len(words)])
		content.WriteString(" ")
	}

	docs := []ai.Document{
		{
			ID:      "large_doc",
			Content: content.String(),
		},
	}

	err := idx.Index(ctx, docs)
	if err != nil {
		t.Fatalf("Failed to index large document: %v", err)
	}

	// Search should still work
	results, err := idx.Search(ctx, "fox jumps", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	stats := idx.GetStats()
	if stats.AverageDocLen == 0 {
		t.Error("Average document length should not be 0")
	}
}

func TestBM25Index_ConcurrentAccess(t *testing.T) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Index initial documents
	docs := []ai.Document{
		{ID: "doc1", Content: "concurrent test document one"},
		{ID: "doc2", Content: "concurrent test document two"},
	}
	idx.Index(ctx, docs)

	// Run concurrent operations
	done := make(chan bool, 3)

	// Concurrent searches
	go func() {
		for i := 0; i < 100; i++ {
			_, err := idx.Search(ctx, "test", 10)
			if err != nil {
				t.Errorf("Search failed: %v", err)
			}
		}
		done <- true
	}()

	// Concurrent indexing
	go func() {
		for i := 0; i < 50; i++ {
			doc := ai.Document{
				ID:      fmt.Sprintf("doc_%d", i+100),
				Content: fmt.Sprintf("document number %d", i),
			}
			err := idx.Index(ctx, []ai.Document{doc})
			if err != nil {
				t.Errorf("Index failed: %v", err)
			}
		}
		done <- true
	}()

	// Concurrent stats reading
	go func() {
		for i := 0; i < 100; i++ {
			stats := idx.GetStats()
			if stats.DocumentCount < 2 {
				t.Errorf("Unexpected document count: %d", stats.DocumentCount)
			}
		}
		done <- true
	}()

	// Wait for all goroutines
	for i := 0; i < 3; i++ {
		<-done
	}
}

func BenchmarkBM25Index_Index(b *testing.B) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Prepare documents
	docs := make([]ai.Document, 100)
	for i := 0; i < 100; i++ {
		docs[i] = ai.Document{
			ID:      fmt.Sprintf("doc_%d", i),
			Content: fmt.Sprintf("This is document number %d with some random content about various topics", i),
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		idx = NewBM25Index() // Reset index
		err := idx.Index(ctx, docs)
		if err != nil {
			b.Fatalf("Index failed: %v", err)
		}
	}
}

func BenchmarkBM25Index_Search(b *testing.B) {
	ctx := context.Background()
	idx := NewBM25Index()

	// Index documents
	docs := make([]ai.Document, 1000)
	for i := 0; i < 1000; i++ {
		docs[i] = ai.Document{
			ID:      fmt.Sprintf("doc_%d", i),
			Content: fmt.Sprintf("Document %d contains information about machine learning, artificial intelligence, and data science", i),
		}
	}
	idx.Index(ctx, docs)

	queries := []string{"machine learning", "artificial intelligence", "data science", "document information"}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		_, err := idx.Search(ctx, query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
