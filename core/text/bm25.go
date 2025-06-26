package text

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// BM25Index implements the BM25 ranking algorithm for text search
type BM25Index struct {
	mu           sync.RWMutex
	documents    map[string]*Document      // docID -> document
	termFreqs    map[string]map[string]int // term -> docID -> frequency
	docLengths   map[string]int            // docID -> document length
	avgDocLength float64
	docCount     int
	lastUpdate   time.Time

	// BM25 parameters
	k1 float64 // term frequency saturation parameter (default: 1.2)
	b  float64 // length normalization parameter (default: 0.75)

	// Text processing
	tokenizer Tokenizer
	stopWords map[string]bool
}

// Document represents a document in the BM25 index
type Document struct {
	ID       string
	Content  string
	Metadata map[string]interface{}
	Terms    []string // preprocessed terms
	Length   int      // number of terms
}

// NewBM25Index creates a new BM25 text search index
func NewBM25Index() *BM25Index {
	return &BM25Index{
		documents:  make(map[string]*Document),
		termFreqs:  make(map[string]map[string]int),
		docLengths: make(map[string]int),
		k1:         1.2,
		b:          0.75,
		tokenizer:  NewSimpleTokenizer(),
		stopWords:  defaultStopWords(),
	}
}

// Index adds documents to the BM25 index
func (idx *BM25Index) Index(ctx context.Context, docs []ai.Document) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, doc := range docs {
		// Process document
		terms := idx.processDocument(doc.Content)

		// Create internal document
		internalDoc := &Document{
			ID:       doc.ID,
			Content:  doc.Content,
			Metadata: doc.Metadata,
			Terms:    terms,
			Length:   len(terms),
		}

		// Update document store
		idx.documents[doc.ID] = internalDoc
		idx.docLengths[doc.ID] = len(terms)

		// Update term frequencies
		termCounts := make(map[string]int)
		for _, term := range terms {
			termCounts[term]++
		}

		for term, count := range termCounts {
			if idx.termFreqs[term] == nil {
				idx.termFreqs[term] = make(map[string]int)
			}
			idx.termFreqs[term][doc.ID] = count
		}
	}

	// Update statistics
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()
	idx.lastUpdate = time.Now()

	return nil
}

// Search performs BM25 text search
func (idx *BM25Index) Search(ctx context.Context, query string, limit int) ([]ai.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Process query
	queryTerms := idx.processDocument(query)
	if len(queryTerms) == 0 {
		return []ai.SearchResult{}, nil
	}

	// Calculate BM25 scores for all documents
	scores := make(map[string]float64)

	for _, term := range queryTerms {
		// Get documents containing this term
		docFreqs, exists := idx.termFreqs[term]
		if !exists {
			continue
		}

		// Calculate IDF (Inverse Document Frequency)
		df := float64(len(docFreqs)) // document frequency
		idf := math.Log((float64(idx.docCount)-df+0.5)/(df+0.5) + 1)

		// Calculate score contribution for each document
		for docID, termFreq := range docFreqs {
			docLength := float64(idx.docLengths[docID])

			// BM25 formula
			numerator := float64(termFreq) * (idx.k1 + 1)
			denominator := float64(termFreq) + idx.k1*(1-idx.b+idx.b*(docLength/idx.avgDocLength))

			scores[docID] += idf * (numerator / denominator)
		}
	}

	// Convert to search results and sort by score
	results := make([]ai.SearchResult, 0, len(scores))
	for docID, score := range scores {
		doc := idx.documents[docID]
		results = append(results, ai.SearchResult{
			ID:       docID,
			Score:    score,
			Content:  doc.Content,
			Metadata: doc.Metadata,
			Explanation: &ai.SearchExplanation{
				TextScore:    score,
				MatchedTerms: idx.findMatchedTerms(doc.Terms, queryTerms),
			},
		})
	}

	// Sort by score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// Delete removes a document from the index
func (idx *BM25Index) Delete(ctx context.Context, docID string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	doc, exists := idx.documents[docID]
	if !exists {
		return nil // Document not found, nothing to delete
	}

	// Remove from term frequencies
	for _, term := range doc.Terms {
		if termDocs, exists := idx.termFreqs[term]; exists {
			delete(termDocs, docID)
			if len(termDocs) == 0 {
				delete(idx.termFreqs, term)
			}
		}
	}

	// Remove from document store
	delete(idx.documents, docID)
	delete(idx.docLengths, docID)

	// Update statistics
	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()
	idx.lastUpdate = time.Now()

	return nil
}

// GetStats returns index statistics
func (idx *BM25Index) GetStats() ai.TextIndexStats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	termCount := int64(len(idx.termFreqs))

	// Calculate index size (rough estimate)
	indexSize := int64(0)
	for term, docs := range idx.termFreqs {
		indexSize += int64(len(term) + len(docs)*12) // term + docID->freq pairs
	}

	return ai.TextIndexStats{
		DocumentCount: int64(idx.docCount),
		TermCount:     termCount,
		IndexSize:     indexSize,
		AverageDocLen: idx.avgDocLength,
		LastUpdate:    idx.lastUpdate,
	}
}

// SetParameters sets BM25 parameters
func (idx *BM25Index) SetParameters(k1, b float64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.k1 = k1
	idx.b = b
}

// processDocument tokenizes and processes a document
func (idx *BM25Index) processDocument(content string) []string {
	// Tokenize
	tokens := idx.tokenizer.Tokenize(content)

	// Filter stop words and normalize
	processed := make([]string, 0, len(tokens))
	for _, token := range tokens {
		normalized := strings.ToLower(token)
		if !idx.stopWords[normalized] && len(normalized) > 1 {
			processed = append(processed, normalized)
		}
	}

	return processed
}

// updateAverageDocLength updates the average document length
func (idx *BM25Index) updateAverageDocLength() {
	if idx.docCount == 0 {
		idx.avgDocLength = 0
		return
	}

	totalLength := 0
	for _, length := range idx.docLengths {
		totalLength += length
	}

	idx.avgDocLength = float64(totalLength) / float64(idx.docCount)
}

// findMatchedTerms finds which query terms matched in a document
func (idx *BM25Index) findMatchedTerms(docTerms, queryTerms []string) []string {
	// Create a set of document terms for quick lookup
	docTermSet := make(map[string]bool)
	for _, term := range docTerms {
		docTermSet[term] = true
	}

	// Find matches
	matched := make([]string, 0)
	seen := make(map[string]bool)

	for _, qTerm := range queryTerms {
		if docTermSet[qTerm] && !seen[qTerm] {
			matched = append(matched, qTerm)
			seen[qTerm] = true
		}
	}

	return matched
}

// Tokenizer interface for text tokenization
type Tokenizer interface {
	Tokenize(text string) []string
}

// SimpleTokenizer implements basic tokenization
type SimpleTokenizer struct{}

// NewSimpleTokenizer creates a new simple tokenizer
func NewSimpleTokenizer() *SimpleTokenizer {
	return &SimpleTokenizer{}
}

// Tokenize splits text into tokens
func (t *SimpleTokenizer) Tokenize(text string) []string {
	var tokens []string
	var currentToken strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentToken.WriteRune(r)
		} else {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
		}
	}

	// Don't forget the last token
	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}

	return tokens
}

// defaultStopWords returns a set of common English stop words
func defaultStopWords() map[string]bool {
	words := []string{
		"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
		"has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
		"to", "was", "will", "with", "the", "this", "but", "they", "have",
		"had", "what", "when", "where", "who", "which", "why", "how",
		"all", "would", "there", "their", "been", "if", "more", "can",
		"her", "him", "could", "may", "about", "after", "before", "just",
		"should", "than", "then", "these", "those", "through", "under",
		"up", "down", "out", "over", "again", "further", "once",
	}

	stopWords := make(map[string]bool)
	for _, word := range words {
		stopWords[word] = true
	}

	return stopWords
}
