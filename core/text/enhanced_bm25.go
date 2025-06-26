package text

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// EnhancedBM25Index extends BM25 with additional features
type EnhancedBM25Index struct {
	*BM25Index

	// Additional features
	enableStemming bool
	enablePhrases  bool
	enableSynonyms bool
	enableFuzzy    bool

	// Phrase search support
	bigramIndex  map[string]map[string][]int // bigram -> docID -> positions
	trigramIndex map[string]map[string][]int // trigram -> docID -> positions

	// Term positions for phrase search
	termPositions map[string]map[string][]int // term -> docID -> positions

	// Stemmer and synonym expansion
	stemmer    Stemmer
	synonymMap map[string][]string

	// Query expansion
	queryExpander *QueryExpander

	// Field-specific indexing
	fieldIndices map[string]*BM25Index // field name -> index
	fieldBoosts  map[string]float64    // field name -> boost factor
}

// NewEnhancedBM25Index creates an enhanced BM25 index
func NewEnhancedBM25Index() *EnhancedBM25Index {
	return &EnhancedBM25Index{
		BM25Index:      NewBM25Index(),
		enableStemming: true,
		enablePhrases:  true,
		enableSynonyms: true,
		enableFuzzy:    false,
		bigramIndex:    make(map[string]map[string][]int),
		trigramIndex:   make(map[string]map[string][]int),
		termPositions:  make(map[string]map[string][]int),
		stemmer:        NewPorterStemmer(),
		synonymMap:     defaultSynonyms(),
		queryExpander:  NewQueryExpander(),
		fieldIndices:   make(map[string]*BM25Index),
		fieldBoosts:    defaultFieldBoosts(),
	}
}

// IndexWithFields indexes documents with field-specific processing
func (idx *EnhancedBM25Index) IndexWithFields(ctx context.Context, docs []FieldDocument) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, doc := range docs {
		// Index in main index
		mainDoc := ai.Document{
			ID:       doc.ID,
			Content:  doc.GetAllContent(),
			Metadata: doc.Metadata,
		}

		// Process and index main content
		idx.indexSingleDocument(mainDoc)

		// Index individual fields
		for fieldName, fieldContent := range doc.Fields {
			if fieldContent == "" {
				continue
			}

			// Get or create field index
			fieldIdx, exists := idx.fieldIndices[fieldName]
			if !exists {
				fieldIdx = NewBM25Index()
				idx.fieldIndices[fieldName] = fieldIdx
			}

			// Index in field-specific index
			fieldDoc := ai.Document{
				ID:       doc.ID,
				Content:  fieldContent,
				Metadata: doc.Metadata,
			}
			fieldIdx.Index(ctx, []ai.Document{fieldDoc})
		}
	}

	idx.lastUpdate = time.Now()
	return nil
}

// SearchWithOptions performs enhanced search with various options
func (idx *EnhancedBM25Index) SearchWithOptions(ctx context.Context, req EnhancedSearchRequest) ([]ai.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Parse query for special operators
	parsedQuery := idx.parseQuery(req.Query)

	// Expand query if enabled
	if req.ExpandQuery && idx.enableSynonyms {
		parsedQuery = idx.expandQuery(parsedQuery)
	}

	var results []ai.SearchResult

	// Handle phrase queries
	if parsedQuery.HasPhrases && idx.enablePhrases {
		results = idx.searchPhrases(parsedQuery, req.Limit)
	} else {
		// Regular BM25 search
		results, _ = idx.BM25Index.Search(ctx, parsedQuery.ProcessedQuery, req.Limit*2)
	}

	// Apply field-specific search if requested
	if len(req.SearchFields) > 0 {
		fieldResults := idx.searchFields(ctx, parsedQuery, req.SearchFields, req.Limit)
		results = idx.mergeResults(results, fieldResults, req.FieldWeights)
	}

	// Apply fuzzy matching if enabled
	if req.EnableFuzzy && idx.enableFuzzy {
		fuzzyResults := idx.fuzzySearch(parsedQuery, req.FuzzyDistance, req.Limit)
		results = idx.mergeResults(results, fuzzyResults, map[string]float64{"fuzzy": 0.8})
	}

	// Filter by minimum score
	if req.MinScore > 0 {
		filtered := make([]ai.SearchResult, 0)
		for _, result := range results {
			if result.Score >= req.MinScore {
				filtered = append(filtered, result)
			}
		}
		results = filtered
	}

	// Limit final results
	if len(results) > req.Limit {
		results = results[:req.Limit]
	}

	return results, nil
}

// indexSingleDocument indexes a single document with enhanced features
func (idx *EnhancedBM25Index) indexSingleDocument(doc ai.Document) {
	// Tokenize content
	tokens := idx.tokenizer.Tokenize(doc.Content)

	// Track positions for phrase search
	positions := make(map[string][]int)
	processedTerms := make([]string, 0, len(tokens))

	for i, token := range tokens {
		normalized := strings.ToLower(token)

		// Skip stop words for indexing but keep positions
		if idx.stopWords[normalized] {
			continue
		}

		// Apply stemming if enabled
		term := normalized
		if idx.enableStemming && len(term) > 2 {
			term = idx.stemmer.Stem(term)
		}

		processedTerms = append(processedTerms, term)

		// Track positions
		if positions[term] == nil {
			positions[term] = make([]int, 0)
		}
		positions[term] = append(positions[term], i)
	}

	// Store term positions for phrase search
	for term, pos := range positions {
		if idx.termPositions[term] == nil {
			idx.termPositions[term] = make(map[string][]int)
		}
		idx.termPositions[term][doc.ID] = pos
	}

	// Generate and index bigrams/trigrams for phrase search
	if idx.enablePhrases {
		idx.indexNgrams(doc.ID, processedTerms, 2) // bigrams
		idx.indexNgrams(doc.ID, processedTerms, 3) // trigrams
	}

	// Update main index
	internalDoc := &Document{
		ID:       doc.ID,
		Content:  doc.Content,
		Metadata: doc.Metadata,
		Terms:    processedTerms,
		Length:   len(processedTerms),
	}

	idx.documents[doc.ID] = internalDoc
	idx.docLengths[doc.ID] = len(processedTerms)

	// Update term frequencies
	termCounts := make(map[string]int)
	for _, term := range processedTerms {
		termCounts[term]++
	}

	for term, count := range termCounts {
		if idx.termFreqs[term] == nil {
			idx.termFreqs[term] = make(map[string]int)
		}
		idx.termFreqs[term][doc.ID] = count
	}

	idx.docCount = len(idx.documents)
	idx.updateAverageDocLength()
}

// indexNgrams indexes n-grams for phrase search
func (idx *EnhancedBM25Index) indexNgrams(docID string, terms []string, n int) {
	if len(terms) < n {
		return
	}

	for i := 0; i <= len(terms)-n; i++ {
		ngram := strings.Join(terms[i:i+n], " ")

		var ngramIndex map[string]map[string][]int
		if n == 2 {
			ngramIndex = idx.bigramIndex
		} else if n == 3 {
			ngramIndex = idx.trigramIndex
		} else {
			continue
		}

		if ngramIndex[ngram] == nil {
			ngramIndex[ngram] = make(map[string][]int)
		}

		ngramIndex[ngram][docID] = append(ngramIndex[ngram][docID], i)
	}
}

// searchPhrases searches for exact phrase matches
func (idx *EnhancedBM25Index) searchPhrases(query ParsedQuery, limit int) []ai.SearchResult {
	results := make([]ai.SearchResult, 0)

	for _, phrase := range query.Phrases {
		// Tokenize and process phrase
		tokens := idx.tokenizer.Tokenize(phrase)
		if len(tokens) == 0 {
			continue
		}

		processedTokens := make([]string, 0, len(tokens))
		for _, token := range tokens {
			normalized := strings.ToLower(token)
			if !idx.stopWords[normalized] {
				if idx.enableStemming && len(normalized) > 2 {
					normalized = idx.stemmer.Stem(normalized)
				}
				processedTokens = append(processedTokens, normalized)
			}
		}

		// Search for phrase
		var candidateDocs map[string][]int

		if len(processedTokens) == 2 {
			// Use bigram index
			bigram := strings.Join(processedTokens, " ")
			candidateDocs = idx.bigramIndex[bigram]
		} else if len(processedTokens) == 3 {
			// Use trigram index
			trigram := strings.Join(processedTokens, " ")
			candidateDocs = idx.trigramIndex[trigram]
		} else {
			// For longer phrases, use position-based search
			candidateDocs = idx.searchLongPhrase(processedTokens)
		}

		// Convert to search results
		for docID, positions := range candidateDocs {
			doc := idx.documents[docID]
			results = append(results, ai.SearchResult{
				ID:       docID,
				Score:    float64(len(positions)) * 10.0, // Boost phrase matches
				Content:  doc.Content,
				Metadata: doc.Metadata,
				Explanation: &ai.SearchExplanation{
					TextScore:    float64(len(positions)) * 10.0,
					MatchedTerms: []string{phrase},
				},
			})
		}
	}

	return results
}

// searchLongPhrase searches for phrases longer than trigrams
func (idx *EnhancedBM25Index) searchLongPhrase(terms []string) map[string][]int {
	if len(terms) == 0 {
		return nil
	}

	// Start with documents containing the first term
	firstTermDocs := idx.termPositions[terms[0]]
	candidates := make(map[string][]int)

	for docID, firstPositions := range firstTermDocs {
		// Check if document contains all terms in sequence
		matches := make([]int, 0)

		for _, startPos := range firstPositions {
			found := true

			// Check subsequent terms at consecutive positions
			for i := 1; i < len(terms); i++ {
				term := terms[i]
				positions, exists := idx.termPositions[term][docID]
				if !exists {
					found = false
					break
				}

				// Check if term appears at expected position
				expectedPos := startPos + i
				posFound := false
				for _, pos := range positions {
					if pos == expectedPos {
						posFound = true
						break
					}
				}

				if !posFound {
					found = false
					break
				}
			}

			if found {
				matches = append(matches, startPos)
			}
		}

		if len(matches) > 0 {
			candidates[docID] = matches
		}
	}

	return candidates
}

// parseQuery parses the query for special operators and phrases
func (idx *EnhancedBM25Index) parseQuery(query string) ParsedQuery {
	parsed := ParsedQuery{
		OriginalQuery: query,
		Terms:         make([]string, 0),
		Phrases:       make([]string, 0),
		MustTerms:     make([]string, 0),
		MustNotTerms:  make([]string, 0),
	}

	// Extract phrases (quoted strings)
	phraseRegex := regexp.MustCompile(`"([^"]+)"`)
	phrases := phraseRegex.FindAllStringSubmatch(query, -1)
	for _, match := range phrases {
		parsed.Phrases = append(parsed.Phrases, match[1])
		parsed.HasPhrases = true
	}

	// Remove phrases from query
	processedQuery := phraseRegex.ReplaceAllString(query, "")

	// Extract must/must-not terms
	mustRegex := regexp.MustCompile(`\+(\w+)`)
	mustMatches := mustRegex.FindAllStringSubmatch(processedQuery, -1)
	for _, match := range mustMatches {
		parsed.MustTerms = append(parsed.MustTerms, match[1])
	}

	mustNotRegex := regexp.MustCompile(`-(\w+)`)
	mustNotMatches := mustNotRegex.FindAllStringSubmatch(processedQuery, -1)
	for _, match := range mustNotMatches {
		parsed.MustNotTerms = append(parsed.MustNotTerms, match[1])
	}

	// Remove operators from query
	processedQuery = mustRegex.ReplaceAllString(processedQuery, "")
	processedQuery = mustNotRegex.ReplaceAllString(processedQuery, "")

	// Tokenize remaining query
	tokens := idx.tokenizer.Tokenize(processedQuery)
	for _, token := range tokens {
		normalized := strings.ToLower(token)
		if !idx.stopWords[normalized] && len(normalized) > 1 {
			parsed.Terms = append(parsed.Terms, normalized)
		}
	}

	parsed.ProcessedQuery = strings.Join(parsed.Terms, " ")

	return parsed
}

// expandQuery expands query with synonyms
func (idx *EnhancedBM25Index) expandQuery(query ParsedQuery) ParsedQuery {
	expanded := query
	expandedTerms := make([]string, 0, len(query.Terms))

	for _, term := range query.Terms {
		expandedTerms = append(expandedTerms, term)

		// Add synonyms
		if synonyms, exists := idx.synonymMap[term]; exists {
			expandedTerms = append(expandedTerms, synonyms...)
		}
	}

	expanded.Terms = expandedTerms
	expanded.ProcessedQuery = strings.Join(expandedTerms, " ")

	return expanded
}

// searchFields performs field-specific search
func (idx *EnhancedBM25Index) searchFields(ctx context.Context, query ParsedQuery, fields []string, limit int) []ai.SearchResult {
	fieldResults := make(map[string][]ai.SearchResult)

	for _, field := range fields {
		if fieldIdx, exists := idx.fieldIndices[field]; exists {
			results, _ := fieldIdx.Search(ctx, query.ProcessedQuery, limit)
			fieldResults[field] = results
		}
	}

	// Merge field results with boosting
	merged := make([]ai.SearchResult, 0)
	docScores := make(map[string]float64)
	docContent := make(map[string]ai.SearchResult)

	for field, results := range fieldResults {
		boost := idx.fieldBoosts[field]
		if boost == 0 {
			boost = 1.0
		}

		for _, result := range results {
			docScores[result.ID] += result.Score * boost
			if _, exists := docContent[result.ID]; !exists {
				docContent[result.ID] = result
			}
		}
	}

	for docID, score := range docScores {
		result := docContent[docID]
		result.Score = score
		merged = append(merged, result)
	}

	return merged
}

// mergeResults merges multiple result sets with weights
func (idx *EnhancedBM25Index) mergeResults(results1, results2 []ai.SearchResult, weights map[string]float64) []ai.SearchResult {
	// Create score map
	docScores := make(map[string]float64)
	docContent := make(map[string]ai.SearchResult)

	// Add first result set
	weight1 := weights["main"]
	if weight1 == 0 {
		weight1 = 1.0
	}

	for _, result := range results1 {
		docScores[result.ID] = result.Score * weight1
		docContent[result.ID] = result
	}

	// Add second result set
	weight2 := weights["secondary"]
	if weight2 == 0 {
		weight2 = weights["fuzzy"]
		if weight2 == 0 {
			weight2 = 0.8
		}
	}

	for _, result := range results2 {
		docScores[result.ID] += result.Score * weight2
		if _, exists := docContent[result.ID]; !exists {
			docContent[result.ID] = result
		}
	}

	// Convert back to slice and sort
	merged := make([]ai.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		result := docContent[docID]
		result.Score = score
		merged = append(merged, result)
	}

	// Sort by score
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged
}

// fuzzySearch performs fuzzy matching
func (idx *EnhancedBM25Index) fuzzySearch(query ParsedQuery, maxDistance int, limit int) []ai.SearchResult {
	// Simple fuzzy search implementation
	// In production, you'd want to use a more sophisticated algorithm
	results := make([]ai.SearchResult, 0)

	// For each query term, find similar terms in the index
	for _, queryTerm := range query.Terms {
		for term := range idx.termFreqs {
			distance := editDistance(queryTerm, term)
			if distance <= maxDistance && distance > 0 {
				// Found a fuzzy match, search for this term
				docs := idx.termFreqs[term]
				for docID := range docs {
					doc := idx.documents[docID]
					results = append(results, ai.SearchResult{
						ID:       docID,
						Score:    1.0 / float64(distance+1), // Score based on edit distance
						Content:  doc.Content,
						Metadata: doc.Metadata,
						Explanation: &ai.SearchExplanation{
							TextScore:    1.0 / float64(distance+1),
							MatchedTerms: []string{fmt.Sprintf("%s~%s", queryTerm, term)},
						},
					})
				}
			}
		}
	}

	return results
}

// Helper structures and functions

// ParsedQuery represents a parsed search query
type ParsedQuery struct {
	OriginalQuery  string
	ProcessedQuery string
	Terms          []string
	Phrases        []string
	MustTerms      []string
	MustNotTerms   []string
	HasPhrases     bool
}

// EnhancedSearchRequest represents an enhanced search request
type EnhancedSearchRequest struct {
	Query         string             `json:"query"`
	Limit         int                `json:"limit"`
	MinScore      float64            `json:"min_score"`
	ExpandQuery   bool               `json:"expand_query"`
	EnableFuzzy   bool               `json:"enable_fuzzy"`
	FuzzyDistance int                `json:"fuzzy_distance"`
	SearchFields  []string           `json:"search_fields"`
	FieldWeights  map[string]float64 `json:"field_weights"`
}

// FieldDocument represents a document with field-specific content
type FieldDocument struct {
	ID       string                 `json:"id"`
	Fields   map[string]string      `json:"fields"`
	Metadata map[string]interface{} `json:"metadata"`
}

// GetAllContent returns all field content concatenated
func (d *FieldDocument) GetAllContent() string {
	var content strings.Builder
	for _, fieldContent := range d.Fields {
		content.WriteString(fieldContent)
		content.WriteString(" ")
	}
	return content.String()
}

// Stemmer interface for word stemming
type Stemmer interface {
	Stem(word string) string
}

// PorterStemmer implements Porter stemming algorithm
type PorterStemmer struct{}

// NewPorterStemmer creates a new Porter stemmer
func NewPorterStemmer() *PorterStemmer {
	return &PorterStemmer{}
}

// Stem applies Porter stemming to a word
func (s *PorterStemmer) Stem(word string) string {
	// Simplified Porter stemming - in production use a proper implementation
	// This is just a basic example

	// Remove common suffixes
	suffixes := []string{"ing", "ed", "ly", "es", "s"}
	for _, suffix := range suffixes {
		if strings.HasSuffix(word, suffix) && len(word) > len(suffix)+2 {
			return strings.TrimSuffix(word, suffix)
		}
	}

	return word
}

// QueryExpander expands queries with related terms
type QueryExpander struct {
	concepts map[string][]string
}

// NewQueryExpander creates a new query expander
func NewQueryExpander() *QueryExpander {
	return &QueryExpander{
		concepts: defaultConcepts(),
	}
}

// defaultSynonyms returns default synonym mappings
func defaultSynonyms() map[string][]string {
	return map[string][]string{
		"search": {"find", "query", "lookup"},
		"delete": {"remove", "erase", "drop"},
		"create": {"make", "build", "construct"},
		"update": {"modify", "change", "edit"},
		"fast":   {"quick", "rapid", "speedy"},
		"big":    {"large", "huge", "massive"},
		"small":  {"tiny", "little", "minor"},
		"good":   {"great", "excellent", "fine"},
		"bad":    {"poor", "terrible", "awful"},
		"new":    {"fresh", "recent", "latest"},
		"old":    {"ancient", "aged", "vintage"},
	}
}

// defaultConcepts returns default concept mappings
func defaultConcepts() map[string][]string {
	return map[string][]string{
		"database": {"storage", "persistence", "data"},
		"search":   {"retrieval", "query", "find"},
		"ai":       {"artificial intelligence", "machine learning", "ml"},
		"vector":   {"embedding", "representation", "feature"},
	}
}

// defaultFieldBoosts returns default field boost values
func defaultFieldBoosts() map[string]float64 {
	return map[string]float64{
		"title":       2.0,
		"description": 1.5,
		"content":     1.0,
		"tags":        1.8,
		"category":    1.3,
	}
}

// editDistance calculates the Levenshtein distance between two strings
func editDistance(s1, s2 string) int {
	if len(s1) == 0 {
		return len(s2)
	}
	if len(s2) == 0 {
		return len(s1)
	}

	// Create distance matrix
	matrix := make([][]int, len(s1)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(s2)+1)
	}

	// Initialize first column and row
	for i := 0; i <= len(s1); i++ {
		matrix[i][0] = i
	}
	for j := 0; j <= len(s2); j++ {
		matrix[0][j] = j
	}

	// Calculate distances
	for i := 1; i <= len(s1); i++ {
		for j := 1; j <= len(s2); j++ {
			if s1[i-1] == s2[j-1] {
				matrix[i][j] = matrix[i-1][j-1]
			} else {
				matrix[i][j] = 1 + min3(
					matrix[i-1][j],   // deletion
					matrix[i][j-1],   // insertion
					matrix[i-1][j-1], // substitution
				)
			}
		}
	}

	return matrix[len(s1)][len(s2)]
}

// min3 returns the minimum of three integers
func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}
