package ai

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"
)

// ExtendedQueryIntent extends the basic QueryIntent with additional fields
type ExtendedQueryIntent struct {
	QueryIntent
	Domain     string             `json:"domain"`     // tech, business, general, etc.
	Modifiers  []string           `json:"modifiers"`  // latest, best, how-to, etc.
	Attributes map[string]float64 `json:"attributes"` // urgency, specificity, complexity
}

// QueryEntity represents an entity extracted from the query
type QueryEntity struct {
	Text       string  `json:"text"`
	Type       string  `json:"type"` // person, org, location, date, etc.
	Confidence float64 `json:"confidence"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
}

// ExpandedQuery represents an expanded version of the original query
type ExpandedQuery struct {
	Original        string              `json:"original"`
	Normalized      string              `json:"normalized"`
	Tokens          []string            `json:"tokens"`
	Entities        []QueryEntity       `json:"entities"`
	Intent          ExtendedQueryIntent `json:"intent"`
	RelatedTerms    []string            `json:"related_terms"`
	Synonyms        map[string][]string `json:"synonyms"`
	ConceptualTerms []string            `json:"conceptual_terms"`
}

// QueryUnderstanding provides semantic analysis of search queries
type QueryUnderstanding struct {
	mu               sync.RWMutex
	intentClassifier *IntentClassifier
	entityExtractor  *EntityExtractor
	conceptExpander  *ConceptExpander
	modelManager     ModelManager
	cache            *queryCache
}

// NewQueryUnderstanding creates a new query understanding system
func NewQueryUnderstanding(modelManager ModelManager) *QueryUnderstanding {
	return &QueryUnderstanding{
		intentClassifier: NewIntentClassifier(),
		entityExtractor:  NewEntityExtractor(),
		conceptExpander:  NewConceptExpander(),
		modelManager:     modelManager,
		cache:            newQueryCache(1000),
	}
}

// AnalyzeQuery performs comprehensive query analysis
func (qu *QueryUnderstanding) AnalyzeQuery(ctx context.Context, query string) (*ExpandedQuery, error) {
	// Check cache first
	if cached, found := qu.cache.get(query); found {
		return cached, nil
	}

	// Normalize query
	normalized := qu.normalizeQuery(query)
	tokens := qu.tokenizeQuery(normalized)

	// Extract entities using the existing entity extractor
	extractedEntities, err := qu.entityExtractor.ExtractEntities(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("entity extraction failed: %w", err)
	}

	// Convert Entity to QueryEntity
	entities := make([]QueryEntity, len(extractedEntities))
	for i, e := range extractedEntities {
		entities[i] = QueryEntity{
			Text:       e.Text,
			Type:       e.Label,
			Confidence: e.Confidence,
			Start:      e.StartPos,
			End:        e.EndPos,
		}
	}

	// Classify intent
	intent := qu.intentClassifier.Classify(query, entities)

	// Expand query with related concepts
	relatedTerms := qu.conceptExpander.ExpandConcepts(tokens, intent)

	// Get synonyms for key terms
	synonyms := qu.getSynonyms(tokens)

	// Generate conceptual terms using embeddings
	conceptualTerms, err := qu.getConceptualTerms(ctx, query)
	if err != nil {
		// Non-fatal error, continue without conceptual terms
		conceptualTerms = []string{}
	}

	result := &ExpandedQuery{
		Original:        query,
		Normalized:      normalized,
		Tokens:          tokens,
		Entities:        entities,
		Intent:          intent,
		RelatedTerms:    relatedTerms,
		Synonyms:        synonyms,
		ConceptualTerms: conceptualTerms,
	}

	// Cache result
	qu.cache.set(query, result)

	return result, nil
}

// normalizeQuery normalizes the query text
func (qu *QueryUnderstanding) normalizeQuery(query string) string {
	// Convert to lowercase
	normalized := strings.ToLower(query)

	// Remove extra whitespace
	normalized = strings.TrimSpace(normalized)
	normalized = regexp.MustCompile(`\s+`).ReplaceAllString(normalized, " ")

	// Expand common contractions
	contractions := map[string]string{
		"what's":  "what is",
		"where's": "where is",
		"it's":    "it is",
		"don't":   "do not",
		"doesn't": "does not",
		"won't":   "will not",
		"can't":   "cannot",
		"i'm":     "i am",
		"you're":  "you are",
		"they're": "they are",
	}

	for contraction, expansion := range contractions {
		normalized = strings.ReplaceAll(normalized, contraction, expansion)
	}

	return normalized
}

// tokenizeQuery splits query into tokens
func (qu *QueryUnderstanding) tokenizeQuery(query string) []string {
	var tokens []string
	var currentToken strings.Builder

	for _, r := range query {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentToken.WriteRune(r)
		} else {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
		}
	}

	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}

	return tokens
}

// getSynonyms returns synonyms for query terms
func (qu *QueryUnderstanding) getSynonyms(tokens []string) map[string][]string {
	synonyms := make(map[string][]string)

	// Common synonym mappings
	synonymMap := map[string][]string{
		"find":    {"search", "locate", "discover", "retrieve"},
		"create":  {"make", "build", "generate", "construct"},
		"delete":  {"remove", "erase", "destroy", "eliminate"},
		"update":  {"modify", "change", "edit", "revise"},
		"show":    {"display", "present", "reveal", "exhibit"},
		"get":     {"retrieve", "fetch", "obtain", "acquire"},
		"best":    {"top", "optimal", "finest", "greatest"},
		"fast":    {"quick", "rapid", "swift", "speedy"},
		"new":     {"latest", "recent", "fresh", "novel"},
		"help":    {"assist", "aid", "support", "guide"},
		"error":   {"mistake", "fault", "bug", "issue"},
		"problem": {"issue", "trouble", "difficulty", "challenge"},
	}

	for _, token := range tokens {
		if syns, exists := synonymMap[token]; exists {
			synonyms[token] = syns
		}
	}

	return synonyms
}

// getConceptualTerms generates conceptually related terms using embeddings
func (qu *QueryUnderstanding) getConceptualTerms(ctx context.Context, query string) ([]string, error) {
	// Use embedding model to find conceptually similar terms
	_, err := qu.modelManager.GetEngine("all-MiniLM-L6-v2")
	if err != nil {
		return nil, err
	}

	// This would typically query a pre-built concept database
	// For now, return predefined conceptual expansions
	conceptMap := map[string][]string{
		"machine learning": {"artificial intelligence", "deep learning", "neural networks", "AI models"},
		"database":         {"data storage", "persistence", "data management", "information retrieval"},
		"search":           {"query", "retrieval", "finding", "discovery"},
		"vector":           {"embedding", "representation", "feature vector", "numerical encoding"},
	}

	var concepts []string
	lowerQuery := strings.ToLower(query)

	for key, values := range conceptMap {
		if strings.Contains(lowerQuery, key) {
			concepts = append(concepts, values...)
		}
	}

	return concepts, nil
}

// IntentClassifier classifies query intent
type IntentClassifier struct {
	patterns map[string][]*regexp.Regexp
}

// NewIntentClassifier creates a new intent classifier
func NewIntentClassifier() *IntentClassifier {
	return &IntentClassifier{
		patterns: map[string][]*regexp.Regexp{
			"question": {
				regexp.MustCompile(`^(what|where|when|who|why|how|which|can|does|is|are)`),
				regexp.MustCompile(`\?$`),
			},
			"navigation": {
				regexp.MustCompile(`(go to|navigate|open|show me|take me to)`),
				regexp.MustCompile(`(page|site|url|link)`),
			},
			"lookup": {
				regexp.MustCompile(`(find|search|look for|lookup|get|retrieve)`),
				regexp.MustCompile(`(information|data|details|record)`),
			},
			"transactional": {
				regexp.MustCompile(`\b(create|update|delete|modify|add|remove|insert|send)\s+\w+`),
				regexp.MustCompile(`(buy|purchase|order|download|subscribe)`),
			},
		},
	}
}

// Classify determines the intent of a query
func (ic *IntentClassifier) Classify(query string, entities []QueryEntity) ExtendedQueryIntent {
	lowerQuery := strings.ToLower(query)
	scores := make(map[string]float64)

	// Pattern matching with priority-based scoring
	patternWeights := map[string]float64{
		"transactional": 0.8, // Higher weight for action-based queries
		"question":      0.7, // High weight for question patterns
		"navigation":    0.6,
		"lookup":        0.4, // Lower weight for generic lookup patterns
	}

	for intentType, patterns := range ic.patterns {
		weight := patternWeights[intentType]
		if weight == 0 {
			weight = 0.5 // Default weight
		}

		for _, pattern := range patterns {
			if pattern.MatchString(lowerQuery) {
				scores[intentType] += weight
			}
		}
	}

	// Entity-based scoring
	for _, entity := range entities {
		switch entity.Type {
		case "command":
			scores["transactional"] += 0.3
		case "question_word":
			scores["question"] += 0.3
		}
	}

	// Find highest scoring intent
	var bestIntent string
	var bestScore float64
	for intent, score := range scores {
		if score > bestScore {
			bestScore = score
			bestIntent = intent
		}
	}

	// Default to lookup if no clear intent
	if bestIntent == "" {
		bestIntent = "lookup"
		bestScore = 0.5
	}

	// Determine domain
	domain := ic.detectDomain(query)

	// Extract modifiers
	modifiers := ic.extractModifiers(query)

	// Calculate attributes
	attributes := map[string]float64{
		"urgency":     ic.calculateUrgency(query),
		"specificity": ic.calculateSpecificity(query, entities),
		"complexity":  ic.calculateComplexity(query),
	}

	// Normalize confidence to be between 0 and 1
	confidence := bestScore
	if confidence > 1.0 {
		confidence = 1.0
	}

	return ExtendedQueryIntent{
		QueryIntent: QueryIntent{
			Type:       bestIntent,
			Confidence: confidence,
			Subtype:    "", // Can be filled based on more detailed classification
		},
		Domain:     domain,
		Modifiers:  modifiers,
		Attributes: attributes,
	}
}

// detectDomain identifies the query domain
func (ic *IntentClassifier) detectDomain(query string) string {
	domains := map[string][]string{
		"tech":     {"software", "code", "programming", "computer", "technology", "api", "database", "python", "java", "tensorflow", "machine", "learning", "ai", "artificial", "intelligence", "vector", "embedding"},
		"business": {"company", "revenue", "sales", "marketing", "customer", "profit", "market"},
		"science":  {"research", "study", "experiment", "theory", "hypothesis", "scientific"},
		"health":   {"medical", "health", "disease", "treatment", "doctor", "patient", "medicine"},
	}

	lowerQuery := strings.ToLower(query)
	for domain, keywords := range domains {
		for _, keyword := range keywords {
			if strings.Contains(lowerQuery, keyword) {
				return domain
			}
		}
	}

	return "general"
}

// extractModifiers finds query modifiers
func (ic *IntentClassifier) extractModifiers(query string) []string {
	modifiers := []string{}
	modifierPatterns := map[string]*regexp.Regexp{
		"latest":   regexp.MustCompile(`\b(latest|newest|recent|current)\b`),
		"best":     regexp.MustCompile(`\b(best|top|optimal|recommended)\b`),
		"tutorial": regexp.MustCompile(`\b(how to|tutorial|guide|learn)\b`),
		"compare":  regexp.MustCompile(`\b(compare|versus|vs|difference)\b`),
		"example":  regexp.MustCompile(`\b(example|sample|demo|illustration)\b`),
	}

	lowerQuery := strings.ToLower(query)
	for modifier, pattern := range modifierPatterns {
		if pattern.MatchString(lowerQuery) {
			modifiers = append(modifiers, modifier)
		}
	}

	return modifiers
}

// calculateUrgency estimates query urgency
func (ic *IntentClassifier) calculateUrgency(query string) float64 {
	urgentWords := []string{"urgent", "asap", "immediately", "now", "quickly", "fast"}
	lowerQuery := strings.ToLower(query)

	urgency := 0.0
	for _, word := range urgentWords {
		if strings.Contains(lowerQuery, word) {
			urgency += 0.2
		}
	}

	if urgency > 1.0 {
		urgency = 1.0
	}

	return urgency
}

// calculateSpecificity measures how specific the query is
func (ic *IntentClassifier) calculateSpecificity(query string, entities []QueryEntity) float64 {
	// More entities and longer queries tend to be more specific
	specificity := float64(len(entities)) * 0.1

	// Word count factor
	words := strings.Fields(query)
	specificity += float64(len(words)) * 0.05

	// Technical terms increase specificity
	technicalTerms := regexp.MustCompile(`\b(api|algorithm|function|method|class|interface)\b`)
	if technicalTerms.MatchString(strings.ToLower(query)) {
		specificity += 0.3
	}

	if specificity > 1.0 {
		specificity = 1.0
	}

	return specificity
}

// calculateComplexity estimates query complexity
func (ic *IntentClassifier) calculateComplexity(query string) float64 {
	complexity := 0.0

	// Logical operators
	if strings.Contains(query, " AND ") || strings.Contains(query, " OR ") || strings.Contains(query, " NOT ") {
		complexity += 0.3
	}

	// Multiple clauses
	if strings.Contains(query, ",") || strings.Contains(query, ";") {
		complexity += 0.2
	}

	// Question depth
	questionWords := []string{"why", "how", "what if", "explain"}
	lowerQuery := strings.ToLower(query)
	for _, word := range questionWords {
		if strings.Contains(lowerQuery, word) {
			complexity += 0.2
		}
	}

	if complexity > 1.0 {
		complexity = 1.0
	}

	return complexity
}

// queryCache provides caching for analyzed queries
type queryCache struct {
	mu       sync.RWMutex
	cache    map[string]cacheEntry
	maxSize  int
	eviction []string
}

type cacheEntry struct {
	query      *ExpandedQuery
	expiration time.Time
}

func newQueryCache(maxSize int) *queryCache {
	return &queryCache{
		cache:    make(map[string]cacheEntry),
		maxSize:  maxSize,
		eviction: make([]string, 0, maxSize),
	}
}

func (qc *queryCache) get(query string) (*ExpandedQuery, bool) {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	entry, exists := qc.cache[query]
	if !exists || time.Now().After(entry.expiration) {
		return nil, false
	}

	return entry.query, true
}

func (qc *queryCache) set(query string, expanded *ExpandedQuery) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	// Evict if at capacity
	if len(qc.cache) >= qc.maxSize && qc.cache[query].expiration.IsZero() {
		if len(qc.eviction) > 0 {
			oldest := qc.eviction[0]
			delete(qc.cache, oldest)
			qc.eviction = qc.eviction[1:]
		}
	}

	qc.cache[query] = cacheEntry{
		query:      expanded,
		expiration: time.Now().Add(10 * time.Minute),
	}
	qc.eviction = append(qc.eviction, query)
}
