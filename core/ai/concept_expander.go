package ai

import (
	"strings"
	"sync"
)

// ConceptExpander expands queries with related concepts and terms
type ConceptExpander struct {
	mu              sync.RWMutex
	conceptMap      map[string][]string
	domainConcepts  map[string]map[string][]string
	relatedTerms    map[string][]string
}

// NewConceptExpander creates a new concept expander
func NewConceptExpander() *ConceptExpander {
	ce := &ConceptExpander{
		conceptMap:     make(map[string][]string),
		domainConcepts: make(map[string]map[string][]string),
		relatedTerms:   make(map[string][]string),
	}

	ce.initializeConcepts()
	return ce
}

// initializeConcepts sets up concept mappings
func (ce *ConceptExpander) initializeConcepts() {
	// General concept expansions
	ce.conceptMap = map[string][]string{
		// Tech concepts
		"ai":                {"artificial intelligence", "machine learning", "ml", "deep learning", "neural network"},
		"machine learning":  {"ml", "ai", "artificial intelligence", "predictive modeling", "data science"},
		"deep learning":     {"neural networks", "dl", "artificial neural networks", "deep neural networks"},
		"database":          {"db", "data storage", "persistence", "data management", "repository"},
		"search":            {"query", "find", "retrieve", "lookup", "discover"},
		"vector":            {"embedding", "feature vector", "representation", "encoding"},
		"embedding":         {"vector", "representation", "encoding", "feature extraction"},
		
		// Programming concepts
		"code":              {"source code", "program", "script", "implementation"},
		"function":          {"method", "procedure", "routine", "operation"},
		"class":             {"object", "type", "structure", "entity"},
		"api":               {"interface", "endpoint", "service", "integration"},
		"bug":               {"error", "issue", "defect", "problem", "fault"},
		"performance":       {"speed", "efficiency", "optimization", "throughput", "latency"},
		
		// Data concepts
		"data":              {"information", "records", "dataset", "content"},
		"analysis":          {"analytics", "examination", "study", "investigation"},
		"visualization":     {"viz", "chart", "graph", "plot", "diagram"},
		"statistics":        {"stats", "metrics", "measures", "analytics"},
		
		// Business concepts
		"customer":          {"client", "user", "consumer", "buyer"},
		"revenue":           {"income", "earnings", "sales", "profit"},
		"cost":              {"expense", "price", "investment", "spending"},
		"market":            {"marketplace", "industry", "sector", "economy"},
	}

	// Domain-specific concept expansions
	ce.domainConcepts = map[string]map[string][]string{
		"tech": {
			"deploy":    {"deployment", "release", "launch", "rollout", "publish"},
			"scale":     {"scaling", "scalability", "grow", "expand", "resize"},
			"cloud":     {"aws", "azure", "gcp", "saas", "paas", "iaas"},
			"security":  {"secure", "protection", "encryption", "authentication", "authorization"},
			"optimize":  {"optimization", "improve", "enhance", "tune", "refine"},
		},
		"business": {
			"growth":    {"increase", "expansion", "development", "progress"},
			"strategy":  {"plan", "approach", "tactics", "methodology"},
			"metrics":   {"kpi", "measures", "indicators", "analytics"},
			"roi":       {"return on investment", "profit", "value", "benefit"},
		},
		"science": {
			"hypothesis": {"theory", "assumption", "proposition", "conjecture"},
			"experiment": {"test", "trial", "study", "research"},
			"analysis":   {"examination", "investigation", "review", "assessment"},
			"data":       {"observations", "measurements", "results", "findings"},
		},
	}

	// Related terms for query expansion
	ce.relatedTerms = map[string][]string{
		// Action verbs
		"create":    {"make", "build", "generate", "construct", "develop"},
		"delete":    {"remove", "erase", "destroy", "eliminate", "clear"},
		"update":    {"modify", "change", "edit", "revise", "alter"},
		"find":      {"search", "locate", "discover", "identify", "detect"},
		"analyze":   {"examine", "study", "investigate", "evaluate", "assess"},
		
		// Descriptive terms
		"fast":      {"quick", "rapid", "speedy", "swift", "efficient"},
		"slow":      {"sluggish", "delayed", "lagging", "gradual"},
		"big":       {"large", "huge", "massive", "extensive", "substantial"},
		"small":     {"tiny", "little", "minor", "compact", "minimal"},
		"good":      {"excellent", "great", "superior", "quality", "effective"},
		"bad":       {"poor", "inferior", "problematic", "faulty", "defective"},
		
		// Temporal terms
		"new":       {"latest", "recent", "current", "modern", "fresh"},
		"old":       {"previous", "former", "legacy", "outdated", "historical"},
		"now":       {"currently", "presently", "immediately", "today"},
		"later":     {"future", "upcoming", "subsequently", "afterwards"},
	}
}

// ExpandConcepts expands query terms with related concepts
func (ce *ConceptExpander) ExpandConcepts(tokens []string, intent ExtendedQueryIntent) []string {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	expandedTerms := make(map[string]bool)
	
	// Process each token
	for _, token := range tokens {
		lowerToken := strings.ToLower(token)
		
		// Add general concept expansions
		if concepts, exists := ce.conceptMap[lowerToken]; exists {
			for _, concept := range concepts {
				expandedTerms[concept] = true
			}
		}
		
		// Add domain-specific expansions
		if domainConcepts, exists := ce.domainConcepts[intent.Domain]; exists {
			if concepts, exists := domainConcepts[lowerToken]; exists {
				for _, concept := range concepts {
					expandedTerms[concept] = true
				}
			}
		}
		
		// Add related terms
		if related, exists := ce.relatedTerms[lowerToken]; exists {
			for _, term := range related {
				expandedTerms[term] = true
			}
		}
	}

	// Convert map to slice
	result := make([]string, 0, len(expandedTerms))
	for term := range expandedTerms {
		result = append(result, term)
	}

	return result
}

// GetConceptsForTerm returns all expanded concepts for a single term
func (ce *ConceptExpander) GetConceptsForTerm(term string) []string {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	lowerTerm := strings.ToLower(term)
	concepts := []string{}

	// Check general concepts
	if generalConcepts, exists := ce.conceptMap[lowerTerm]; exists {
		concepts = append(concepts, generalConcepts...)
	}

	// Check related terms
	if related, exists := ce.relatedTerms[lowerTerm]; exists {
		concepts = append(concepts, related...)
	}

	return concepts
}

// AddCustomConcept adds a custom concept mapping
func (ce *ConceptExpander) AddCustomConcept(term string, concepts []string) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	lowerTerm := strings.ToLower(term)
	ce.conceptMap[lowerTerm] = concepts
}

// AddDomainConcept adds a domain-specific concept mapping
func (ce *ConceptExpander) AddDomainConcept(domain, term string, concepts []string) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	if _, exists := ce.domainConcepts[domain]; !exists {
		ce.domainConcepts[domain] = make(map[string][]string)
	}

	lowerTerm := strings.ToLower(term)
	ce.domainConcepts[domain][lowerTerm] = concepts
}

// GetRelatedTerms returns related terms for a given term
func (ce *ConceptExpander) GetRelatedTerms(term string) []string {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	lowerTerm := strings.ToLower(term)
	if related, exists := ce.relatedTerms[lowerTerm]; exists {
		return related
	}

	return []string{}
}

// ExpandWithContext expands concepts considering the full query context
func (ce *ConceptExpander) ExpandWithContext(query string, tokens []string, intent ExtendedQueryIntent) []string {
	// First get basic expansions
	expandedTerms := ce.ExpandConcepts(tokens, intent)

	// Add context-aware expansions
	contextTerms := ce.getContextualExpansions(query, intent)
	
	// Merge results
	termSet := make(map[string]bool)
	for _, term := range expandedTerms {
		termSet[term] = true
	}
	for _, term := range contextTerms {
		termSet[term] = true
	}

	// Convert to slice
	result := make([]string, 0, len(termSet))
	for term := range termSet {
		result = append(result, term)
	}

	return result
}

// getContextualExpansions provides context-aware term expansions
func (ce *ConceptExpander) getContextualExpansions(query string, intent ExtendedQueryIntent) []string {
	lowerQuery := strings.ToLower(query)
	contextTerms := []string{}

	// Pattern-based expansions
	patterns := map[string][]string{
		"how to":        {"tutorial", "guide", "instructions", "steps"},
		"what is":       {"definition", "explanation", "meaning", "description"},
		"difference":    {"comparison", "versus", "vs", "contrast"},
		"best":          {"top", "recommended", "optimal", "preferred"},
		"problem with":  {"issue", "error", "bug", "trouble"},
		"example":       {"sample", "demo", "instance", "illustration"},
		"list of":       {"collection", "set", "group", "catalog"},
	}

	for pattern, expansions := range patterns {
		if strings.Contains(lowerQuery, pattern) {
			contextTerms = append(contextTerms, expansions...)
		}
	}

	// Intent-based expansions
	switch intent.Type {
	case "question":
		contextTerms = append(contextTerms, "answer", "solution", "explanation", "information")
	case "lookup":
		contextTerms = append(contextTerms, "find", "search", "locate", "retrieve")
	case "transactional":
		contextTerms = append(contextTerms, "action", "operation", "execute", "perform")
	}

	return contextTerms
}