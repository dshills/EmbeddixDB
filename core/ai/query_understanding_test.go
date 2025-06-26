package ai

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestQueryUnderstanding_AnalyzeQuery(t *testing.T) {
	// Create a mock model manager
	mockManager := &MockModelManager{
		engines: make(map[string]EmbeddingEngine),
	}
	mockManager.engines["all-MiniLM-L6-v2"] = &MockEmbeddingEngine{}

	qu := NewQueryUnderstanding(mockManager)
	ctx := context.Background()

	testCases := []struct {
		name            string
		query           string
		expectedIntent  string
		expectedDomain  string
		minEntities     int
		expectedTokens  []string
	}{
		{
			name:           "Simple question",
			query:          "What is machine learning?",
			expectedIntent: "question",
			expectedDomain: "tech",
			minEntities:    0,
			expectedTokens: []string{"what", "is", "machine", "learning"},
		},
		{
			name:           "Search query with entities",
			query:          "Find all documents about AI from January 2024",
			expectedIntent: "lookup",
			expectedDomain: "tech",
			minEntities:    1, // Should find date
			expectedTokens: []string{"find", "all", "documents", "about", "ai", "from", "january"},
		},
		{
			name:           "Transactional query",
			query:          "Delete old records from the database",
			expectedIntent: "transactional",
			expectedDomain: "tech",
			minEntities:    0,
			expectedTokens: []string{"delete", "old", "records", "from", "the", "database"},
		},
		{
			name:           "Complex technical query",
			query:          "How to optimize vector search performance in large databases?",
			expectedIntent: "question",
			expectedDomain: "tech",
			minEntities:    0,
			expectedTokens: []string{"how", "to", "optimize", "vector", "search", "performance", "in", "large", "databases"},
		},
		{
			name:           "Query with email",
			query:          "Send report to john.doe@example.com",
			expectedIntent: "transactional",
			expectedDomain: "general",
			minEntities:    1, // Should find email
			expectedTokens: []string{"send", "report", "to"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := qu.AnalyzeQuery(ctx, tc.query)
			if err != nil {
				t.Fatalf("AnalyzeQuery failed: %v", err)
			}

			// Check intent
			if result.Intent.Type != tc.expectedIntent {
				t.Errorf("Expected intent %s, got %s", tc.expectedIntent, result.Intent.Type)
			}

			// Check domain
			if result.Intent.Domain != tc.expectedDomain {
				t.Errorf("Expected domain %s, got %s", tc.expectedDomain, result.Intent.Domain)
			}

			// Check entities
			if len(result.Entities) < tc.minEntities {
				t.Errorf("Expected at least %d entities, got %d", tc.minEntities, len(result.Entities))
			}

			// Check tokens
			for _, expectedToken := range tc.expectedTokens {
				found := false
				for _, token := range result.Tokens {
					if token == expectedToken {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected token '%s' not found in result", expectedToken)
				}
			}

			// Verify expanded query has content
			if result.Normalized == "" {
				t.Error("Normalized query should not be empty")
			}

			// Check that related terms or synonyms were found
			if len(result.RelatedTerms) == 0 && len(result.Synonyms) == 0 {
				t.Log("Warning: No related terms or synonyms found")
			}
		})
	}
}

func TestQueryUnderstanding_NormalizeQuery(t *testing.T) {
	qu := NewQueryUnderstanding(&MockModelManager{})

	testCases := []struct {
		input    string
		expected string
	}{
		{
			input:    "What's the BEST way?",
			expected: "what is the best way?",
		},
		{
			input:    "   Multiple   spaces   between   words   ",
			expected: "multiple spaces between words",
		},
		{
			input:    "Don't won't can't",
			expected: "do not will not cannot",
		},
		{
			input:    "I'm sure you're right",
			expected: "i am sure you are right",
		},
	}

	for _, tc := range testCases {
		result := qu.normalizeQuery(tc.input)
		if result != tc.expected {
			t.Errorf("normalizeQuery(%s) = %s, expected %s", tc.input, result, tc.expected)
		}
	}
}

func TestIntentClassifier_Classify(t *testing.T) {
	ic := NewIntentClassifier()

	testCases := []struct {
		query          string
		expectedType   string
		expectedDomain string
		hasModifiers   bool
	}{
		{
			query:          "What is the latest version of TensorFlow?",
			expectedType:   "question",
			expectedDomain: "tech",
			hasModifiers:   true, // "latest"
		},
		{
			query:          "Create a new database connection",
			expectedType:   "transactional",
			expectedDomain: "tech",
			hasModifiers:   false,
		},
		{
			query:          "Navigate to the homepage",
			expectedType:   "navigation",
			expectedDomain: "general",
			hasModifiers:   false,
		},
		{
			query:          "Find the best restaurants in New York",
			expectedType:   "lookup",
			expectedDomain: "general",
			hasModifiers:   true, // "best"
		},
		{
			query:          "Compare Python vs Java performance",
			expectedType:   "lookup",
			expectedDomain: "tech",
			hasModifiers:   true, // "compare"
		},
	}

	for _, tc := range testCases {
		t.Run(tc.query, func(t *testing.T) {
			intent := ic.Classify(tc.query, []QueryEntity{})

			if intent.Type != tc.expectedType {
				t.Errorf("Expected intent type %s, got %s", tc.expectedType, intent.Type)
			}

			if intent.Domain != tc.expectedDomain {
				t.Errorf("Expected domain %s, got %s", tc.expectedDomain, intent.Domain)
			}

			if tc.hasModifiers && len(intent.Modifiers) == 0 {
				t.Error("Expected modifiers but found none")
			}

			// Check confidence is reasonable
			if intent.Confidence < 0 || intent.Confidence > 1 {
				t.Errorf("Invalid confidence score: %f", intent.Confidence)
			}
		})
	}
}

func TestIntentClassifier_Attributes(t *testing.T) {
	ic := NewIntentClassifier()

	// Test urgency calculation
	urgentQuery := "I need this fixed urgently ASAP!"
	intent := ic.Classify(urgentQuery, []QueryEntity{})
	if intent.Attributes["urgency"] < 0.3 {
		t.Errorf("Expected high urgency, got %f", intent.Attributes["urgency"])
	}

	// Test specificity calculation
	specificQuery := "Find the getUserById function in the UserService class of the auth module"
	specificEntities := []QueryEntity{
		{Text: "getUserById", Type: "function"},
		{Text: "UserService", Type: "class"},
	}
	intent = ic.Classify(specificQuery, specificEntities)
	if intent.Attributes["specificity"] < 0.3 {
		t.Errorf("Expected high specificity, got %f", intent.Attributes["specificity"])
	}

	// Test complexity calculation
	complexQuery := "How can I optimize the performance of nested queries in PostgreSQL when dealing with large datasets AND maintain ACID compliance?"
	intent = ic.Classify(complexQuery, []QueryEntity{})
	if intent.Attributes["complexity"] < 0.3 {
		t.Errorf("Expected high complexity, got %f", intent.Attributes["complexity"])
	}
}

func TestEntityExtractor_Integration(t *testing.T) {
	ee := NewEntityExtractor()
	ctx := context.Background()

	testCases := []struct {
		text         string
		expectedTypes []string
	}{
		{
			text:         "Contact john.doe@example.com or call (555) 123-4567",
			expectedTypes: []string{"EMAIL", "PHONE"},
		},
		{
			text:         "The meeting is on January 15, 2024 at 3:30 PM",
			expectedTypes: []string{"DATE"},
		},
		{
			text:         "Visit our website at https://www.example.com",
			expectedTypes: []string{"URL"},
		},
		{
			text:         "The project costs $150,000 USD",
			expectedTypes: []string{"MONEY"},
		},
		{
			text:         "Microsoft Corporation announced a partnership with OpenAI",
			expectedTypes: []string{"ORG"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.text, func(t *testing.T) {
			entities, err := ee.ExtractEntities(ctx, tc.text)
			if err != nil {
				t.Fatalf("ExtractEntities failed: %v", err)
			}

			foundTypes := make(map[string]bool)
			for _, entity := range entities {
				foundTypes[entity.Label] = true
			}

			for _, expectedType := range tc.expectedTypes {
				if !foundTypes[expectedType] {
					t.Errorf("Expected to find entity of type %s, but didn't", expectedType)
				}
			}
		})
	}
}

func TestConceptExpander_ExpandConcepts(t *testing.T) {
	ce := NewConceptExpander()

	testCases := []struct {
		tokens         []string
		domain         string
		expectedTerms  []string
	}{
		{
			tokens:        []string{"machine", "learning"},
			domain:        "tech",
			expectedTerms: []string{"ml", "ai", "artificial intelligence"},
		},
		{
			tokens:        []string{"create", "database"},
			domain:        "tech",
			expectedTerms: []string{"make", "build", "db", "data storage"},
		},
		{
			tokens:        []string{"fast", "search"},
			domain:        "tech",
			expectedTerms: []string{"quick", "rapid", "query", "find"},
		},
		{
			tokens:        []string{"customer", "revenue"},
			domain:        "business",
			expectedTerms: []string{"client", "user", "income", "sales"},
		},
	}

	for _, tc := range testCases {
		t.Run(strings.Join(tc.tokens, " "), func(t *testing.T) {
			intent := ExtendedQueryIntent{
				QueryIntent: QueryIntent{Type: "lookup", Confidence: 0.8},
				Domain: tc.domain,
			}
			expanded := ce.ExpandConcepts(tc.tokens, intent)

			// Check that at least some expected terms are present
			foundCount := 0
			for _, expected := range tc.expectedTerms {
				for _, term := range expanded {
					if strings.Contains(term, expected) {
						foundCount++
						break
					}
				}
			}

			if foundCount == 0 {
				t.Errorf("None of the expected terms %v were found in expanded terms %v", 
					tc.expectedTerms, expanded)
			}
		})
	}
}

func TestQueryCache(t *testing.T) {
	cache := newQueryCache(2) // Small cache for testing

	// Test basic get/set
	query1 := "test query 1"
	result1 := &ExpandedQuery{Original: query1}
	cache.set(query1, result1)

	cached, found := cache.get(query1)
	if !found {
		t.Error("Expected to find cached query")
	}
	if cached.Original != query1 {
		t.Error("Cached result doesn't match")
	}

	// Test cache miss
	_, found = cache.get("non-existent query")
	if found {
		t.Error("Expected cache miss")
	}

	// Test eviction
	query2 := "test query 2"
	result2 := &ExpandedQuery{Original: query2}
	cache.set(query2, result2)

	query3 := "test query 3"
	result3 := &ExpandedQuery{Original: query3}
	cache.set(query3, result3) // Should evict query1

	_, found = cache.get(query1)
	if found {
		t.Error("Expected query1 to be evicted")
	}

	// Both query2 and query3 should still be present
	_, found2 := cache.get(query2)
	_, found3 := cache.get(query3)
	if !found2 || !found3 {
		t.Error("Expected query2 and query3 to be in cache")
	}
}

func BenchmarkQueryUnderstanding_AnalyzeQuery(b *testing.B) {
	mockManager := &MockModelManager{
		engines: make(map[string]EmbeddingEngine),
	}
	mockManager.engines["all-MiniLM-L6-v2"] = &MockEmbeddingEngine{}
	
	qu := NewQueryUnderstanding(mockManager)
	ctx := context.Background()

	queries := []string{
		"What is machine learning?",
		"Find all documents about AI from 2024",
		"How to optimize database performance",
		"Create a new user account with email test@example.com",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		_, err := qu.AnalyzeQuery(ctx, query)
		if err != nil {
			b.Fatalf("AnalyzeQuery failed: %v", err)
		}
	}
}

// MockModelManager for testing
type MockModelManager struct {
	engines map[string]EmbeddingEngine
}

func (m *MockModelManager) LoadModel(ctx context.Context, modelName string, config ModelConfig) error {
	return nil
}

func (m *MockModelManager) UnloadModel(modelName string) error {
	return nil
}

func (m *MockModelManager) GetEngine(modelName string) (EmbeddingEngine, error) {
	if engine, exists := m.engines[modelName]; exists {
		return engine, nil
	}
	return nil, fmt.Errorf("model not found: %s", modelName)
}

func (m *MockModelManager) ListModels() []ModelInfo {
	return []ModelInfo{}
}

func (m *MockModelManager) GetModelHealth(modelName string) (ModelHealth, error) {
	return ModelHealth{
		ModelName: modelName,
		Status:    "healthy",
	}, nil
}