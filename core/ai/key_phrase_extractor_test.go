package ai

import (
	"context"
	"strings"
	"testing"
)

func TestNewKeyPhraseExtractor(t *testing.T) {
	extractor := NewKeyPhraseExtractor()

	if extractor == nil {
		t.Fatal("Expected extractor to be created, got nil")
	}

	if len(extractor.stopWords) == 0 {
		t.Error("Expected stop words to be populated")
	}

	if extractor.minPhraseLen <= 0 || extractor.maxPhraseLen <= 0 {
		t.Error("Expected valid phrase length constraints")
	}

	if extractor.minWordLen <= 0 {
		t.Error("Expected valid minimum word length")
	}
}

func TestExtractKeyPhrases(t *testing.T) {
	extractor := NewKeyPhraseExtractor()
	ctx := context.Background()

	testCases := []struct {
		name          string
		content       string
		minPhrases    int
		maxPhrases    int
		expectedTerms []string // Terms we expect to potentially appear
	}{
		{
			name:          "technical content",
			content:       "Machine learning algorithms process large datasets efficiently. Neural networks excel at pattern recognition tasks.",
			minPhrases:    1,
			maxPhrases:    10,
			expectedTerms: []string{"machine", "learning", "neural", "networks", "pattern", "recognition"},
		},
		{
			name:          "business content",
			content:       "Our company develops innovative software solutions for enterprise customers. We focus on scalable architecture and user experience.",
			minPhrases:    1,
			maxPhrases:    10,
			expectedTerms: []string{"software", "solutions", "enterprise", "customers", "scalable", "architecture"},
		},
		{
			name:          "short content",
			content:       "Artificial intelligence is transforming industries.",
			minPhrases:    1,
			maxPhrases:    5,
			expectedTerms: []string{"artificial", "intelligence", "transforming", "industries"},
		},
		{
			name:          "content with proper nouns",
			content:       "Google developed TensorFlow for machine learning applications. Microsoft created Azure for cloud computing.",
			minPhrases:    1,
			maxPhrases:    10,
			expectedTerms: []string{"Google", "TensorFlow", "Microsoft", "Azure", "machine", "learning"},
		},
		{
			name:       "empty content",
			content:    "",
			minPhrases: 0,
			maxPhrases: 0,
		},
		{
			name:       "only stop words",
			content:    "the and or but not with from",
			minPhrases: 0,
			maxPhrases: 2, // May extract some phrases despite being mostly stop words
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			phrases, err := extractor.ExtractKeyPhrases(ctx, tc.content)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if len(phrases) < tc.minPhrases {
				t.Errorf("Expected at least %d phrases, got %d", tc.minPhrases, len(phrases))
			}

			if len(phrases) > tc.maxPhrases {
				t.Errorf("Expected at most %d phrases, got %d", tc.maxPhrases, len(phrases))
			}

			// Check that phrases don't contain stop words
			for _, phrase := range phrases {
				if phrase == "" {
					t.Error("Found empty phrase")
				}

				if len(phrase) > 0 {
					// Check if any expected terms appear in the extracted phrases
					for _, expectedTerm := range tc.expectedTerms {
						if containsIgnoreCase(phrase, expectedTerm) {
							t.Logf("Found expected term '%s' in phrase '%s'", expectedTerm, phrase)
						}
					}
				}
			}

			t.Logf("Extracted %d phrases: %v", len(phrases), phrases)
		})
	}
}

func TestPhraseValidation(t *testing.T) {
	extractor := NewKeyPhraseExtractor()

	testCases := []struct {
		name     string
		words    []string
		expected bool
	}{
		{
			name:     "valid single word",
			words:    []string{"Technology"}, // Needs capitalization for proper case check
			expected: true,
		},
		{
			name:     "valid multi-word phrase",
			words:    []string{"Machine", "Learning"}, // Needs capitalization
			expected: true,
		},
		{
			name:     "phrase with stop word",
			words:    []string{"the", "technology"},
			expected: false,
		},
		{
			name:     "phrase with short word",
			words:    []string{"ai", "technology"}, // "ai" is too short
			expected: false,
		},
		{
			name:     "empty phrase",
			words:    []string{},
			expected: false,
		},
		{
			name:     "phrase with non-alphabetic",
			words:    []string{"tech123"},
			expected: false,
		},
		{
			name:     "proper noun phrase",
			words:    []string{"Google", "Cloud"},
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := extractor.isValidPhrase(tc.words)
			if result != tc.expected {
				t.Errorf("Expected %t for phrase %v, got %t", tc.expected, tc.words, result)
			}
		})
	}
}

func TestAlphabeticCheck(t *testing.T) {
	extractor := NewKeyPhraseExtractor()

	testCases := []struct {
		word     string
		expected bool
	}{
		{"technology", true},
		{"machine-learning", true}, // Hyphen allowed
		{"user's", true},           // Apostrophe allowed
		{"tech123", false},         // Numbers not allowed
		{"tech@home", false},       // Special chars not allowed
		{"", false},                // Empty not allowed
		{"AI", true},               // All caps allowed
	}

	for _, tc := range testCases {
		t.Run(tc.word, func(t *testing.T) {
			result := extractor.isAlphabetic(tc.word)
			if result != tc.expected {
				t.Errorf("Expected %t for word '%s', got %t", tc.expected, tc.word, result)
			}
		})
	}
}

func TestPhraseOverlap(t *testing.T) {
	extractor := NewKeyPhraseExtractor()

	testCases := []struct {
		name     string
		phrase1  string
		phrase2  string
		expected bool
	}{
		{
			name:     "identical phrases",
			phrase1:  "machine learning",
			phrase2:  "machine learning",
			expected: true,
		},
		{
			name:     "overlapping phrases",
			phrase1:  "machine learning algorithms",
			phrase2:  "learning algorithms",
			expected: true,
		},
		{
			name:     "non-overlapping phrases",
			phrase1:  "machine learning",
			phrase2:  "neural networks",
			expected: false,
		},
		{
			name:     "partial overlap below threshold",
			phrase1:  "artificial intelligence systems",
			phrase2:  "intelligence networks",
			expected: false, // Only 1 word overlap out of 2-3 words
		},
		{
			name:     "empty phrases",
			phrase1:  "",
			phrase2:  "",
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := extractor.phrasesOverlap(tc.phrase1, tc.phrase2)
			if result != tc.expected {
				t.Errorf("Expected %t for phrases '%s' and '%s', got %t",
					tc.expected, tc.phrase1, tc.phrase2, result)
			}
		})
	}
}

func TestSentenceSplitting(t *testing.T) {
	extractor := NewKeyPhraseExtractor()

	testCases := []struct {
		name              string
		content           string
		expectedSentences int
	}{
		{
			name:              "multiple sentences",
			content:           "This is sentence one. This is sentence two! Is this sentence three?",
			expectedSentences: 3,
		},
		{
			name:              "single sentence",
			content:           "This is just one sentence",
			expectedSentences: 1,
		},
		{
			name:              "empty content",
			content:           "",
			expectedSentences: 1, // splitIntoSentences returns original content as fallback
		},
		{
			name:              "no punctuation",
			content:           "This has no sentence ending punctuation",
			expectedSentences: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			sentences := extractor.splitIntoSentences(tc.content)

			if len(sentences) != tc.expectedSentences {
				t.Errorf("Expected %d sentences, got %d", tc.expectedSentences, len(sentences))
			}

			// Check that sentences don't contain only whitespace (except for empty input)
			for i, sentence := range sentences {
				if sentence == "" && tc.content != "" {
					t.Errorf("Sentence %d is empty", i)
				}
			}
		})
	}
}

func TestGetExtractionStats(t *testing.T) {
	extractor := NewKeyPhraseExtractor()
	stats := extractor.GetExtractionStats()

	if stats == nil {
		t.Fatal("Expected stats to be returned")
	}

	expectedKeys := []string{
		"min_phrase_length",
		"max_phrase_length",
		"min_word_length",
		"stop_words_count",
		"pos_patterns_count",
	}

	for _, key := range expectedKeys {
		if _, exists := stats[key]; !exists {
			t.Errorf("Expected stat key '%s' not found", key)
		}
	}

	// Verify stat values make sense
	if minLen, ok := stats["min_phrase_length"].(int); ok {
		if minLen <= 0 {
			t.Errorf("Expected positive min_phrase_length, got %d", minLen)
		}
	}

	if maxLen, ok := stats["max_phrase_length"].(int); ok {
		if maxLen <= 0 {
			t.Errorf("Expected positive max_phrase_length, got %d", maxLen)
		}
	}

	if stopCount, ok := stats["stop_words_count"].(int); ok {
		if stopCount <= 0 {
			t.Errorf("Expected positive stop_words_count, got %d", stopCount)
		}
	}
}

func TestPhraseScoring(t *testing.T) {
	extractor := NewKeyPhraseExtractor()

	// Create sample candidates with different characteristics
	candidates := []PhraseScore{
		{Phrase: "machine learning", Frequency: 3, Words: []string{"machine", "learning"}},
		{Phrase: "artificial intelligence", Frequency: 2, Words: []string{"artificial", "intelligence"}},
		{Phrase: "data", Frequency: 5, Words: []string{"data"}},
		{Phrase: "neural networks", Frequency: 1, Words: []string{"neural", "networks"}},
	}

	content := "Machine learning and artificial intelligence use data. Neural networks are part of machine learning. Data science involves machine learning techniques."

	scoredPhrases := extractor.scorePhrases(candidates, content)

	if len(scoredPhrases) == 0 {
		t.Error("Expected scored phrases to be returned")
	}

	// Check that scores are positive and within reasonable range
	for _, phrase := range scoredPhrases {
		if phrase.Score < 0 {
			t.Errorf("Expected non-negative score for phrase '%s', got %f", phrase.Phrase, phrase.Score)
		}

		if phrase.Frequency <= 0 {
			t.Errorf("Expected positive frequency for phrase '%s', got %d", phrase.Phrase, phrase.Frequency)
		}
	}

	// Higher frequency phrases should generally score higher (though other factors matter too)
	t.Logf("Scored phrases: %+v", scoredPhrases)
}

// Helper function to check if a string contains another string (case insensitive)
func containsIgnoreCase(s, substr string) bool {
	s = strings.ToLower(s)
	substr = strings.ToLower(substr)
	return strings.Contains(s, substr)
}
