package ai

import (
	"context"
	"testing"
)

func TestNewSentimentAnalyzer(t *testing.T) {
	analyzer := NewSentimentAnalyzer()

	if analyzer == nil {
		t.Fatal("Expected analyzer to be created, got nil")
	}

	if len(analyzer.positiveWords) == 0 {
		t.Error("Expected positive words to be populated")
	}

	if len(analyzer.negativeWords) == 0 {
		t.Error("Expected negative words to be populated")
	}

	if len(analyzer.intensifiers) == 0 {
		t.Error("Expected intensifiers to be populated")
	}

	if len(analyzer.negators) == 0 {
		t.Error("Expected negators to be populated")
	}
}

func TestAnalyzeSentiment(t *testing.T) {
	analyzer := NewSentimentAnalyzer()
	ctx := context.Background()

	testCases := []struct {
		name          string
		content       string
		expectedLabel string
		minPolarity   float64
		maxPolarity   float64
		minConfidence float64
	}{
		{
			name:          "positive text",
			content:       "This is an amazing product! I absolutely love the excellent design and fantastic features.",
			expectedLabel: "positive",
			minPolarity:   0.1,
			maxPolarity:   1.0,
			minConfidence: 0.3,
		},
		{
			name:          "negative text",
			content:       "This is a terrible experience. The service was awful and disappointing.",
			expectedLabel: "negative",
			minPolarity:   -1.0,
			maxPolarity:   -0.1,
			minConfidence: 0.3,
		},
		{
			name:          "neutral text",
			content:       "The weather report indicates partly cloudy skies with moderate temperatures.",
			expectedLabel: "neutral",
			minPolarity:   -0.1,
			maxPolarity:   0.1,
			minConfidence: 0.0,
		},
		{
			name:          "mixed sentiment",
			content:       "The product is good but the service was bad.",
			expectedLabel: "neutral", // Should balance out
			minPolarity:   -0.5,
			maxPolarity:   0.5,
			minConfidence: 0.0,
		},
		{
			name:          "intensified positive",
			content:       "This is extremely amazing and absolutely fantastic!",
			expectedLabel: "positive",
			minPolarity:   0.3,
			maxPolarity:   1.0,
			minConfidence: 0.4,
		},
		{
			name:          "intensified negative",
			content:       "This is incredibly terrible and completely awful!",
			expectedLabel: "negative",
			minPolarity:   -1.0,
			maxPolarity:   -0.3,
			minConfidence: 0.4,
		},
		{
			name:          "negated positive",
			content:       "This is not good at all.",
			expectedLabel: "negative",
			minPolarity:   -1.0,
			maxPolarity:   0.0,
			minConfidence: 0.1,
		},
		{
			name:          "negated negative",
			content:       "This is not bad actually.",
			expectedLabel: "positive",
			minPolarity:   0.0,
			maxPolarity:   1.0,
			minConfidence: 0.1,
		},
		{
			name:          "empty content",
			content:       "",
			expectedLabel: "neutral",
			minPolarity:   0.0,
			maxPolarity:   0.0,
			minConfidence: 0.0,
		},
		{
			name:          "only punctuation",
			content:       "!@#$%^&*()",
			expectedLabel: "neutral",
			minPolarity:   0.0,
			maxPolarity:   0.0,
			minConfidence: 0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := analyzer.AnalyzeSentiment(ctx, tc.content)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Label != tc.expectedLabel {
				t.Errorf("Expected label %s, got %s", tc.expectedLabel, result.Label)
			}

			if result.Polarity < tc.minPolarity || result.Polarity > tc.maxPolarity {
				t.Errorf("Expected polarity between %f and %f, got %f",
					tc.minPolarity, tc.maxPolarity, result.Polarity)
			}

			if result.Confidence < tc.minConfidence {
				t.Errorf("Expected confidence at least %f, got %f",
					tc.minConfidence, result.Confidence)
			}

			if result.Confidence < 0 || result.Confidence > 1 {
				t.Errorf("Expected confidence between 0 and 1, got %f", result.Confidence)
			}
		})
	}
}

func TestSentimentLabelAssignment(t *testing.T) {
	analyzer := NewSentimentAnalyzer()

	testCases := []struct {
		score         float64
		expectedLabel string
	}{
		{0.5, "positive"},
		{0.15, "positive"},
		{0.1, "neutral"},
		{0.05, "neutral"},
		{0.0, "neutral"},
		{-0.05, "neutral"},
		{-0.1, "neutral"},
		{-0.15, "negative"},
		{-0.5, "negative"},
	}

	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			label := analyzer.getSentimentLabel(tc.score)
			if label != tc.expectedLabel {
				t.Errorf("Score %f: expected label %s, got %s",
					tc.score, tc.expectedLabel, label)
			}
		})
	}
}

func TestNegationDetection(t *testing.T) {
	analyzer := NewSentimentAnalyzer()

	testCases := []struct {
		name         string
		words        []string
		currentIndex int
		expected     bool
	}{
		{
			name:         "direct negation",
			words:        []string{"not", "good"},
			currentIndex: 1,
			expected:     true,
		},
		{
			name:         "no negation",
			words:        []string{"very", "good"},
			currentIndex: 1,
			expected:     false,
		},
		{
			name:         "distant negation",
			words:        []string{"not", "very", "very", "good"},
			currentIndex: 3,
			expected:     true,
		},
		{
			name:         "too distant negation",
			words:        []string{"not", "very", "very", "very", "good"},
			currentIndex: 4,
			expected:     false,
		},
		{
			name:         "multiple negators",
			words:        []string{"never", "really", "good"},
			currentIndex: 2,
			expected:     true,
		},
		{
			name:         "first word",
			words:        []string{"good", "morning"},
			currentIndex: 0,
			expected:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := analyzer.isNegated(tc.words, tc.currentIndex)
			if result != tc.expected {
				t.Errorf("Expected %t, got %t for words %v at index %d",
					tc.expected, result, tc.words, tc.currentIndex)
			}
		})
	}
}

func TestIntensityMultiplier(t *testing.T) {
	analyzer := NewSentimentAnalyzer()

	testCases := []struct {
		name          string
		words         []string
		currentIndex  int
		minMultiplier float64
		maxMultiplier float64
	}{
		{
			name:          "no intensifier",
			words:         []string{"good", "day"},
			currentIndex:  1,
			minMultiplier: 1.0,
			maxMultiplier: 1.0,
		},
		{
			name:          "very intensifier",
			words:         []string{"very", "good"},
			currentIndex:  1,
			minMultiplier: 1.5,
			maxMultiplier: 1.5,
		},
		{
			name:          "extremely intensifier",
			words:         []string{"extremely", "good"},
			currentIndex:  1,
			minMultiplier: 2.0,
			maxMultiplier: 2.0,
		},
		{
			name:          "multiple intensifiers",
			words:         []string{"very", "extremely", "good"},
			currentIndex:  2,
			minMultiplier: 2.5, // 1.5 * 2.0 = 3.0, but we only look back 2 words
			maxMultiplier: 3.0,
		},
		{
			name:          "diminisher",
			words:         []string{"slightly", "good"},
			currentIndex:  1,
			minMultiplier: 0.8,
			maxMultiplier: 0.8,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			multiplier := analyzer.getIntensityMultiplier(tc.words, tc.currentIndex)
			if multiplier < tc.minMultiplier || multiplier > tc.maxMultiplier {
				t.Errorf("Expected multiplier between %f and %f, got %f",
					tc.minMultiplier, tc.maxMultiplier, multiplier)
			}
		})
	}
}

func TestTextNormalization(t *testing.T) {
	analyzer := NewSentimentAnalyzer()

	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "basic text",
			input:    "This is a simple test.",
			expected: "this is a simple test",
		},
		{
			name:     "with URL",
			input:    "Check out https://example.com for more info.",
			expected: "check out for more info",
		},
		{
			name:     "with email",
			input:    "Contact us at test@example.com for support.",
			expected: "contact us at for support",
		},
		{
			name:     "with punctuation",
			input:    "Great!!! Awesome product!!!",
			expected: "great awesome product",
		},
		{
			name:     "multiple spaces",
			input:    "Too   many    spaces   here",
			expected: "too many spaces here",
		},
		{
			name:     "mixed content",
			input:    "Visit https://test.com or email info@test.com!!! Great service!!!",
			expected: "visit or email great service", // Note: some normalization may leave trace words
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := analyzer.normalizeText(tc.input)
			if result != tc.expected {
				t.Errorf("Expected '%s', got '%s'", tc.expected, result)
			}
		})
	}
}

func TestConfidenceCalculation(t *testing.T) {
	analyzer := NewSentimentAnalyzer()

	testCases := []struct {
		name          string
		score         float64
		wordCount     int
		minConfidence float64
		maxConfidence float64
	}{
		{
			name:          "high score, long text",
			score:         0.8,
			wordCount:     15,
			minConfidence: 0.8,
			maxConfidence: 1.0,
		},
		{
			name:          "high score, short text",
			score:         0.8,
			wordCount:     3,
			minConfidence: 0.6,
			maxConfidence: 0.8,
		},
		{
			name:          "low score, any length",
			score:         0.1,
			wordCount:     10,
			minConfidence: 0.1,
			maxConfidence: 0.2,
		},
		{
			name:          "zero score",
			score:         0.0,
			wordCount:     10,
			minConfidence: 0.0,
			maxConfidence: 0.0,
		},
		{
			name:          "negative score",
			score:         -0.5,
			wordCount:     10,
			minConfidence: 0.5,
			maxConfidence: 0.7,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			confidence := analyzer.calculateConfidence(tc.score, tc.wordCount)

			if confidence < tc.minConfidence || confidence > tc.maxConfidence {
				t.Errorf("Expected confidence between %f and %f, got %f",
					tc.minConfidence, tc.maxConfidence, confidence)
			}

			if confidence < 0 || confidence > 1 {
				t.Errorf("Confidence should be between 0 and 1, got %f", confidence)
			}
		})
	}
}

func TestGetSupportedFeatures(t *testing.T) {
	analyzer := NewSentimentAnalyzer()
	features := analyzer.GetSupportedFeatures()

	if len(features) == 0 {
		t.Error("Expected supported features to be returned")
	}

	expectedFeatures := map[string]bool{
		"polarity_detection":        true,
		"confidence_scoring":        true,
		"negation_handling":         true,
		"intensification_detection": true,
		"lexicon_based_analysis":    true,
	}

	for _, feature := range features {
		if !expectedFeatures[feature] {
			t.Errorf("Unexpected feature: %s", feature)
		}
	}

	for expectedFeature := range expectedFeatures {
		found := false
		for _, feature := range features {
			if feature == expectedFeature {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected feature %s not found", expectedFeature)
		}
	}
}

func TestSentimentConsistency(t *testing.T) {
	analyzer := NewSentimentAnalyzer()
	ctx := context.Background()

	// Test that similar texts produce similar results
	testPairs := []struct {
		text1   string
		text2   string
		maxDiff float64
	}{
		{
			text1:   "This is great!",
			text2:   "This is amazing!",
			maxDiff: 0.3,
		},
		{
			text1:   "This is terrible.",
			text2:   "This is awful.",
			maxDiff: 0.3,
		},
		{
			text1:   "The weather is nice.",
			text2:   "The temperature is moderate.",
			maxDiff: 0.6, // Neutral texts can vary more in sentiment detection
		},
	}

	for _, pair := range testPairs {
		t.Run("", func(t *testing.T) {
			result1, err1 := analyzer.AnalyzeSentiment(ctx, pair.text1)
			if err1 != nil {
				t.Fatalf("Error analyzing text1: %v", err1)
			}

			result2, err2 := analyzer.AnalyzeSentiment(ctx, pair.text2)
			if err2 != nil {
				t.Fatalf("Error analyzing text2: %v", err2)
			}

			diff := result1.Polarity - result2.Polarity
			if diff < 0 {
				diff = -diff
			}

			if diff > pair.maxDiff {
				t.Errorf("Results too different: %f vs %f (diff: %f, max: %f)",
					result1.Polarity, result2.Polarity, diff, pair.maxDiff)
			}
		})
	}
}
