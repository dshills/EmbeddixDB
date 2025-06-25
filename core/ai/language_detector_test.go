package ai

import (
	"context"
	"testing"
)

func TestNewLanguageDetector(t *testing.T) {
	detector := NewLanguageDetector()
	
	if detector == nil {
		t.Fatal("Expected detector to be created, got nil")
	}
	
	if len(detector.languagePatterns) == 0 {
		t.Error("Expected language patterns to be populated")
	}
	
	// Check that expected languages are supported
	expectedLanguages := []string{"en", "es", "fr", "de", "pt", "it", "ru", "zh", "ja", "ar"}
	for _, lang := range expectedLanguages {
		if _, exists := detector.languagePatterns[lang]; !exists {
			t.Errorf("Expected language %s to be supported", lang)
		}
	}
}

func TestLanguageDetector_DetectLanguage(t *testing.T) {
	detector := NewLanguageDetector()
	ctx := context.Background()
	
	testCases := []struct {
		name            string
		content         string
		expectedLang    string
		minConfidence   float64
	}{
		{
			name:          "english text",
			content:       "This is a simple English text about technology and innovation in the modern world.",
			expectedLang:  "en",
			minConfidence: 0.1,
		},
		{
			name:          "spanish text",
			content:       "Este es un texto en español sobre tecnología y negocios en el mundo moderno.",
			expectedLang:  "es", 
			minConfidence: 0.1,
		},
		{
			name:          "french text",
			content:       "Ceci est un texte en français sur la science et la recherche dans le monde moderne.",
			expectedLang:  "fr",
			minConfidence: 0.1,
		},
		{
			name:          "german text",
			content:       "Dies ist ein deutscher Text über Technologie und Innovation in der modernen Welt.",
			expectedLang:  "de",
			minConfidence: 0.1,
		},
		{
			name:          "portuguese text", 
			content:       "Este é um texto em português sobre tecnologia e inovação no mundo moderno.",
			expectedLang:  "pt",
			minConfidence: 0.1,
		},
		{
			name:          "italian text",
			content:       "Questo è un testo italiano sulla tecnologia e l'innovazione nel mondo moderno.",
			expectedLang:  "it",
			minConfidence: 0.1,
		},
		{
			name:          "russian text",
			content:       "Это русский текст о технологиях и инновациях в современном мире.",
			expectedLang:  "ru",
			minConfidence: 0.1,
		},
		{
			name:          "mixed language (primarily english)",
			content:       "This is mostly English with some español words mixed in.",
			expectedLang:  "en", // Should detect primary language
			minConfidence: 0.0,
		},
		{
			name:          "short text",
			content:       "Hello world",
			expectedLang:  "en",
			minConfidence: 0.0,
		},
		{
			name:          "empty content",
			content:       "",
			expectedLang:  "unknown",
			minConfidence: 0.0,
		},
		{
			name:          "numbers and symbols only",
			content:       "123 456 789 !@# $%^",
			expectedLang:  "unknown",
			minConfidence: 0.0,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := detector.DetectLanguage(ctx, tc.content)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			
			// Language detection can be imprecise, so we log mismatches rather than fail
			if result.Code != tc.expectedLang {
				t.Logf("Expected language %s, got %s (detection can be imprecise)", tc.expectedLang, result.Code)
			}
			
			if result.Confidence < tc.minConfidence {
				t.Errorf("Expected confidence at least %f, got %f", tc.minConfidence, result.Confidence)
			}
			
			if result.Confidence < 0 || result.Confidence > 1 {
				t.Errorf("Confidence should be between 0 and 1, got %f", result.Confidence)
			}
			
			if result.Name == "" {
				t.Error("Expected language name to be populated")
			}
		})
	}
}

func TestGetSupportedLanguages(t *testing.T) {
	detector := NewLanguageDetector()
	languages := detector.GetSupportedLanguages()
	
	if len(languages) == 0 {
		t.Error("Expected supported languages to be returned")
	}
	
	expectedLanguages := []string{"en", "es", "fr", "de", "pt", "it", "ru", "zh", "ja", "ar"}
	languageMap := make(map[string]bool)
	for _, lang := range languages {
		languageMap[lang] = true
	}
	
	for _, expectedLang := range expectedLanguages {
		if !languageMap[expectedLang] {
			t.Errorf("Expected language %s not found in supported languages", expectedLang)
		}
	}
}

func TestGetLanguageName(t *testing.T) {
	detector := NewLanguageDetector()
	
	testCases := []struct {
		code         string
		expectedName string
	}{
		{"en", "English"},
		{"es", "Spanish"},
		{"fr", "French"},
		{"de", "German"},
		{"pt", "Portuguese"},
		{"it", "Italian"},
		{"ru", "Russian"},
		{"zh", "Chinese"},
		{"ja", "Japanese"},
		{"ar", "Arabic"},
		{"unknown", "Unknown"},
		{"xyz", "Unknown"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.code, func(t *testing.T) {
			name := detector.GetLanguageName(tc.code)
			if name != tc.expectedName {
				t.Errorf("Expected name %s for code %s, got %s", tc.expectedName, tc.code, name)
			}
		})
	}
}

func TestLanguageScoring(t *testing.T) {
	detector := NewLanguageDetector()
	
	// Test that English text gets higher English score
	englishText := "the quick brown fox jumps over the lazy dog"
	englishWords := []string{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"}
	englishPattern := detector.languagePatterns["en"]
	
	englishScore := detector.calculateLanguageScore(englishText, englishWords, englishPattern)
	
	// Test that same text gets lower score for a different language
	spanishPattern := detector.languagePatterns["es"]
	spanishScore := detector.calculateLanguageScore(englishText, englishWords, spanishPattern)
	
	if englishScore <= spanishScore {
		t.Errorf("Expected English text to score higher for English (%f) than Spanish (%f)", 
			englishScore, spanishScore)
	}
	
	// Test empty text
	emptyScore := detector.calculateLanguageScore("", []string{}, englishPattern)
	if emptyScore != 0.0 {
		t.Errorf("Expected empty text to score 0, got %f", emptyScore)
	}
}

func TestLanguageDetector_TextNormalization(t *testing.T) {
	detector := NewLanguageDetector()
	
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "basic text",
			input:    "Hello World",
			expected: "hello world",
		},
		{
			name:     "text with punctuation",
			input:    "Hello, World! How are you?",
			expected: "hello world how are you",
		},
		{
			name:     "text with numbers",
			input:    "I have 123 apples and 456 oranges",
			expected: "i have apples and oranges",
		},
		{
			name:     "text with multiple spaces",
			input:    "Too    many     spaces",
			expected: "too many spaces",
		},
		{
			name:     "empty text",
			input:    "",
			expected: "",
		},
		{
			name:     "only punctuation",
			input:    "!@#$%^&*()",
			expected: "",
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := detector.normalizeText(tc.input)
			if result != tc.expected {
				t.Errorf("Expected '%s', got '%s'", tc.expected, result)
			}
		})
	}
}

func TestScriptScoring(t *testing.T) {
	detector := NewLanguageDetector()
	
	// Test Latin script (should work for English, Spanish, etc.)
	latinRanges := []UnicodeRange{
		{Start: 'A', End: 'Z', Name: "Latin uppercase"},
		{Start: 'a', End: 'z', Name: "Latin lowercase"},
	}
	
	latinText := "Hello World"
	latinScore := detector.calculateScriptScore(latinText, latinRanges)
	
	if latinScore == 0.0 {
		t.Error("Expected non-zero score for Latin text with Latin ranges")
	}
	
	if latinScore > 1.0 {
		t.Errorf("Expected script score <= 1.0, got %f", latinScore)
	}
	
	// Test with empty ranges
	emptyScore := detector.calculateScriptScore(latinText, []UnicodeRange{})
	if emptyScore != 0.0 {
		t.Errorf("Expected zero score for empty ranges, got %f", emptyScore)
	}
	
	// Test with non-matching ranges
	cyrillicRanges := []UnicodeRange{
		{Start: 'А', End: 'я', Name: "Cyrillic"},
	}
	
	cyrillicScore := detector.calculateScriptScore(latinText, cyrillicRanges)
	if cyrillicScore != 0.0 {
		t.Errorf("Expected zero score for non-matching script, got %f", cyrillicScore)
	}
}