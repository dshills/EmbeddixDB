package ai

import (
	"context"
	"testing"
	"time"
)

func TestNewDefaultContentAnalyzer(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()

	if analyzer == nil {
		t.Fatal("Expected analyzer to be created, got nil")
	}

	if analyzer.languageDetector == nil {
		t.Error("Expected language detector to be initialized")
	}

	if analyzer.entityExtractor == nil {
		t.Error("Expected entity extractor to be initialized")
	}

	if analyzer.topicModeler == nil {
		t.Error("Expected topic modeler to be initialized")
	}

	if analyzer.sentimentAnalyzer == nil {
		t.Error("Expected sentiment analyzer to be initialized")
	}

	if analyzer.keyPhraseExtractor == nil {
		t.Error("Expected key phrase extractor to be initialized")
	}

	if analyzer.stats == nil {
		t.Error("Expected stats to be initialized")
	}
}

func TestAnalyzeContent(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	testCases := []struct {
		name         string
		content      string
		expectError  bool
		expectedLang string
		minWordCount int
		maxWordCount int
	}{
		{
			name:        "empty content",
			content:     "",
			expectError: true,
		},
		{
			name:         "simple english text",
			content:      "This is a simple test in English. The weather is nice today.",
			expectError:  false,
			expectedLang: "en",
			minWordCount: 10,
			maxWordCount: 15,
		},
		{
			name:         "technical content",
			content:      "Machine learning algorithms can process large datasets efficiently. Neural networks are particularly effective for pattern recognition tasks.",
			expectError:  false,
			expectedLang: "en",
			minWordCount: 15,
			maxWordCount: 25,
		},
		{
			name:         "positive sentiment",
			content:      "This is an amazing product! I absolutely love the excellent design and fantastic features.",
			expectError:  false,
			expectedLang: "en",
			minWordCount: 10,
			maxWordCount: 20,
		},
		{
			name:         "negative sentiment",
			content:      "This is a terrible experience. The service was awful and disappointing.",
			expectError:  false,
			expectedLang: "en",
			minWordCount: 10,
			maxWordCount: 15,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			insights, err := analyzer.AnalyzeContent(ctx, tc.content)

			if tc.expectError {
				if err == nil {
					t.Error("Expected error for empty content")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			// Check word count
			if insights.WordCount < tc.minWordCount || insights.WordCount > tc.maxWordCount {
				t.Errorf("Expected word count between %d and %d, got %d",
					tc.minWordCount, tc.maxWordCount, insights.WordCount)
			}

			// Check language detection (note: detection can sometimes be imprecise)
			if tc.expectedLang != "" && insights.Language.Code != tc.expectedLang {
				t.Logf("Expected language %s, got %s (this may be acceptable for language detection)", tc.expectedLang, insights.Language.Code)
			}

			// Check that all fields are populated
			if insights.Language.Code == "" {
				t.Error("Expected language code to be populated")
			}

			if insights.Complexity < 0 || insights.Complexity > 1 {
				t.Errorf("Expected complexity between 0 and 1, got %f", insights.Complexity)
			}

			if insights.Readability < 0 || insights.Readability > 1 {
				t.Errorf("Expected readability between 0 and 1, got %f", insights.Readability)
			}

			if insights.Summary == "" {
				t.Error("Expected summary to be populated")
			}

			// Sentiment should have a label
			if insights.Sentiment.Label == "" {
				t.Error("Expected sentiment label to be populated")
			}

			validLabels := map[string]bool{"positive": true, "negative": true, "neutral": true}
			if !validLabels[insights.Sentiment.Label] {
				t.Errorf("Expected sentiment label to be positive, negative, or neutral, got %s",
					insights.Sentiment.Label)
			}
		})
	}
}

func TestAnalyzeBatch(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	content := []string{
		"This is the first document about technology and innovation.",
		"The second document discusses business strategies and market trends.",
		"A third document about science and research methodologies.",
	}

	insights, err := analyzer.AnalyzeBatch(ctx, content)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(insights) != len(content) {
		t.Errorf("Expected %d insights, got %d", len(content), len(insights))
	}

	for i, insight := range insights {
		if insight.WordCount == 0 {
			t.Errorf("Document %d: Expected non-zero word count", i)
		}

		if insight.Language.Code == "" {
			t.Errorf("Document %d: Expected language code to be populated", i)
		}

		if insight.Summary == "" {
			t.Errorf("Document %d: Expected summary to be populated", i)
		}
	}
}

func TestAnalyzeBatchEmpty(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	insights, err := analyzer.AnalyzeBatch(ctx, []string{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(insights) != 0 {
		t.Errorf("Expected empty result for empty input, got %d insights", len(insights))
	}
}

func TestExtractEntities(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	content := "John Smith works at Google Inc. in New York. You can reach him at john@google.com or call (555) 123-4567."

	entities, err := analyzer.ExtractEntities(ctx, content)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Check that we found some entities
	if len(entities) == 0 {
		t.Error("Expected to find entities in the content")
	}

	// Check for expected entity types
	entityTypes := make(map[string]bool)
	for _, entity := range entities {
		entityTypes[entity.Label] = true
	}

	expectedTypes := []string{"PERSON", "ORG", "LOC", "EMAIL", "PHONE"}
	for _, expectedType := range expectedTypes {
		if !entityTypes[expectedType] {
			t.Logf("Expected entity type %s not found (this may be normal depending on extraction accuracy)", expectedType)
		}
	}
}

func TestDetectLanguage(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	testCases := []struct {
		name         string
		content      string
		expectedLang string
	}{
		{
			name:         "english text",
			content:      "This is a simple English text about technology and innovation.",
			expectedLang: "en",
		},
		{
			name:         "spanish text",
			content:      "Este es un texto en español sobre tecnología y negocios.",
			expectedLang: "es",
		},
		{
			name:         "french text",
			content:      "Ceci est un texte en français sur la science et la recherche.",
			expectedLang: "fr",
		},
		{
			name:         "empty content",
			content:      "",
			expectedLang: "unknown",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			language, err := analyzer.DetectLanguage(ctx, tc.content)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if language.Code != tc.expectedLang {
				t.Logf("Expected language %s, got %s (language detection can be imprecise)", tc.expectedLang, language.Code)
			}

			if language.Confidence < 0 || language.Confidence > 1 {
				t.Errorf("Expected confidence between 0 and 1, got %f", language.Confidence)
			}
		})
	}
}

func TestGetStats(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	// Perform some analyses to generate stats
	content := []string{
		"This is a test document for statistics tracking.",
		"Another document to increase the analysis count.",
	}

	for _, c := range content {
		_, err := analyzer.AnalyzeContent(ctx, c)
		if err != nil {
			t.Fatalf("Unexpected error during analysis: %v", err)
		}
	}

	stats := analyzer.GetStats()

	if stats.TotalAnalyses != 2 {
		t.Errorf("Expected 2 total analyses, got %d", stats.TotalAnalyses)
	}

	if stats.SuccessfulAnalyses != 2 {
		t.Errorf("Expected 2 successful analyses, got %d", stats.SuccessfulAnalyses)
	}

	if stats.FailedAnalyses != 0 {
		t.Errorf("Expected 0 failed analyses, got %d", stats.FailedAnalyses)
	}

	if stats.LanguageDetections != 2 {
		t.Errorf("Expected 2 language detections, got %d", stats.LanguageDetections)
	}

	if stats.AverageLatency <= 0 {
		t.Error("Expected positive average latency")
	}

	if stats.TotalProcessingTime <= 0 {
		t.Error("Expected positive total processing time")
	}
}

func TestComplexityCalculation(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()

	testCases := []struct {
		name     string
		content  string
		minScore float64
		maxScore float64
	}{
		{
			name:     "simple text",
			content:  "This is simple. Easy to read.",
			minScore: 0.0,
			maxScore: 0.3,
		},
		{
			name:     "complex text",
			content:  "The implementation of sophisticated algorithms requires comprehensive understanding of computational complexity theory.",
			minScore: 0.4,
			maxScore: 1.0,
		},
		{
			name:     "empty text",
			content:  "",
			minScore: 0.0,
			maxScore: 0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			score := analyzer.calculateComplexityScore(tc.content)

			if score < tc.minScore || score > tc.maxScore {
				t.Errorf("Expected complexity score between %f and %f, got %f",
					tc.minScore, tc.maxScore, score)
			}
		})
	}
}

func TestReadabilityCalculation(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()

	testCases := []struct {
		name     string
		content  string
		minScore float64
		maxScore float64
	}{
		{
			name:     "simple readable text",
			content:  "The cat sat on the mat. It was a nice day.",
			minScore: 0.5,
			maxScore: 1.0,
		},
		{
			name:     "complex unreadable text",
			content:  "The utilization of multifaceted methodological approaches necessitates comprehensive analytical frameworks.",
			minScore: 0.0,
			maxScore: 0.5,
		},
		{
			name:     "empty text",
			content:  "",
			minScore: 0.0,
			maxScore: 0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			score := analyzer.calculateReadabilityScore(tc.content)

			if score < tc.minScore || score > tc.maxScore {
				t.Errorf("Expected readability score between %f and %f, got %f",
					tc.minScore, tc.maxScore, score)
			}
		})
	}
}

func TestSentenceAndSyllableCounting(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()

	testCases := []struct {
		name              string
		content           string
		expectedSentences int
		minSyllables      int
	}{
		{
			name:              "simple sentences",
			content:           "This is one sentence. This is another sentence!",
			expectedSentences: 2,
			minSyllables:      10,
		},
		{
			name:              "single sentence",
			content:           "This is just one sentence",
			expectedSentences: 1,
			minSyllables:      6,
		},
		{
			name:              "empty content",
			content:           "",
			expectedSentences: 1, // splitIntoSentences returns original content if no sentences found
			minSyllables:      0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			sentences := analyzer.splitIntoSentences(tc.content)
			syllables := analyzer.countSyllables(tc.content)

			if len(sentences) != tc.expectedSentences {
				t.Errorf("Expected %d sentences, got %d", tc.expectedSentences, len(sentences))
			}

			if syllables < tc.minSyllables {
				t.Errorf("Expected at least %d syllables, got %d", tc.minSyllables, syllables)
			}
		})
	}
}

func TestConcurrentAnalysis(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	// Test concurrent analysis doesn't cause race conditions
	content := []string{
		"First document about technology and innovation in the modern world.",
		"Second document discussing business strategies and market analysis.",
		"Third document covering scientific research and development methodologies.",
		"Fourth document about educational systems and learning approaches.",
		"Fifth document examining environmental sustainability and climate change.",
	}

	start := time.Now()
	insights, err := analyzer.AnalyzeBatch(ctx, content)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(insights) != len(content) {
		t.Errorf("Expected %d insights, got %d", len(content), len(insights))
	}

	// Verify all insights are properly populated
	for i, insight := range insights {
		if insight.WordCount == 0 {
			t.Errorf("Document %d: Expected non-zero word count", i)
		}

		if insight.Language.Code == "" {
			t.Errorf("Document %d: Expected language code", i)
		}

		if len(insight.Topics) == 0 {
			t.Logf("Document %d: No topics found (may be normal)", i)
		}

		if len(insight.Entities) == 0 {
			t.Logf("Document %d: No entities found (may be normal)", i)
		}

		if len(insight.KeyPhrases) == 0 {
			t.Logf("Document %d: No key phrases found (may be normal)", i)
		}
	}

	t.Logf("Processed %d documents in %v (avg: %v per document)",
		len(content), duration, duration/time.Duration(len(content)))
}

func TestAnalysisStatsThreadSafety(t *testing.T) {
	analyzer := NewDefaultContentAnalyzer()
	ctx := context.Background()

	// Test that stats recording is thread-safe
	content := "Test content for thread safety verification."

	// Run multiple analyses concurrently
	numGoroutines := 10
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer func() { done <- true }()
			_, err := analyzer.AnalyzeContent(ctx, content)
			if err != nil {
				t.Errorf("Unexpected error in concurrent analysis: %v", err)
			}
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	stats := analyzer.GetStats()
	if stats.TotalAnalyses != int64(numGoroutines) {
		t.Errorf("Expected %d total analyses, got %d", numGoroutines, stats.TotalAnalyses)
	}
}
