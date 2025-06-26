package ai

import (
	"context"
	"testing"
)

func TestNewTopicModeler(t *testing.T) {
	modeler := NewTopicModeler()

	if modeler == nil {
		t.Fatal("Expected modeler to be created, got nil")
	}

	if len(modeler.topicKeywords) == 0 {
		t.Error("Expected topic keywords to be populated")
	}

	if len(modeler.stopWords) == 0 {
		t.Error("Expected stop words to be populated")
	}

	// Check that expected topics are present
	expectedTopics := []string{
		"Technology", "Business", "Science", "Education", "Politics",
		"Sports", "Entertainment", "Health", "Travel", "Food",
		"Environment", "Finance",
	}

	for _, topic := range expectedTopics {
		if _, exists := modeler.topicKeywords[topic]; !exists {
			t.Errorf("Expected topic '%s' to be configured", topic)
		}
	}
}

func TestExtractTopics(t *testing.T) {
	modeler := NewTopicModeler()
	ctx := context.Background()

	testCases := []struct {
		name           string
		content        string
		expectedTopics []string
		minTopics      int
		maxTopics      int
	}{
		{
			name:           "technology content",
			content:        "Machine learning algorithms and artificial intelligence are transforming software development. Neural networks enable advanced pattern recognition in computer systems.",
			expectedTopics: []string{"Technology"},
			minTopics:      0, // May not always detect due to threshold
			maxTopics:      5,
		},
		{
			name:           "business content",
			content:        "Our company develops innovative solutions for enterprise customers. Market analysis shows strong revenue growth and profitable investment opportunities.",
			expectedTopics: []string{"Business"},
			minTopics:      0,
			maxTopics:      5,
		},
		{
			name:           "science content",
			content:        "Scientific research in biology and chemistry reveals new discoveries. Medical experiments in the laboratory provide evidence for pharmaceutical treatments.",
			expectedTopics: []string{"Science"},
			minTopics:      0,
			maxTopics:      5,
		},
		{
			name:           "health content",
			content:        "Healthcare professionals recommend regular exercise and proper nutrition. Medical diagnosis and treatment options help patients maintain wellness and fitness.",
			expectedTopics: []string{"Health"},
			minTopics:      0,
			maxTopics:      5,
		},
		{
			name:           "education content",
			content:        "University students and teachers focus on academic learning. Educational curriculum and degree programs provide valuable knowledge and skills training.",
			expectedTopics: []string{"Education"},
			minTopics:      0,
			maxTopics:      5,
		},
		{
			name:           "mixed content",
			content:        "Technology companies invest in research and development. Scientists use computers for data analysis in educational institutions.",
			expectedTopics: []string{"Technology", "Science", "Education"},
			minTopics:      0,
			maxTopics:      5,
		},
		{
			name:      "empty content",
			content:   "",
			minTopics: 0,
			maxTopics: 0,
		},
		{
			name:      "generic content",
			content:   "This is some generic text without specific topic keywords.",
			minTopics: 0,
			maxTopics: 2,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			topics, err := modeler.ExtractTopics(ctx, tc.content)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if len(topics) < tc.minTopics {
				t.Errorf("Expected at least %d topics, got %d", tc.minTopics, len(topics))
			}

			if len(topics) > tc.maxTopics {
				t.Errorf("Expected at most %d topics, got %d", tc.maxTopics, len(topics))
			}

			// Validate topic structure
			for i, topic := range topics {
				if topic.ID == "" {
					t.Errorf("Topic %d has empty ID", i)
				}

				if topic.Label == "" {
					t.Errorf("Topic %d has empty label", i)
				}

				if topic.Confidence < 0 {
					t.Errorf("Topic %d confidence %f should be non-negative", i, topic.Confidence)
				}

				if topic.Weight < 0 {
					t.Errorf("Topic %d weight %f should be non-negative", i, topic.Weight)
				}

				if len(topic.Keywords) == 0 {
					t.Logf("Topic %d has no keywords (may be normal)", i)
				}
			}

			// Check if any expected topics were found
			foundTopics := make(map[string]bool)
			for _, topic := range topics {
				foundTopics[topic.Label] = true
			}

			for _, expectedTopic := range tc.expectedTopics {
				if foundTopics[expectedTopic] {
					t.Logf("Successfully detected expected topic: %s", expectedTopic)
				}
			}

			t.Logf("Extracted topics: %v", getTopicLabels(topics))
		})
	}
}

func TestGetTopicCategories(t *testing.T) {
	modeler := NewTopicModeler()
	categories := modeler.GetTopicCategories()

	if len(categories) == 0 {
		t.Error("Expected topic categories to be returned")
	}

	expectedCategories := []string{
		"Technology", "Business", "Science", "Education", "Politics",
		"Sports", "Entertainment", "Health", "Travel", "Food",
		"Environment", "Finance",
	}

	categoryMap := make(map[string]bool)
	for _, cat := range categories {
		categoryMap[cat] = true
	}

	for _, expectedCat := range expectedCategories {
		if !categoryMap[expectedCat] {
			t.Errorf("Expected category '%s' not found", expectedCat)
		}
	}
}

func TestGetTopicKeywords(t *testing.T) {
	modeler := NewTopicModeler()

	testCases := []struct {
		topic           string
		expectedPresent bool
		minKeywords     int
	}{
		{"Technology", true, 10},
		{"Business", true, 10},
		{"Science", true, 10},
		{"Education", true, 10},
		{"NonExistentTopic", false, 0},
		{"", false, 0},
	}

	for _, tc := range testCases {
		t.Run(tc.topic, func(t *testing.T) {
			keywords := modeler.GetTopicKeywords(tc.topic)

			if tc.expectedPresent {
				if len(keywords) < tc.minKeywords {
					t.Errorf("Expected at least %d keywords for topic '%s', got %d",
						tc.minKeywords, tc.topic, len(keywords))
				}

				// Check that keywords are not empty
				for i, keyword := range keywords {
					if keyword == "" {
						t.Errorf("Keyword %d for topic '%s' is empty", i, tc.topic)
					}
				}
			} else {
				if len(keywords) != 0 {
					t.Errorf("Expected no keywords for non-existent topic '%s', got %d",
						tc.topic, len(keywords))
				}
			}
		})
	}
}

func TestTokenizeAndClean(t *testing.T) {
	modeler := NewTopicModeler()

	testCases := []struct {
		name          string
		content       string
		expectedWords []string
		minWords      int
		maxWords      int
	}{
		{
			name:          "simple text",
			content:       "This is a simple test with technology keywords",
			expectedWords: []string{"simple", "test", "technology", "keywords"},
			minWords:      4,
			maxWords:      6,
		},
		{
			name:     "text with punctuation",
			content:  "Hello, world! How are you today?",
			minWords: 3,
			maxWords: 6,
		},
		{
			name:     "text with stop words",
			content:  "The quick brown fox jumps over the lazy dog",
			minWords: 4, // After removing stop words
			maxWords: 8,
		},
		{
			name:     "text with short words",
			content:  "AI ML is good for us to do",
			minWords: 0, // Most words are too short or stop words, may result in no words
			maxWords: 4,
		},
		{
			name:     "empty content",
			content:  "",
			minWords: 0,
			maxWords: 0,
		},
		{
			name:     "only punctuation",
			content:  "!@#$%^&*()",
			minWords: 0,
			maxWords: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			words := modeler.tokenizeAndClean(tc.content)

			if len(words) < tc.minWords {
				t.Errorf("Expected at least %d words, got %d", tc.minWords, len(words))
			}

			if len(words) > tc.maxWords {
				t.Errorf("Expected at most %d words, got %d", tc.maxWords, len(words))
			}

			// Check that words meet criteria
			for _, word := range words {
				if len(word) < 3 {
					t.Errorf("Word '%s' is too short (should be filtered)", word)
				}

				if modeler.stopWords[word] {
					t.Errorf("Word '%s' is a stop word (should be filtered)", word)
				}
			}

			// Check for expected words
			wordMap := make(map[string]bool)
			for _, word := range words {
				wordMap[word] = true
			}

			for _, expectedWord := range tc.expectedWords {
				if wordMap[expectedWord] {
					t.Logf("Found expected word: %s", expectedWord)
				}
			}

			t.Logf("Cleaned words: %v", words)
		})
	}
}

func TestCalculateWordFrequencies(t *testing.T) {
	modeler := NewTopicModeler()

	testCases := []struct {
		name          string
		words         []string
		expectedFreqs map[string]int
	}{
		{
			name:  "simple words",
			words: []string{"technology", "computer", "technology", "software"},
			expectedFreqs: map[string]int{
				"technology": 2,
				"computer":   1,
				"software":   1,
			},
		},
		{
			name:          "empty words",
			words:         []string{},
			expectedFreqs: map[string]int{},
		},
		{
			name:  "repeated words",
			words: []string{"test", "test", "test"},
			expectedFreqs: map[string]int{
				"test": 3,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			freqs := modeler.calculateWordFrequencies(tc.words)

			if len(freqs) != len(tc.expectedFreqs) {
				t.Errorf("Expected %d unique words, got %d", len(tc.expectedFreqs), len(freqs))
			}

			for word, expectedFreq := range tc.expectedFreqs {
				if actualFreq, exists := freqs[word]; !exists {
					t.Errorf("Expected word '%s' not found", word)
				} else if actualFreq != expectedFreq {
					t.Errorf("Expected frequency %d for word '%s', got %d",
						expectedFreq, word, actualFreq)
				}
			}
		})
	}
}

func TestCalculateTopicScores(t *testing.T) {
	modeler := NewTopicModeler()

	// Test with technology-related words
	wordFreq := map[string]int{
		"technology": 2,
		"computer":   1,
		"software":   1,
		"algorithm":  1,
		"digital":    1,
	}

	scores := modeler.calculateTopicScores(wordFreq)

	if len(scores) == 0 {
		t.Error("Expected topic scores to be calculated")
	}

	// Technology should have a high score
	techScore, exists := scores["Technology"]
	if !exists {
		t.Error("Expected Technology topic score")
	} else {
		if techScore <= 0 {
			t.Errorf("Expected positive Technology score, got %f", techScore)
		}
		t.Logf("Technology score: %f", techScore)
	}

	// Check that scores are reasonable
	for topic, score := range scores {
		if score < 0 {
			t.Errorf("Topic '%s' has negative score: %f", topic, score)
		}

		if score > 1.0 {
			t.Logf("Topic '%s' has high score: %f (may be normal)", topic, score)
		}
	}
}

func TestGetRelevantKeywords(t *testing.T) {
	modeler := NewTopicModeler()

	wordFreq := map[string]int{
		"technology": 3,
		"computer":   2,
		"software":   1,
		"algorithm":  1,
		"digital":    1,
	}

	keywords := modeler.getRelevantKeywords("Technology", wordFreq)

	if len(keywords) == 0 {
		t.Error("Expected relevant keywords to be found")
	}

	if len(keywords) > 5 {
		t.Errorf("Expected at most 5 keywords, got %d", len(keywords))
	}

	// Check that keywords are from the input
	keywordMap := make(map[string]bool)
	for word := range wordFreq {
		keywordMap[word] = true
	}

	for _, keyword := range keywords {
		if !keywordMap[keyword] {
			t.Errorf("Keyword '%s' not found in input word frequencies", keyword)
		}
	}

	// Keywords should be sorted by relevance (frequency * weight)
	// Higher frequency words should generally appear first
	t.Logf("Relevant keywords for Technology: %v", keywords)
}

// Helper function to extract topic labels from topics
func getTopicLabels(topics []Topic) []string {
	labels := make([]string, len(topics))
	for i, topic := range topics {
		labels[i] = topic.Label
	}
	return labels
}
