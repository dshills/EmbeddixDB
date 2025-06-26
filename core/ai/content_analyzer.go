package ai

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// DefaultContentAnalyzer implements the ContentAnalyzer interface
type DefaultContentAnalyzer struct {
	languageDetector   *LanguageDetector
	entityExtractor    *EntityExtractor
	topicModeler       *TopicModeler
	sentimentAnalyzer  *SentimentAnalyzer
	keyPhraseExtractor *KeyPhraseExtractor
	mutex              sync.RWMutex
	stats              *AnalysisStats
}

// AnalysisStats tracks content analysis performance
type AnalysisStats struct {
	TotalAnalyses       int64         `json:"total_analyses"`
	SuccessfulAnalyses  int64         `json:"successful_analyses"`
	FailedAnalyses      int64         `json:"failed_analyses"`
	AverageLatency      time.Duration `json:"average_latency"`
	TotalProcessingTime time.Duration `json:"total_processing_time"`
	LanguageDetections  int64         `json:"language_detections"`
	EntityExtractions   int64         `json:"entity_extractions"`
	TopicAnalyses       int64         `json:"topic_analyses"`
	SentimentAnalyses   int64         `json:"sentiment_analyses"`
	mutex               sync.RWMutex
}

// NewDefaultContentAnalyzer creates a new content analyzer with default components
func NewDefaultContentAnalyzer() *DefaultContentAnalyzer {
	return &DefaultContentAnalyzer{
		languageDetector:   NewLanguageDetector(),
		entityExtractor:    NewEntityExtractor(),
		topicModeler:       NewTopicModeler(),
		sentimentAnalyzer:  NewSentimentAnalyzer(),
		keyPhraseExtractor: NewKeyPhraseExtractor(),
		stats:              &AnalysisStats{},
	}
}

// AnalyzeContent extracts insights from content
func (a *DefaultContentAnalyzer) AnalyzeContent(ctx context.Context, content string) (ContentInsights, error) {
	start := time.Now()
	defer func() {
		a.stats.recordAnalysis(time.Since(start))
	}()

	if content == "" {
		return ContentInsights{}, fmt.Errorf("content cannot be empty")
	}

	insights := ContentInsights{
		WordCount: len(strings.Fields(content)),
	}

	// Detect language
	language, err := a.DetectLanguage(ctx, content)
	if err != nil {
		a.stats.recordFailure()
		return insights, fmt.Errorf("language detection failed: %w", err)
	}
	insights.Language = language
	a.stats.recordLanguageDetection()

	// Extract entities
	entities, err := a.ExtractEntities(ctx, content)
	if err != nil {
		// Don't fail the entire analysis for entity extraction errors
		entities = []Entity{}
	} else {
		a.stats.recordEntityExtraction()
	}
	insights.Entities = entities

	// Analyze topics
	topics, err := a.topicModeler.ExtractTopics(ctx, content)
	if err != nil {
		topics = []Topic{}
	} else {
		a.stats.recordTopicAnalysis()
	}
	insights.Topics = topics

	// Analyze sentiment
	sentiment, err := a.sentimentAnalyzer.AnalyzeSentiment(ctx, content)
	if err != nil {
		sentiment = SentimentScore{
			Polarity:   0.0,
			Confidence: 0.0,
			Label:      "neutral",
		}
	} else {
		a.stats.recordSentimentAnalysis()
	}
	insights.Sentiment = sentiment

	// Extract key phrases
	keyPhrases, err := a.keyPhraseExtractor.ExtractKeyPhrases(ctx, content)
	if err != nil {
		keyPhrases = []string{}
	}
	insights.KeyPhrases = keyPhrases

	// Calculate complexity and readability scores
	insights.Complexity = a.calculateComplexityScore(content)
	insights.Readability = a.calculateReadabilityScore(content)

	// Generate summary
	insights.Summary = a.generateSummary(content, keyPhrases)

	a.stats.recordSuccess()
	return insights, nil
}

// AnalyzeBatch processes multiple content items
func (a *DefaultContentAnalyzer) AnalyzeBatch(ctx context.Context, content []string) ([]ContentInsights, error) {
	if len(content) == 0 {
		return []ContentInsights{}, nil
	}

	insights := make([]ContentInsights, len(content))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var lastError error

	// Process content items concurrently
	for i, text := range content {
		wg.Add(1)
		go func(index int, contentText string) {
			defer wg.Done()

			result, err := a.AnalyzeContent(ctx, contentText)
			if err != nil {
				mu.Lock()
				lastError = err
				mu.Unlock()
				return
			}

			mu.Lock()
			insights[index] = result
			mu.Unlock()
		}(i, text)
	}

	wg.Wait()

	return insights, lastError
}

// ExtractEntities finds named entities in content
func (a *DefaultContentAnalyzer) ExtractEntities(ctx context.Context, content string) ([]Entity, error) {
	return a.entityExtractor.ExtractEntities(ctx, content)
}

// DetectLanguage identifies content language
func (a *DefaultContentAnalyzer) DetectLanguage(ctx context.Context, content string) (LanguageInfo, error) {
	return a.languageDetector.DetectLanguage(ctx, content)
}

// GetStats returns analysis statistics
func (a *DefaultContentAnalyzer) GetStats() *AnalysisStats {
	a.stats.mutex.RLock()
	defer a.stats.mutex.RUnlock()

	return &AnalysisStats{
		TotalAnalyses:       a.stats.TotalAnalyses,
		SuccessfulAnalyses:  a.stats.SuccessfulAnalyses,
		FailedAnalyses:      a.stats.FailedAnalyses,
		AverageLatency:      a.stats.AverageLatency,
		TotalProcessingTime: a.stats.TotalProcessingTime,
		LanguageDetections:  a.stats.LanguageDetections,
		EntityExtractions:   a.stats.EntityExtractions,
		TopicAnalyses:       a.stats.TopicAnalyses,
		SentimentAnalyses:   a.stats.SentimentAnalyses,
	}
}

// Helper methods

func (a *DefaultContentAnalyzer) calculateComplexityScore(content string) float64 {
	words := strings.Fields(content)
	if len(words) == 0 {
		return 0.0
	}

	// Simple complexity based on average word length and sentence length
	totalWordLength := 0
	for _, word := range words {
		totalWordLength += len(word)
	}
	avgWordLength := float64(totalWordLength) / float64(len(words))

	sentences := a.splitIntoSentences(content)
	avgSentenceLength := float64(len(words)) / float64(len(sentences))

	// Normalized complexity score (0-1)
	complexity := (avgWordLength/10.0 + avgSentenceLength/25.0) / 2.0
	if complexity > 1.0 {
		complexity = 1.0
	}

	return complexity
}

func (a *DefaultContentAnalyzer) calculateReadabilityScore(content string) float64 {
	words := strings.Fields(content)
	sentences := a.splitIntoSentences(content)
	syllables := a.countSyllables(content)

	if len(sentences) == 0 || len(words) == 0 {
		return 0.0
	}

	// Flesch Reading Ease Score (simplified)
	avgSentenceLength := float64(len(words)) / float64(len(sentences))
	avgSyllablesPerWord := float64(syllables) / float64(len(words))

	// Flesch formula: 206.835 - (1.015 × ASL) - (84.6 × ASW)
	score := 206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord)

	// Normalize to 0-1 range
	normalizedScore := score / 100.0
	if normalizedScore < 0 {
		normalizedScore = 0
	} else if normalizedScore > 1 {
		normalizedScore = 1
	}

	return normalizedScore
}

func (a *DefaultContentAnalyzer) splitIntoSentences(content string) []string {
	// Simple sentence splitting
	re := regexp.MustCompile(`[.!?]+\s+`)
	sentences := re.Split(content, -1)

	var result []string
	for _, sentence := range sentences {
		trimmed := strings.TrimSpace(sentence)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}

	if len(result) == 0 {
		return []string{content}
	}

	return result
}

func (a *DefaultContentAnalyzer) countSyllables(content string) int {
	words := strings.Fields(strings.ToLower(content))
	totalSyllables := 0

	for _, word := range words {
		syllables := a.countWordSyllables(word)
		if syllables == 0 {
			syllables = 1 // Every word has at least one syllable
		}
		totalSyllables += syllables
	}

	return totalSyllables
}

func (a *DefaultContentAnalyzer) countWordSyllables(word string) int {
	word = strings.ToLower(word)
	word = regexp.MustCompile(`[^a-z]`).ReplaceAllString(word, "")

	if word == "" {
		return 0
	}

	// Simple syllable counting rules
	vowels := "aeiouy"
	syllableCount := 0
	previousWasVowel := false

	for _, char := range word {
		isVowel := strings.ContainsRune(vowels, char)
		if isVowel && !previousWasVowel {
			syllableCount++
		}
		previousWasVowel = isVowel
	}

	// Handle silent 'e'
	if strings.HasSuffix(word, "e") && syllableCount > 1 {
		syllableCount--
	}

	// Every word has at least one syllable
	if syllableCount == 0 {
		syllableCount = 1
	}

	return syllableCount
}

func (a *DefaultContentAnalyzer) generateSummary(content string, keyPhrases []string) string {
	sentences := a.splitIntoSentences(content)
	if len(sentences) <= 2 {
		return content
	}

	// Score sentences based on key phrase presence
	type scoredSentence struct {
		sentence string
		score    float64
	}

	var scored []scoredSentence
	for _, sentence := range sentences {
		score := 0.0
		sentenceLower := strings.ToLower(sentence)

		// Score based on key phrase presence
		for _, phrase := range keyPhrases {
			if strings.Contains(sentenceLower, strings.ToLower(phrase)) {
				score += 1.0
			}
		}

		// Bonus for sentence position (first and last sentences often important)
		if len(scored) == 0 || len(scored) == len(sentences)-1 {
			score += 0.5
		}

		scored = append(scored, scoredSentence{
			sentence: sentence,
			score:    score,
		})
	}

	// Sort by score and take top sentences
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Take top 2 sentences or 1/3 of total sentences, whichever is larger
	summaryLength := 2
	if len(sentences)/3 > summaryLength {
		summaryLength = len(sentences) / 3
	}

	if summaryLength > len(scored) {
		summaryLength = len(scored)
	}

	var summaryParts []string
	for i := 0; i < summaryLength; i++ {
		summaryParts = append(summaryParts, scored[i].sentence)
	}

	return strings.Join(summaryParts, " ")
}

// AnalysisStats methods

func (s *AnalysisStats) recordAnalysis(duration time.Duration) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.TotalAnalyses++
	s.TotalProcessingTime += duration

	// Update average latency
	if s.TotalAnalyses == 1 {
		s.AverageLatency = duration
	} else {
		s.AverageLatency = s.TotalProcessingTime / time.Duration(s.TotalAnalyses)
	}
}

func (s *AnalysisStats) recordSuccess() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.SuccessfulAnalyses++
}

func (s *AnalysisStats) recordFailure() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.FailedAnalyses++
}

func (s *AnalysisStats) recordLanguageDetection() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.LanguageDetections++
}

func (s *AnalysisStats) recordEntityExtraction() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.EntityExtractions++
}

func (s *AnalysisStats) recordTopicAnalysis() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.TopicAnalyses++
}

func (s *AnalysisStats) recordSentimentAnalysis() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.SentimentAnalyses++
}
