package ai

import (
	"context"
	"regexp"
	"strings"
)

// SentimentAnalyzer performs sentiment analysis on text content
type SentimentAnalyzer struct {
	positiveWords map[string]float64
	negativeWords map[string]float64
	intensifiers  map[string]float64
	negators      map[string]bool
}

// NewSentimentAnalyzer creates a new sentiment analyzer with predefined lexicons
func NewSentimentAnalyzer() *SentimentAnalyzer {
	analyzer := &SentimentAnalyzer{
		positiveWords: make(map[string]float64),
		negativeWords: make(map[string]float64),
		intensifiers:  make(map[string]float64),
		negators:      make(map[string]bool),
	}

	analyzer.initializeLexicons()
	return analyzer
}

// AnalyzeSentiment analyzes the sentiment of the given content
func (sa *SentimentAnalyzer) AnalyzeSentiment(ctx context.Context, content string) (SentimentScore, error) {
	if content == "" {
		return SentimentScore{
			Polarity:   0.0,
			Confidence: 0.0,
			Label:      "neutral",
		}, nil
	}

	// Normalize text
	normalizedText := sa.normalizeText(content)
	words := strings.Fields(normalizedText)

	if len(words) == 0 {
		return SentimentScore{
			Polarity:   0.0,
			Confidence: 0.0,
			Label:      "neutral",
		}, nil
	}

	// Calculate sentiment score
	score := sa.calculateSentimentScore(words)

	// Determine label and confidence
	label := sa.getSentimentLabel(score)
	confidence := sa.calculateConfidence(score, len(words))

	return SentimentScore{
		Polarity:   score,
		Confidence: confidence,
		Label:      label,
	}, nil
}

// calculateSentimentScore computes the overall sentiment score
func (sa *SentimentAnalyzer) calculateSentimentScore(words []string) float64 {
	var totalScore float64
	var wordCount int

	for i, word := range words {
		wordLower := strings.ToLower(word)
		
		// Check for negation in previous words
		isNegated := sa.isNegated(words, i)
		
		// Check for intensification
		intensityMultiplier := sa.getIntensityMultiplier(words, i)

		// Get word sentiment
		var wordScore float64
		if score, exists := sa.positiveWords[wordLower]; exists {
			wordScore = score
		} else if score, exists := sa.negativeWords[wordLower]; exists {
			wordScore = -score
		} else {
			continue // Skip neutral words
		}

		// Apply negation
		if isNegated {
			wordScore = -wordScore * 0.8 // Negation reduces intensity slightly
		}

		// Apply intensification
		wordScore *= intensityMultiplier

		totalScore += wordScore
		wordCount++
	}

	if wordCount == 0 {
		return 0.0
	}

	// Normalize by word count and clamp to [-1, 1]
	normalizedScore := totalScore / float64(wordCount)
	if normalizedScore > 1.0 {
		normalizedScore = 1.0
	} else if normalizedScore < -1.0 {
		normalizedScore = -1.0
	}

	return normalizedScore
}

// isNegated checks if a word is negated by preceding words
func (sa *SentimentAnalyzer) isNegated(words []string, currentIndex int) bool {
	// Look at the previous 3 words for negation
	start := currentIndex - 3
	if start < 0 {
		start = 0
	}

	for i := start; i < currentIndex; i++ {
		wordLower := strings.ToLower(words[i])
		if sa.negators[wordLower] {
			return true
		}
	}

	return false
}

// getIntensityMultiplier checks for intensifying words
func (sa *SentimentAnalyzer) getIntensityMultiplier(words []string, currentIndex int) float64 {
	multiplier := 1.0

	// Look at the previous 2 words for intensifiers
	start := currentIndex - 2
	if start < 0 {
		start = 0
	}

	for i := start; i < currentIndex; i++ {
		wordLower := strings.ToLower(words[i])
		if intensity, exists := sa.intensifiers[wordLower]; exists {
			multiplier *= intensity
		}
	}

	return multiplier
}

// getSentimentLabel determines the sentiment label based on score
func (sa *SentimentAnalyzer) getSentimentLabel(score float64) string {
	if score > 0.1 {
		return "positive"
	} else if score < -0.1 {
		return "negative"
	}
	return "neutral"
}

// calculateConfidence estimates confidence based on score magnitude and text length
func (sa *SentimentAnalyzer) calculateConfidence(score float64, wordCount int) float64 {
	// Base confidence from score magnitude
	confidence := absFloat64(score)

	// Increase confidence for longer texts (more evidence)
	if wordCount > 10 {
		confidence *= 1.2
	} else if wordCount < 5 {
		confidence *= 0.8
	}

	// Ensure confidence is within [0, 1]
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// normalizeText performs basic text normalization
func (sa *SentimentAnalyzer) normalizeText(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove URLs
	urlRegex := regexp.MustCompile(`https?://[^\s]+`)
	text = urlRegex.ReplaceAllString(text, "")

	// Remove email addresses
	emailRegex := regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	text = emailRegex.ReplaceAllString(text, "")

	// Remove excessive punctuation
	punctRegex := regexp.MustCompile(`[^\w\s]`)
	text = punctRegex.ReplaceAllString(text, " ")

	// Normalize whitespace
	spaceRegex := regexp.MustCompile(`\s+`)
	text = spaceRegex.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// initializeLexicons sets up sentiment word lists
func (sa *SentimentAnalyzer) initializeLexicons() {
	// Positive words with weights
	positiveWords := map[string]float64{
		"amazing":     0.9,
		"awesome":     0.8,
		"excellent":   0.9,
		"fantastic":   0.8,
		"great":       0.7,
		"good":        0.6,
		"wonderful":   0.8,
		"perfect":     0.9,
		"outstanding": 0.9,
		"brilliant":   0.8,
		"superb":      0.8,
		"magnificent": 0.9,
		"exceptional": 0.9,
		"marvelous":   0.8,
		"terrific":    0.7,
		"fabulous":    0.8,
		"incredible":  0.8,
		"remarkable":  0.7,
		"impressive":  0.7,
		"beautiful":   0.7,
		"lovely":      0.6,
		"nice":        0.5,
		"pleasant":    0.6,
		"delightful":  0.7,
		"charming":    0.6,
		"enjoyable":   0.6,
		"happy":       0.7,
		"joyful":      0.8,
		"cheerful":    0.7,
		"excited":     0.7,
		"thrilled":    0.8,
		"pleased":     0.6,
		"satisfied":   0.6,
		"grateful":    0.7,
		"thankful":    0.7,
		"optimistic":  0.6,
		"positive":    0.6,
		"successful":  0.7,
		"effective":   0.6,
		"efficient":   0.6,
		"helpful":     0.6,
		"useful":      0.5,
		"valuable":    0.6,
		"beneficial":  0.6,
		"advantageous": 0.6,
		"favorable":   0.6,
		"promising":   0.6,
		"encouraging": 0.6,
		"inspiring":   0.7,
		"motivating":  0.6,
		"uplifting":   0.7,
		"refreshing":  0.6,
	}

	// Negative words with weights
	negativeWords := map[string]float64{
		"terrible":     0.9,
		"awful":        0.8,
		"horrible":     0.9,
		"dreadful":     0.8,
		"disgusting":   0.9,
		"appalling":    0.9,
		"atrocious":    0.9,
		"abysmal":      0.9,
		"deplorable":   0.8,
		"despicable":   0.8,
		"detestable":   0.8,
		"revolting":    0.8,
		"repulsive":    0.8,
		"offensive":    0.7,
		"bad":          0.6,
		"poor":         0.6,
		"disappointing": 0.7,
		"unsatisfactory": 0.7,
		"inadequate":   0.6,
		"insufficient": 0.6,
		"deficient":    0.6,
		"faulty":       0.6,
		"flawed":       0.6,
		"broken":       0.6,
		"damaged":      0.6,
		"ruined":       0.7,
		"destroyed":    0.8,
		"devastated":   0.8,
		"upset":        0.6,
		"angry":        0.7,
		"furious":      0.8,
		"outraged":     0.8,
		"irritated":    0.6,
		"annoyed":      0.5,
		"frustrated":   0.6,
		"disappointed": 0.6,
		"sad":          0.6,
		"depressed":    0.7,
		"miserable":    0.8,
		"unhappy":      0.6,
		"gloomy":       0.6,
		"pessimistic":  0.6,
		"negative":     0.6,
		"problematic":  0.6,
		"troublesome":  0.6,
		"difficult":    0.5,
		"challenging":  0.4,
		"complex":      0.3,
		"complicated":  0.4,
		"confusing":    0.5,
		"unclear":      0.4,
		"ambiguous":    0.4,
		"uncertain":    0.4,
		"doubtful":     0.5,
		"suspicious":   0.6,
		"questionable": 0.5,
		"unreliable":   0.6,
		"untrustworthy": 0.7,
	}

	// Intensifiers
	intensifiers := map[string]float64{
		"very":        1.5,
		"extremely":   2.0,
		"incredibly":  1.8,
		"absolutely":  1.7,
		"completely":  1.6,
		"totally":     1.6,
		"entirely":    1.5,
		"quite":       1.3,
		"rather":      1.2,
		"really":      1.4,
		"truly":       1.4,
		"deeply":      1.5,
		"highly":      1.4,
		"remarkably":  1.6,
		"exceptionally": 1.8,
		"extraordinarily": 1.9,
		"tremendously": 1.7,
		"immensely":   1.6,
		"enormously":  1.6,
		"vastly":      1.5,
		"significantly": 1.4,
		"considerably": 1.3,
		"substantially": 1.4,
		"moderately":  1.1,
		"somewhat":    1.1,
		"slightly":    0.8,
		"barely":      0.7,
		"hardly":      0.6,
		"scarcely":    0.6,
	}

	// Negators
	negators := []string{
		"not", "no", "never", "nothing", "nobody", "nowhere", "neither", "nor",
		"none", "cannot", "can't", "won't", "wouldn't", "shouldn't", "couldn't",
		"don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
		"haven't", "hasn't", "hadn't", "without", "lack", "lacking", "absent",
		"deny", "denies", "denied", "refuse", "refuses", "refused", "reject",
		"rejects", "rejected", "oppose", "opposes", "opposed", "against",
	}

	// Populate maps
	for word, weight := range positiveWords {
		sa.positiveWords[word] = weight
	}

	for word, weight := range negativeWords {
		sa.negativeWords[word] = weight
	}

	for word, weight := range intensifiers {
		sa.intensifiers[word] = weight
	}

	for _, word := range negators {
		sa.negators[word] = true
	}
}

// GetSupportedFeatures returns the features supported by this analyzer
func (sa *SentimentAnalyzer) GetSupportedFeatures() []string {
	return []string{
		"polarity_detection",
		"confidence_scoring",
		"negation_handling",
		"intensification_detection",
		"lexicon_based_analysis",
	}
}

// absFloat64 returns the absolute value of a float64
func absFloat64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}