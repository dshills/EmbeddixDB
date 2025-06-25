package ai

import (
	"context"
	"regexp"
	"sort"
	"strings"
)

// TopicModeler extracts topics from text content
type TopicModeler struct {
	topicKeywords map[string][]string
	stopWords     map[string]bool
}

// TopicKeywordMap represents topic categories and their associated keywords
type TopicKeywordMap struct {
	Topic    string
	Keywords []string
	Weight   float64
}

// NewTopicModeler creates a new topic modeler with predefined topic categories
func NewTopicModeler() *TopicModeler {
	modeler := &TopicModeler{
		topicKeywords: make(map[string][]string),
		stopWords:     make(map[string]bool),
	}

	modeler.initializeTopics()
	modeler.initializeStopWords()
	return modeler
}

// ExtractTopics identifies topics in the given content
func (tm *TopicModeler) ExtractTopics(ctx context.Context, content string) ([]Topic, error) {
	if content == "" {
		return []Topic{}, nil
	}

	// Normalize and tokenize content
	words := tm.tokenizeAndClean(content)
	if len(words) == 0 {
		return []Topic{}, nil
	}

	// Calculate word frequencies
	wordFreq := tm.calculateWordFrequencies(words)

	// Score topics based on keyword presence
	topicScores := tm.calculateTopicScores(wordFreq)

	// Convert scores to topics and sort by relevance
	var topics []Topic
	for topicName, score := range topicScores {
		if score > 0.1 { // Minimum threshold
			topic := Topic{
				ID:         strings.ToLower(strings.ReplaceAll(topicName, " ", "_")),
				Label:      topicName,
				Keywords:   tm.getRelevantKeywords(topicName, wordFreq),
				Confidence: score,
				Weight:     score,
			}
			topics = append(topics, topic)
		}
	}

	// Sort topics by confidence/weight
	sort.Slice(topics, func(i, j int) bool {
		return topics[i].Confidence > topics[j].Confidence
	})

	// Limit to top 5 topics
	if len(topics) > 5 {
		topics = topics[:5]
	}

	return topics, nil
}

// tokenizeAndClean normalizes and tokenizes content for topic analysis
func (tm *TopicModeler) tokenizeAndClean(content string) []string {
	// Convert to lowercase
	content = strings.ToLower(content)

	// Remove punctuation and special characters
	re := regexp.MustCompile(`[^\p{L}\s]`)
	content = re.ReplaceAllString(content, " ")

	// Split into words
	words := strings.Fields(content)

	// Filter out stop words and short words
	var cleanWords []string
	for _, word := range words {
		if len(word) >= 3 && !tm.stopWords[word] {
			cleanWords = append(cleanWords, word)
		}
	}

	return cleanWords
}

// calculateWordFrequencies counts word occurrences
func (tm *TopicModeler) calculateWordFrequencies(words []string) map[string]int {
	freq := make(map[string]int)
	for _, word := range words {
		freq[word]++
	}
	return freq
}

// calculateTopicScores calculates relevance scores for each topic
func (tm *TopicModeler) calculateTopicScores(wordFreq map[string]int) map[string]float64 {
	scores := make(map[string]float64)

	for topic, keywords := range tm.topicKeywords {
		score := 0.0
		matchedKeywords := 0

		for _, keyword := range keywords {
			if freq, exists := wordFreq[keyword]; exists {
				// TF-IDF-like scoring: frequency * inverse document frequency
				score += float64(freq) * tm.getKeywordWeight(keyword)
				matchedKeywords++
			}
		}

		// Normalize score by topic size and content length
		if matchedKeywords > 0 {
			totalWords := 0
			for _, freq := range wordFreq {
				totalWords += freq
			}

			if totalWords > 0 {
				score = score / float64(totalWords)
				// Boost score based on keyword coverage
				coverage := float64(matchedKeywords) / float64(len(keywords))
				score *= (1.0 + coverage)
			}
		}

		scores[topic] = score
	}

	return scores
}

// getKeywordWeight returns the importance weight of a keyword
func (tm *TopicModeler) getKeywordWeight(keyword string) float64 {
	// More specific keywords get higher weights
	if len(keyword) > 8 {
		return 2.0
	} else if len(keyword) > 5 {
		return 1.5
	}
	return 1.0
}

// getRelevantKeywords returns keywords from the topic that appear in the content
func (tm *TopicModeler) getRelevantKeywords(topicName string, wordFreq map[string]int) []string {
	var relevantKeywords []string
	keywords := tm.topicKeywords[topicName]

	type keywordScore struct {
		keyword string
		score   float64
	}

	var scored []keywordScore
	for _, keyword := range keywords {
		if freq, exists := wordFreq[keyword]; exists {
			score := float64(freq) * tm.getKeywordWeight(keyword)
			scored = append(scored, keywordScore{keyword, score})
		}
	}

	// Sort by score and take top keywords
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Return top 5 keywords
	limit := 5
	if len(scored) < limit {
		limit = len(scored)
	}

	for i := 0; i < limit; i++ {
		relevantKeywords = append(relevantKeywords, scored[i].keyword)
	}

	return relevantKeywords
}

// initializeTopics sets up predefined topic categories
func (tm *TopicModeler) initializeTopics() {
	tm.topicKeywords["Technology"] = []string{
		"technology", "computer", "software", "hardware", "internet", "digital",
		"algorithm", "programming", "code", "development", "artificial", "intelligence",
		"machine", "learning", "data", "analytics", "cloud", "computing", "cybersecurity",
		"blockchain", "cryptocurrency", "automation", "robotics", "innovation",
	}

	tm.topicKeywords["Business"] = []string{
		"business", "company", "market", "marketing", "sales", "revenue", "profit",
		"investment", "finance", "economy", "startup", "entrepreneur", "strategy",
		"management", "leadership", "corporate", "industry", "commercial", "trade",
		"customer", "client", "service", "product", "brand", "competition",
	}

	tm.topicKeywords["Science"] = []string{
		"science", "research", "study", "experiment", "hypothesis", "theory",
		"discovery", "innovation", "biology", "chemistry", "physics", "medicine",
		"health", "medical", "disease", "treatment", "pharmaceutical", "clinical",
		"laboratory", "scientific", "analysis", "methodology", "evidence",
	}

	tm.topicKeywords["Education"] = []string{
		"education", "school", "university", "college", "student", "teacher",
		"learning", "teaching", "academic", "curriculum", "degree", "graduation",
		"knowledge", "skill", "training", "course", "class", "lecture", "study",
		"educational", "pedagogy", "scholarship", "research", "thesis",
	}

	tm.topicKeywords["Politics"] = []string{
		"politics", "government", "political", "policy", "law", "legislation",
		"congress", "senate", "parliament", "election", "vote", "candidate",
		"democracy", "republican", "democrat", "conservative", "liberal",
		"administration", "president", "minister", "mayor", "governor",
	}

	tm.topicKeywords["Sports"] = []string{
		"sports", "game", "team", "player", "athlete", "competition", "tournament",
		"championship", "league", "football", "basketball", "baseball", "soccer",
		"tennis", "golf", "hockey", "swimming", "running", "racing", "olympics",
		"coach", "training", "fitness", "exercise", "match", "score",
	}

	tm.topicKeywords["Entertainment"] = []string{
		"entertainment", "movie", "film", "television", "music", "song", "album",
		"artist", "actor", "actress", "director", "producer", "concert", "show",
		"performance", "theater", "drama", "comedy", "celebrity", "hollywood",
		"streaming", "gaming", "video", "podcast", "media", "culture",
	}

	tm.topicKeywords["Health"] = []string{
		"health", "medical", "medicine", "doctor", "patient", "hospital", "clinic",
		"treatment", "therapy", "diagnosis", "disease", "illness", "wellness",
		"fitness", "nutrition", "diet", "exercise", "mental", "physical",
		"healthcare", "pharmaceutical", "surgery", "prevention", "symptoms",
	}

	tm.topicKeywords["Travel"] = []string{
		"travel", "trip", "vacation", "tourism", "destination", "hotel", "flight",
		"airport", "journey", "adventure", "explore", "visit", "country", "city",
		"culture", "experience", "guide", "tourist", "sightseeing", "restaurant",
		"accommodation", "booking", "itinerary", "passport", "visa",
	}

	tm.topicKeywords["Food"] = []string{
		"food", "recipe", "cooking", "kitchen", "restaurant", "chef", "meal",
		"dinner", "lunch", "breakfast", "ingredient", "flavor", "taste", "cuisine",
		"dish", "drink", "wine", "coffee", "nutrition", "diet", "healthy",
		"vegetarian", "vegan", "organic", "fresh", "delicious",
	}

	tm.topicKeywords["Environment"] = []string{
		"environment", "climate", "change", "global", "warming", "pollution",
		"sustainability", "renewable", "energy", "carbon", "emission", "green",
		"ecology", "conservation", "wildlife", "nature", "forest", "ocean",
		"recycling", "biodiversity", "ecosystem", "environmental", "clean",
	}

	tm.topicKeywords["Finance"] = []string{
		"finance", "money", "investment", "bank", "banking", "stock", "market",
		"trading", "investor", "portfolio", "fund", "asset", "liability", "debt",
		"credit", "loan", "mortgage", "insurance", "economy", "economic",
		"financial", "capital", "revenue", "profit", "budget", "tax",
	}
}

// initializeStopWords sets up common stop words to filter out
func (tm *TopicModeler) initializeStopWords() {
	stopWords := []string{
		"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
		"not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
		"by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
		"one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
		"about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
		"time", "no", "just", "him", "know", "take", "people", "into", "year",
		"your", "good", "some", "could", "them", "see", "other", "than", "then",
		"now", "look", "only", "come", "its", "over", "think", "also", "back",
		"after", "use", "two", "how", "our", "work", "first", "well", "way",
		"even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
	}

	for _, word := range stopWords {
		tm.stopWords[word] = true
	}
}

// GetTopicCategories returns all available topic categories
func (tm *TopicModeler) GetTopicCategories() []string {
	var categories []string
	for topic := range tm.topicKeywords {
		categories = append(categories, topic)
	}
	return categories
}

// GetTopicKeywords returns keywords for a specific topic
func (tm *TopicModeler) GetTopicKeywords(topic string) []string {
	if keywords, exists := tm.topicKeywords[topic]; exists {
		return keywords
	}
	return []string{}
}