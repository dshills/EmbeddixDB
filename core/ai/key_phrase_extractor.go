package ai

import (
	"context"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// KeyPhraseExtractor extracts important phrases from text content
type KeyPhraseExtractor struct {
	stopWords     map[string]bool
	minPhraseLen  int
	maxPhraseLen  int
	minWordLen    int
	posPatterns   []*regexp.Regexp
}

// PhraseScore represents a scored key phrase
type PhraseScore struct {
	Phrase     string
	Score      float64
	Frequency  int
	Words      []string
	StartPos   int
	EndPos     int
}

// NewKeyPhraseExtractor creates a new key phrase extractor
func NewKeyPhraseExtractor() *KeyPhraseExtractor {
	extractor := &KeyPhraseExtractor{
		stopWords:    make(map[string]bool),
		minPhraseLen: 1,
		maxPhraseLen: 4,
		minWordLen:   3,
		posPatterns:  make([]*regexp.Regexp, 0),
	}

	extractor.initializeStopWords()
	extractor.initializePOSPatterns()
	return extractor
}

// ExtractKeyPhrases extracts important phrases from the given content
func (kpe *KeyPhraseExtractor) ExtractKeyPhrases(ctx context.Context, content string) ([]string, error) {
	if content == "" {
		return []string{}, nil
	}

	// Normalize and preprocess content
	processedContent := kpe.preprocessText(content)
	
	// Extract candidate phrases
	candidates := kpe.extractCandidatePhrases(processedContent)
	
	// Score phrases using TF-IDF-like approach
	scoredPhrases := kpe.scorePhrases(candidates, processedContent)
	
	// Filter and rank phrases
	topPhrases := kpe.selectTopPhrases(scoredPhrases, 10)
	
	// Convert to string slice
	phrases := make([]string, len(topPhrases))
	for i, phrase := range topPhrases {
		phrases[i] = phrase.Phrase
	}

	return phrases, nil
}

// preprocessText performs text preprocessing for phrase extraction
func (kpe *KeyPhraseExtractor) preprocessText(text string) string {
	// Remove URLs
	urlRegex := regexp.MustCompile(`https?://[^\s]+`)
	text = urlRegex.ReplaceAllString(text, "")

	// Remove email addresses
	emailRegex := regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	text = emailRegex.ReplaceAllString(text, "")

	// Remove excessive punctuation but keep sentence boundaries
	text = regexp.MustCompile(`[^\w\s.,;:!?-]`).ReplaceAllString(text, " ")

	// Normalize whitespace
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// extractCandidatePhrases extracts potential key phrases
func (kpe *KeyPhraseExtractor) extractCandidatePhrases(content string) []PhraseScore {
	var candidates []PhraseScore
	
	// Split into sentences
	sentences := kpe.splitIntoSentences(content)
	globalOffset := 0

	for _, sentence := range sentences {
		// Extract phrases from each sentence
		sentenceCandidates := kpe.extractPhrasesFromSentence(sentence, globalOffset)
		candidates = append(candidates, sentenceCandidates...)
		globalOffset += len(sentence) + 1 // +1 for space/punctuation
	}

	return candidates
}

// extractPhrasesFromSentence extracts phrases from a single sentence
func (kpe *KeyPhraseExtractor) extractPhrasesFromSentence(sentence string, offset int) []PhraseScore {
	var phrases []PhraseScore
	words := strings.Fields(strings.ToLower(sentence))
	
	if len(words) == 0 {
		return phrases
	}

	// Extract n-grams (1 to maxPhraseLen words)
	for n := kpe.minPhraseLen; n <= kpe.maxPhraseLen && n <= len(words); n++ {
		for i := 0; i <= len(words)-n; i++ {
			phrase := words[i : i+n]
			
			// Filter out phrases with stop words or short words
			if kpe.isValidPhrase(phrase) {
				phraseText := strings.Join(phrase, " ")
				
				// Calculate approximate position in original text
				startPos := offset + kpe.estimateWordPosition(sentence, i)
				endPos := startPos + len(phraseText)
				
				phrases = append(phrases, PhraseScore{
					Phrase:    phraseText,
					Words:     phrase,
					Frequency: 1,
					StartPos:  startPos,
					EndPos:    endPos,
				})
			}
		}
	}

	return phrases
}

// isValidPhrase checks if a phrase is valid for extraction
func (kpe *KeyPhraseExtractor) isValidPhrase(words []string) bool {
	if len(words) == 0 {
		return false
	}

	// Check for stop words
	for _, word := range words {
		if kpe.stopWords[word] || len(word) < kpe.minWordLen {
			return false
		}
	}

	// Check if all words are alphabetic
	for _, word := range words {
		if !kpe.isAlphabetic(word) {
			return false
		}
	}

	// Avoid phrases that are all uppercase (likely acronyms) or all lowercase
	hasProperCase := false
	for _, word := range words {
		if len(word) > 0 && unicode.IsUpper(rune(word[0])) {
			hasProperCase = true
			break
		}
	}

	return hasProperCase || len(words) > 1
}

// isAlphabetic checks if a word contains only alphabetic characters
func (kpe *KeyPhraseExtractor) isAlphabetic(word string) bool {
	for _, r := range word {
		if !unicode.IsLetter(r) && r != '-' && r != '\'' {
			return false
		}
	}
	return len(word) > 0
}

// scorePhrases calculates scores for candidate phrases
func (kpe *KeyPhraseExtractor) scorePhrases(candidates []PhraseScore, content string) []PhraseScore {
	// Count phrase frequencies
	phraseFreq := make(map[string]int)
	phrasePositions := make(map[string][]int)
	
	for _, candidate := range candidates {
		phraseFreq[candidate.Phrase]++
		phrasePositions[candidate.Phrase] = append(phrasePositions[candidate.Phrase], candidate.StartPos)
	}

	// Calculate word frequencies for IDF calculation
	words := strings.Fields(strings.ToLower(content))
	wordFreq := make(map[string]int)
	for _, word := range words {
		wordFreq[word]++
	}

	// Score each unique phrase
	var scoredPhrases []PhraseScore
	processed := make(map[string]bool)

	for _, candidate := range candidates {
		if processed[candidate.Phrase] {
			continue
		}
		processed[candidate.Phrase] = true

		score := kpe.calculatePhraseScore(candidate.Phrase, phraseFreq, wordFreq, phrasePositions, len(words))
		
		scoredPhrases = append(scoredPhrases, PhraseScore{
			Phrase:    candidate.Phrase,
			Score:     score,
			Frequency: phraseFreq[candidate.Phrase],
			Words:     candidate.Words,
			StartPos:  phrasePositions[candidate.Phrase][0], // First occurrence
		})
	}

	return scoredPhrases
}

// calculatePhraseScore computes the score for a phrase
func (kpe *KeyPhraseExtractor) calculatePhraseScore(phrase string, phraseFreq map[string]int, wordFreq map[string]int, phrasePositions map[string][]int, totalWords int) float64 {
	words := strings.Fields(phrase)
	
	// Term Frequency component
	tf := float64(phraseFreq[phrase])
	
	// Inverse Document Frequency component (simplified)
	// Higher score for less common words
	idf := 0.0
	for _, word := range words {
		if freq, exists := wordFreq[word]; exists {
			idf += math.Log(float64(totalWords) / float64(freq))
		}
	}
	idf /= float64(len(words)) // Average IDF for phrase words

	// Length bonus (longer phrases are often more informative)
	lengthBonus := math.Log(float64(len(words)) + 1)

	// Position bonus (phrases appearing early in text are often more important)
	positionBonus := 0.0
	if positions, exists := phrasePositions[phrase]; exists && len(positions) > 0 {
		firstPos := float64(positions[0])
		positionBonus = 1.0 / (1.0 + firstPos/1000.0) // Diminishing bonus for later positions
	}

	// Capitalization bonus (proper nouns are often important)
	capBonus := 0.0
	for _, word := range words {
		if len(word) > 0 && unicode.IsUpper(rune(word[0])) {
			capBonus += 0.2
		}
	}

	// Combine components
	score := tf * idf * lengthBonus * (1.0 + positionBonus + capBonus)

	return score
}

// selectTopPhrases selects the top-scoring phrases
func (kpe *KeyPhraseExtractor) selectTopPhrases(scoredPhrases []PhraseScore, maxPhrases int) []PhraseScore {
	// Sort by score (descending)
	sort.Slice(scoredPhrases, func(i, j int) bool {
		return scoredPhrases[i].Score > scoredPhrases[j].Score
	})

	// Remove overlapping phrases (keep highest scoring)
	filtered := kpe.removeOverlappingPhrases(scoredPhrases)

	// Return top N phrases
	if len(filtered) > maxPhrases {
		filtered = filtered[:maxPhrases]
	}

	return filtered
}

// removeOverlappingPhrases removes phrases that significantly overlap
func (kpe *KeyPhraseExtractor) removeOverlappingPhrases(phrases []PhraseScore) []PhraseScore {
	var result []PhraseScore
	
	for _, phrase := range phrases {
		isOverlapping := false
		
		for _, existing := range result {
			if kpe.phrasesOverlap(phrase.Phrase, existing.Phrase) {
				isOverlapping = true
				break
			}
		}
		
		if !isOverlapping {
			result = append(result, phrase)
		}
	}
	
	return result
}

// phrasesOverlap checks if two phrases have significant word overlap
func (kpe *KeyPhraseExtractor) phrasesOverlap(phrase1, phrase2 string) bool {
	words1 := strings.Fields(phrase1)
	words2 := strings.Fields(phrase2)
	
	if len(words1) == 0 || len(words2) == 0 {
		return false
	}
	
	// Count common words
	wordSet1 := make(map[string]bool)
	for _, word := range words1 {
		wordSet1[word] = true
	}
	
	commonWords := 0
	for _, word := range words2 {
		if wordSet1[word] {
			commonWords++
		}
	}
	
	// Consider overlapping if more than 50% of words are common
	minLen := len(words1)
	if len(words2) < minLen {
		minLen = len(words2)
	}
	
	overlap := float64(commonWords) / float64(minLen)
	return overlap > 0.5
}

// splitIntoSentences splits text into sentences
func (kpe *KeyPhraseExtractor) splitIntoSentences(content string) []string {
	// Simple sentence splitting
	sentenceRegex := regexp.MustCompile(`[.!?]+\s+`)
	sentences := sentenceRegex.Split(content, -1)
	
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

// estimateWordPosition estimates the character position of a word in a sentence
func (kpe *KeyPhraseExtractor) estimateWordPosition(sentence string, wordIndex int) int {
	words := strings.Fields(sentence)
	if wordIndex >= len(words) {
		return len(sentence)
	}
	
	// Simple estimation based on average word length
	avgWordLen := len(sentence) / len(words)
	return wordIndex * avgWordLen
}

// initializeStopWords sets up common stop words
func (kpe *KeyPhraseExtractor) initializeStopWords() {
	stopWords := []string{
		"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
		"he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
		"will", "with", "the", "this", "but", "they", "have", "had", "what",
		"said", "each", "which", "she", "do", "how", "their", "if", "up", "out",
		"many", "then", "them", "these", "so", "some", "her", "would", "make",
		"like", "into", "him", "time", "two", "more", "very", "after", "words",
		"just", "where", "most", "get", "through", "back", "much", "before",
		"go", "good", "new", "write", "our", "me", "man", "too", "any", "day",
		"same", "right", "look", "think", "also", "around", "another", "came",
		"come", "work", "three", "must", "because", "does", "part", "even",
		"place", "well", "such", "here", "take", "why", "help", "put", "different",
		"away", "again", "off", "went", "old", "number", "great", "tell", "men",
		"say", "small", "every", "found", "still", "between", "name", "should",
		"home", "big", "give", "air", "line", "set", "own", "under", "read",
		"last", "never", "us", "left", "end", "along", "while", "might", "next",
		"sound", "below", "saw", "something", "thought", "both", "few", "those",
		"always", "show", "large", "often", "together", "asked", "house", "don",
		"world", "going", "want", "school", "important", "until", "form", "food",
		"keep", "children", "feet", "land", "side", "without", "boy", "once",
		"animal", "life", "enough", "took", "four", "head", "above", "kind",
		"began", "almost", "live", "page", "got", "earth", "need", "far", "hand",
		"high", "year", "mother", "light", "country", "father", "let", "night",
		"picture", "being", "study", "second", "book", "carry", "science", "eat",
		"room", "friend", "began", "idea", "fish", "mountain", "north", "once",
		"base", "hear", "horse", "cut", "sure", "watch", "color", "face", "wood",
		"main", "open", "seem", "together", "white", "begin", "got", "walk",
		"example", "ease", "paper", "group", "always", "music", "those", "both",
		"mark", "often", "letter", "until", "mile", "river", "car", "feet",
		"care", "second", "enough", "plain", "girl", "usual", "young", "ready",
		"above", "ever", "red", "list", "though", "feel", "talk", "bird", "soon",
		"body", "dog", "family", "direct", "pose", "leave", "song", "measure",
		"door", "product", "black", "short", "numeral", "class", "wind", "question",
		"happen", "complete", "ship", "area", "half", "rock", "order", "fire",
		"south", "problem", "piece", "told", "knew", "pass", "since", "top",
		"whole", "king", "space", "heard", "best", "hour", "better", "during",
		"hundred", "five", "remember", "step", "early", "hold", "west", "ground",
		"interest", "reach", "fast", "verb", "sing", "listen", "six", "table",
		"travel", "less", "morning", "ten", "simple", "several", "vowel", "toward",
		"war", "lay", "against", "pattern", "slow", "center", "love", "person",
		"money", "serve", "appear", "road", "map", "rain", "rule", "govern",
		"pull", "cold", "notice", "voice", "unit", "power", "town", "fine",
		"certain", "fly", "fall", "lead", "cry", "dark", "machine", "note",
		"wait", "plan", "figure", "star", "box", "noun", "field", "rest", "correct",
		"able", "pound", "done", "beauty", "drive", "stood", "contain", "front",
		"teach", "week", "final", "gave", "green", "oh", "quick", "develop",
		"ocean", "warm", "free", "minute", "strong", "special", "mind", "behind",
		"clear", "tail", "produce", "fact", "street", "inch", "multiply", "nothing",
		"course", "stay", "wheel", "full", "force", "blue", "object", "decide",
		"surface", "deep", "moon", "island", "foot", "system", "busy", "test",
		"record", "boat", "common", "gold", "possible", "plane", "stead", "dry",
		"wonder", "laugh", "thousands", "ago", "ran", "check", "game", "shape",
		"equate", "hot", "miss", "brought", "heat", "snow", "tire", "bring",
		"yes", "distant", "fill", "east", "paint", "language", "among", "grand",
		"ball", "yet", "wave", "drop", "heart", "am", "present", "heavy", "dance",
		"engine", "position", "arm", "wide", "sail", "material", "size", "vary",
		"settle", "speak", "weight", "general", "ice", "matter", "circle", "pair",
		"include", "divide", "syllable", "felt", "perhaps", "pick", "sudden",
		"count", "square", "reason", "length", "represent", "art", "subject",
		"region", "energy", "hunt", "probable", "bed", "brother", "egg", "ride",
		"cell", "believe", "fraction", "forest", "sit", "race", "window", "store",
		"summer", "train", "sleep", "prove", "lone", "leg", "exercise", "wall",
		"catch", "mount", "wish", "sky", "board", "joy", "winter", "sat", "written",
		"wild", "instrument", "kept", "glass", "grass", "cow", "job", "edge",
		"sign", "visit", "past", "soft", "fun", "bright", "gas", "weather",
		"month", "million", "bear", "finish", "happy", "hope", "flower", "clothe",
		"strange", "gone", "jump", "baby", "eight", "village", "meet", "root",
		"buy", "raise", "solve", "metal", "whether", "push", "seven", "paragraph",
		"third", "shall", "held", "hair", "describe", "cook", "floor", "either",
		"result", "burn", "hill", "safe", "cat", "century", "consider", "type",
		"law", "bit", "coast", "copy", "phrase", "silent", "tall", "sand", "soil",
		"roll", "temperature", "finger", "industry", "value", "fight", "lie",
		"beat", "excite", "natural", "view", "sense", "ear", "else", "quite",
		"broke", "case", "middle", "kill", "son", "lake", "moment", "scale",
		"loud", "spring", "observe", "child", "straight", "consonant", "nation",
		"dictionary", "milk", "speed", "method", "organ", "pay", "age", "section",
		"dress", "cloud", "surprise", "quiet", "stone", "tiny", "climb", "bad",
		"oil", "blood", "touch", "grew", "cent", "mix", "team", "wire", "cost",
		"lost", "brown", "wear", "garden", "equal", "sent", "choose", "fell",
		"fit", "flow", "fair", "bank", "collect", "save", "control", "decimal",
		"gentle", "woman", "captain", "practice", "separate", "difficult", "doctor",
		"please", "protect", "noon", "whose", "locate", "ring", "character",
		"insect", "caught", "period", "indicate", "radio", "spoke", "atom",
		"human", "history", "effect", "electric", "expect", "crop", "modern",
		"element", "hit", "student", "corner", "party", "supply", "bone", "rail",
		"imagine", "provide", "agree", "thus", "capital", "won", "chair", "danger",
		"fruit", "rich", "thick", "soldier", "process", "operate", "guess",
		"necessary", "sharp", "wing", "create", "neighbor", "wash", "bat", "rather",
		"crowd", "corn", "compare", "poem", "string", "bell", "depend", "meat",
		"rub", "tube", "famous", "dollar", "stream", "fear", "sight", "thin",
		"triangle", "planet", "hurry", "chief", "colony", "clock", "mine", "tie",
		"enter", "major", "fresh", "search", "send", "yellow", "gun", "allow",
		"print", "dead", "spot", "desert", "suit", "current", "lift", "rose",
		"continue", "block", "chart", "hat", "sell", "success", "company", "subtract",
		"event", "particular", "deal", "swim", "term", "opposite", "wife", "shoe",
		"shoulder", "spread", "arrange", "camp", "invent", "cotton", "born",
		"determine", "quart", "nine", "truck", "noise", "level", "chance", "gather",
		"shop", "stretch", "throw", "shine", "property", "column", "molecule",
		"select", "wrong", "gray", "repeat", "require", "broad", "prepare", "salt",
		"nose", "plural", "anger", "claim", "continent", "oxygen", "sugar", "death",
		"pretty", "skill", "women", "season", "solution", "magnet", "silver",
		"thank", "branch", "match", "suffix", "especially", "fig", "afraid",
		"huge", "sister", "steel", "discuss", "forward", "similar", "guide",
		"experience", "score", "apple", "bought", "led", "pitch", "coat", "mass",
		"card", "band", "rope", "slip", "win", "dream", "evening", "condition",
		"feed", "tool", "total", "basic", "smell", "valley", "nor", "double",
		"seat", "arrive", "master", "track", "parent", "shore", "division", "sheet",
		"substance", "favor", "connect", "post", "spend", "chord", "fat", "glad",
		"original", "share", "station", "dad", "bread", "charge", "proper", "bar",
		"offer", "segment", "slave", "duck", "instant", "market", "degree", "populate",
		"chick", "dear", "enemy", "reply", "drink", "occur", "support", "speech",
		"nature", "range", "steam", "motion", "path", "liquid", "log", "meant",
		"quotient", "teeth", "shell", "neck",
	}

	for _, word := range stopWords {
		kpe.stopWords[word] = true
	}
}

// initializePOSPatterns sets up part-of-speech patterns for phrase extraction
func (kpe *KeyPhraseExtractor) initializePOSPatterns() {
	// Simplified POS patterns for key phrase extraction
	// In a real implementation, this would use proper POS tagging
	patterns := []string{
		`\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b`,           // Proper nouns
		`\b[a-z]+(?:ing|ion|tion|sion|ness|ment)\b`,    // Gerunds, nouns with suffixes
		`\b(?:new|important|significant|major)\s+[a-z]+\b`, // Adjective + noun
	}

	for _, pattern := range patterns {
		if regex, err := regexp.Compile(pattern); err == nil {
			kpe.posPatterns = append(kpe.posPatterns, regex)
		}
	}
}

// GetExtractionStats returns statistics about the extraction process
func (kpe *KeyPhraseExtractor) GetExtractionStats() map[string]interface{} {
	return map[string]interface{}{
		"min_phrase_length": kpe.minPhraseLen,
		"max_phrase_length": kpe.maxPhraseLen,
		"min_word_length":   kpe.minWordLen,
		"stop_words_count":  len(kpe.stopWords),
		"pos_patterns_count": len(kpe.posPatterns),
	}
}