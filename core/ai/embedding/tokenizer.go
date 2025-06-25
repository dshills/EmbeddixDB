package embedding

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// Tokenizer handles text tokenization for embedding models
type Tokenizer struct {
	modelName    string
	vocab        map[string]int64
	reverseVocab map[int64]string
	maxLength    int
	padToken     int64
	unkToken     int64
	clsToken     int64
	sepToken     int64
}

// TokenizerConfig configures tokenizer behavior
type TokenizerConfig struct {
	ModelName   string `json:"model_name"`
	VocabPath   string `json:"vocab_path"`
	MaxLength   int    `json:"max_length"`
	DoLowerCase bool   `json:"do_lower_case"`
}

// NewTokenizer creates a new tokenizer for the specified model
func NewTokenizer(modelName string) (*Tokenizer, error) {
	tokenizer := &Tokenizer{
		modelName: modelName,
		maxLength: 512,
	}

	// Initialize with basic vocabulary for mock implementation
	err := tokenizer.initializeMockVocab()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize vocabulary: %w", err)
	}

	return tokenizer, nil
}

// initializeMockVocab creates a mock vocabulary for testing
func (t *Tokenizer) initializeMockVocab() error {
	// Create a basic vocabulary with common tokens
	basicVocab := []string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"the", "and", "of", "to", "a", "in", "is", "it", "you", "that",
		"he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
		"at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
		"but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
		"each", "which", "she", "do", "how", "their", "if", "will", "up", "other",
		"about", "out", "many", "then", "them", "these", "so", "some", "her", "would",
		"make", "like", "into", "him", "has", "two", "more", "very", "after", "words",
		"its", "just", "where", "most", "get", "through", "back", "much", "before",
		"go", "good", "new", "write", "our", "me", "man", "too", "any", "day",
		"same", "right", "look", "think", "also", "around", "another", "came", "come", "work",
	}

	t.vocab = make(map[string]int64)
	t.reverseVocab = make(map[int64]string)

	for i, token := range basicVocab {
		id := int64(i)
		t.vocab[token] = id
		t.reverseVocab[id] = token
	}

	// Set special token IDs
	t.padToken = t.vocab["[PAD]"]
	t.unkToken = t.vocab["[UNK]"]
	t.clsToken = t.vocab["[CLS]"]
	t.sepToken = t.vocab["[SEP]"]

	return nil
}

// TokenizeBatch tokenizes a batch of texts
func (t *Tokenizer) TokenizeBatch(texts []string, maxLength int) ([][]int64, error) {
	if maxLength <= 0 {
		maxLength = t.maxLength
	}

	tokens := make([][]int64, len(texts))
	for i, text := range texts {
		tokenized, err := t.Tokenize(text, maxLength)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize text %d: %w", i, err)
		}
		tokens[i] = tokenized
	}

	return tokens, nil
}

// Tokenize converts text to token IDs
func (t *Tokenizer) Tokenize(text string, maxLength int) ([]int64, error) {
	if maxLength <= 0 {
		maxLength = t.maxLength
	}

	// Basic preprocessing
	text = t.preprocess(text)

	// Tokenize into words (simplified)
	words := t.basicTokenize(text)

	// Convert words to token IDs
	var tokens []int64

	// Add [CLS] token at the beginning
	tokens = append(tokens, t.clsToken)

	// Convert words to IDs
	for _, word := range words {
		if len(tokens) >= maxLength-1 { // Reserve space for [SEP]
			break
		}

		// Handle subword tokenization (simplified)
		subwords := t.wordpieceTokenize(word)
		for _, subword := range subwords {
			if len(tokens) >= maxLength-1 {
				break
			}

			if id, exists := t.vocab[subword]; exists {
				tokens = append(tokens, id)
			} else {
				tokens = append(tokens, t.unkToken)
			}
		}
	}

	// Add [SEP] token at the end
	if len(tokens) < maxLength {
		tokens = append(tokens, t.sepToken)
	}

	// Pad to maxLength
	for len(tokens) < maxLength {
		tokens = append(tokens, t.padToken)
	}

	// Truncate if necessary
	if len(tokens) > maxLength {
		tokens = tokens[:maxLength]
		tokens[maxLength-1] = t.sepToken // Ensure [SEP] at the end
	}

	return tokens, nil
}

// preprocess performs basic text preprocessing
func (t *Tokenizer) preprocess(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove extra whitespace
	text = strings.TrimSpace(text)
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	return text
}

// basicTokenize performs basic word tokenization
func (t *Tokenizer) basicTokenize(text string) []string {
	// Split on whitespace and punctuation
	words := make([]string, 0)
	current := make([]rune, 0)

	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if len(current) > 0 {
				words = append(words, string(current))
				current = current[:0]
			}
			if unicode.IsPunct(r) {
				words = append(words, string(r))
			}
		} else {
			current = append(current, r)
		}
	}

	if len(current) > 0 {
		words = append(words, string(current))
	}

	return words
}

// wordpieceTokenize performs simplified wordpiece tokenization
func (t *Tokenizer) wordpieceTokenize(word string) []string {
	// Simplified implementation - just return the word
	// In a real implementation, this would perform subword tokenization
	return []string{word}
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(tokenIDs []int64) string {
	var tokens []string

	for _, id := range tokenIDs {
		if token, exists := t.reverseVocab[id]; exists {
			// Skip special tokens in output
			if token != "[PAD]" && token != "[CLS]" && token != "[SEP]" {
				tokens = append(tokens, token)
			}
		}
	}

	return strings.Join(tokens, " ")
}

// GetVocabSize returns the vocabulary size
func (t *Tokenizer) GetVocabSize() int {
	return len(t.vocab)
}

// GetSpecialTokens returns special token IDs
func (t *Tokenizer) GetSpecialTokens() map[string]int64 {
	return map[string]int64{
		"PAD": t.padToken,
		"UNK": t.unkToken,
		"CLS": t.clsToken,
		"SEP": t.sepToken,
	}
}

// Close releases tokenizer resources
func (t *Tokenizer) Close() error {
	t.vocab = nil
	t.reverseVocab = nil
	return nil
}

// TokenizeWithAttentionMask tokenizes text and returns attention mask
func (t *Tokenizer) TokenizeWithAttentionMask(text string, maxLength int) ([]int64, []int64, error) {
	tokens, err := t.Tokenize(text, maxLength)
	if err != nil {
		return nil, nil, err
	}

	// Create attention mask (1 for real tokens, 0 for padding)
	attentionMask := make([]int64, len(tokens))
	for i, token := range tokens {
		if token != t.padToken {
			attentionMask[i] = 1
		} else {
			attentionMask[i] = 0
		}
	}

	return tokens, attentionMask, nil
}

// BatchTokenizeWithAttentionMask tokenizes a batch of texts with attention masks
func (t *Tokenizer) BatchTokenizeWithAttentionMask(texts []string, maxLength int) ([][]int64, [][]int64, error) {
	if maxLength <= 0 {
		maxLength = t.maxLength
	}

	tokens := make([][]int64, len(texts))
	attentionMasks := make([][]int64, len(texts))

	for i, text := range texts {
		tokenized, mask, err := t.TokenizeWithAttentionMask(text, maxLength)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to tokenize text %d: %w", i, err)
		}
		tokens[i] = tokenized
		attentionMasks[i] = mask
	}

	return tokens, attentionMasks, nil
}

// GetModelName returns the tokenizer's model name
func (t *Tokenizer) GetModelName() string {
	return t.modelName
}

// SetMaxLength sets the maximum sequence length
func (t *Tokenizer) SetMaxLength(maxLength int) {
	t.maxLength = maxLength
}
