package ai

import (
	"regexp"
	"strings"
	"unicode"
)

// ContentProcessor handles text preprocessing and chunking
type ContentProcessor struct {
	// Configuration for processing
	MaxChunkSize       int
	ChunkOverlap       int
	SentenceSplit      bool
	PreserveBoundaries bool
}

// NewContentProcessor creates a new content processor with default settings
func NewContentProcessor() *ContentProcessor {
	return &ContentProcessor{
		MaxChunkSize:       1000, // Default chunk size in characters
		ChunkOverlap:       100,  // Default overlap in characters
		SentenceSplit:      true, // Split on sentence boundaries
		PreserveBoundaries: true, // Try to preserve word/sentence boundaries
	}
}

// ProcessDocument processes a single document into chunks
func (cp *ContentProcessor) ProcessDocument(content string, contentType string) []string {
	// Normalize content based on type
	normalized := cp.normalizeContent(content, contentType)

	// If content is small enough, return as single chunk
	if len(normalized) <= cp.MaxChunkSize {
		return []string{normalized}
	}

	// Split into chunks
	if cp.SentenceSplit {
		return cp.chunkBySentences(normalized)
	}

	return cp.chunkByCharacters(normalized)
}

// ProcessDocuments processes multiple documents into chunks
func (cp *ContentProcessor) ProcessDocuments(documents []string, contentTypes []string) ([]string, []ChunkMetadata) {
	var allChunks []string
	var allMetadata []ChunkMetadata

	for i, content := range documents {
		contentType := ""
		if i < len(contentTypes) {
			contentType = contentTypes[i]
		}

		chunks := cp.ProcessDocument(content, contentType)

		for j, chunk := range chunks {
			allChunks = append(allChunks, chunk)

			metadata := ChunkMetadata{
				SourceIndex:    i,
				ChunkIndex:     j,
				TotalChunks:    len(chunks),
				ContentType:    contentType,
				CharacterStart: cp.findCharacterStart(content, chunk),
				CharacterEnd:   cp.findCharacterEnd(content, chunk),
				WordCount:      cp.countWords(chunk),
			}

			allMetadata = append(allMetadata, metadata)
		}
	}

	return allChunks, allMetadata
}

// ChunkMetadata contains information about a text chunk
type ChunkMetadata struct {
	SourceIndex    int    `json:"source_index"`
	ChunkIndex     int    `json:"chunk_index"`
	TotalChunks    int    `json:"total_chunks"`
	ContentType    string `json:"content_type"`
	CharacterStart int    `json:"character_start"`
	CharacterEnd   int    `json:"character_end"`
	WordCount      int    `json:"word_count"`
}

// normalizeContent cleans and normalizes text based on content type
func (cp *ContentProcessor) normalizeContent(content string, contentType string) string {
	switch strings.ToLower(contentType) {
	case "html":
		return cp.stripHTML(content)
	case "markdown":
		return cp.processMarkdown(content)
	default:
		return cp.normalizeText(content)
	}
}

// stripHTML removes HTML tags and entities
func (cp *ContentProcessor) stripHTML(content string) string {
	// Remove HTML tags
	re := regexp.MustCompile(`<[^>]*>`)
	content = re.ReplaceAllString(content, " ")

	// Replace common HTML entities
	replacements := map[string]string{
		"&nbsp;": " ",
		"&amp;":  "&",
		"&lt;":   "<",
		"&gt;":   ">",
		"&quot;": "\"",
		"&#39;":  "'",
	}

	for entity, replacement := range replacements {
		content = strings.ReplaceAll(content, entity, replacement)
	}

	// Normalize whitespace
	return cp.normalizeWhitespace(content)
}

// processMarkdown handles markdown-specific processing
func (cp *ContentProcessor) processMarkdown(content string) string {
	// Remove markdown headers but keep the text
	re := regexp.MustCompile(`^#+\s*`)
	lines := strings.Split(content, "\n")

	for i, line := range lines {
		lines[i] = re.ReplaceAllString(line, "")
	}

	content = strings.Join(lines, "\n")

	// Remove markdown links but keep the text
	linkRe := regexp.MustCompile(`\[([^\]]*)\]\([^)]*\)`)
	content = linkRe.ReplaceAllString(content, "$1")

	// Remove markdown emphasis
	emphasisRe := regexp.MustCompile(`\*\*([^*]*)\*\*|__([^_]*)__|[*_]([^*_]*)[*_]`)
	content = emphasisRe.ReplaceAllString(content, "$1$2$3")

	// Remove code blocks and inline code
	codeBlockRe := regexp.MustCompile("```[\\s\\S]*?```")
	content = codeBlockRe.ReplaceAllString(content, " ")

	inlineCodeRe := regexp.MustCompile("`[^`]*`")
	content = inlineCodeRe.ReplaceAllString(content, " ")

	return cp.normalizeWhitespace(content)
}

// normalizeText performs basic text normalization
func (cp *ContentProcessor) normalizeText(content string) string {
	return cp.normalizeWhitespace(content)
}

// normalizeWhitespace normalizes whitespace characters
func (cp *ContentProcessor) normalizeWhitespace(content string) string {
	// Replace multiple whitespace with single space
	re := regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	// Trim leading/trailing whitespace
	return strings.TrimSpace(content)
}

// chunkBySentences splits text by sentences while respecting chunk size limits
func (cp *ContentProcessor) chunkBySentences(content string) []string {
	sentences := cp.splitIntoSentences(content)

	var chunks []string
	var currentChunk strings.Builder
	currentSize := 0

	for _, sentence := range sentences {
		sentenceLen := len(sentence)

		// If adding this sentence would exceed the limit, finalize current chunk
		if currentSize > 0 && currentSize+sentenceLen > cp.MaxChunkSize {
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))

			// Start new chunk with overlap from previous chunk if configured
			currentChunk.Reset()
			currentSize = 0

			// Add overlap if configured
			if cp.ChunkOverlap > 0 && len(chunks) > 0 {
				overlapText := cp.getOverlapText(chunks[len(chunks)-1], cp.ChunkOverlap)
				currentChunk.WriteString(overlapText)
				currentSize = len(overlapText)

				if currentSize > 0 {
					currentChunk.WriteString(" ")
					currentSize++
				}
			}
		}

		// Add sentence to current chunk
		if currentSize > 0 {
			currentChunk.WriteString(" ")
			currentSize++
		}
		currentChunk.WriteString(sentence)
		currentSize += sentenceLen
	}

	// Add final chunk if it has content
	if currentSize > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	return chunks
}

// chunkByCharacters splits text by character count while trying to preserve word boundaries
func (cp *ContentProcessor) chunkByCharacters(content string) []string {
	if len(content) <= cp.MaxChunkSize {
		return []string{content}
	}

	var chunks []string
	pos := 0

	for pos < len(content) {
		end := pos + cp.MaxChunkSize
		if end > len(content) {
			end = len(content)
		}

		// Try to find a good break point (word boundary)
		if cp.PreserveBoundaries && end < len(content) {
			breakPoint := cp.findWordBoundary(content, end)
			if breakPoint > pos {
				end = breakPoint
			}
		}

		chunk := content[pos:end]
		chunks = append(chunks, strings.TrimSpace(chunk))

		// Move position accounting for overlap
		pos = end - cp.ChunkOverlap
		if pos <= 0 {
			pos = end
		}
	}

	return chunks
}

// splitIntoSentences splits text into sentences
func (cp *ContentProcessor) splitIntoSentences(text string) []string {
	// Simple sentence splitting on common punctuation
	re := regexp.MustCompile(`[.!?]+\s+`)
	sentences := re.Split(text, -1)

	// Filter out empty sentences and trim whitespace
	var result []string
	for _, sentence := range sentences {
		trimmed := strings.TrimSpace(sentence)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}

	return result
}

// findWordBoundary finds the nearest word boundary before the given position
func (cp *ContentProcessor) findWordBoundary(text string, pos int) int {
	if pos >= len(text) {
		return len(text)
	}

	// Look backwards for whitespace
	for i := pos - 1; i >= 0; i-- {
		if unicode.IsSpace(rune(text[i])) {
			return i
		}

		// Don't go back more than 10% of chunk size
		if pos-i > cp.MaxChunkSize/10 {
			break
		}
	}

	return pos
}

// getOverlapText gets overlap text from the end of a chunk
func (cp *ContentProcessor) getOverlapText(chunk string, overlapSize int) string {
	if len(chunk) <= overlapSize {
		return chunk
	}

	start := len(chunk) - overlapSize

	// Try to find a word boundary for cleaner overlap
	if cp.PreserveBoundaries {
		for i := start; i < len(chunk); i++ {
			if unicode.IsSpace(rune(chunk[i])) {
				start = i + 1
				break
			}
		}
	}

	return chunk[start:]
}

// findCharacterStart finds the character position where a chunk starts in the original text
func (cp *ContentProcessor) findCharacterStart(original, chunk string) int {
	// Simple implementation - in practice, this would need more sophisticated matching
	index := strings.Index(original, chunk)
	if index >= 0 {
		return index
	}
	return 0
}

// findCharacterEnd finds the character position where a chunk ends in the original text
func (cp *ContentProcessor) findCharacterEnd(original, chunk string) int {
	start := cp.findCharacterStart(original, chunk)
	return start + len(chunk)
}

// countWords counts the number of words in text
func (cp *ContentProcessor) countWords(text string) int {
	return len(strings.Fields(text))
}

// DetectContentType attempts to detect the content type of text
func (cp *ContentProcessor) DetectContentType(content string) string {
	content = strings.TrimSpace(content)

	// Check for HTML
	if strings.Contains(content, "<html") || strings.Contains(content, "<!DOCTYPE") {
		return "html"
	}

	// Check for Markdown
	if strings.Contains(content, "# ") || strings.Contains(content, "## ") ||
		strings.Contains(content, "```") || strings.Contains(content, "[](") {
		return "markdown"
	}

	// Check for JSON
	if (strings.HasPrefix(content, "{") && strings.HasSuffix(content, "}")) ||
		(strings.HasPrefix(content, "[") && strings.HasSuffix(content, "]")) {
		return "json"
	}

	// Default to plain text
	return "text"
}

// EstimateProcessingTime estimates how long it will take to process content
func (cp *ContentProcessor) EstimateProcessingTime(contentLength int) int64 {
	// Rough estimate: 1ms per 1000 characters
	return int64(contentLength / 1000)
}
