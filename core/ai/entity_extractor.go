package ai

import (
	"context"
	"regexp"
	"strings"
	"unicode"
)

// EntityExtractor extracts named entities from text
type EntityExtractor struct {
	patterns map[string]*EntityPattern
}

// EntityPattern defines patterns for extracting specific entity types
type EntityPattern struct {
	Label       string
	Patterns    []*regexp.Regexp
	Validators  []EntityValidator
	Priority    int
	MinLength   int
	MaxLength   int
}

// EntityValidator is a function that validates if an extracted entity is valid
type EntityValidator func(text string) bool

// NewEntityExtractor creates a new entity extractor with predefined patterns
func NewEntityExtractor() *EntityExtractor {
	extractor := &EntityExtractor{
		patterns: make(map[string]*EntityPattern),
	}

	extractor.initializeEntityPatterns()
	return extractor
}

// ExtractEntities finds named entities in the given content
func (ee *EntityExtractor) ExtractEntities(ctx context.Context, content string) ([]Entity, error) {
	if content == "" {
		return []Entity{}, nil
	}

	var entities []Entity
	processed := make(map[string]bool) // To avoid duplicates

	// Extract entities for each pattern type
	for label, pattern := range ee.patterns {
		foundEntities := ee.extractEntitiesWithPattern(content, label, pattern)
		
		for _, entity := range foundEntities {
			// Create a unique key for deduplication
			key := strings.ToLower(entity.Text) + "|" + entity.Label
			if !processed[key] {
				entities = append(entities, entity)
				processed[key] = true
			}
		}
	}

	// Sort entities by position in text
	for i := 0; i < len(entities)-1; i++ {
		for j := i + 1; j < len(entities); j++ {
			if entities[i].StartPos > entities[j].StartPos {
				entities[i], entities[j] = entities[j], entities[i]
			}
		}
	}

	return entities, nil
}

// extractEntitiesWithPattern extracts entities using a specific pattern
func (ee *EntityExtractor) extractEntitiesWithPattern(content, label string, pattern *EntityPattern) []Entity {
	var entities []Entity

	for _, regex := range pattern.Patterns {
		matches := regex.FindAllStringSubmatch(content, -1)
		indices := regex.FindAllStringSubmatchIndex(content, -1)

		for i, match := range matches {
			if len(match) > 0 {
				text := strings.TrimSpace(match[0])
				
				// Check length constraints
				if len(text) < pattern.MinLength || (pattern.MaxLength > 0 && len(text) > pattern.MaxLength) {
					continue
				}

				// Validate using custom validators
				if !ee.validateEntity(text, pattern.Validators) {
					continue
				}

				// Calculate position
				startPos := indices[i][0]
				endPos := indices[i][1]

				// Calculate confidence based on pattern priority and text characteristics
				confidence := ee.calculateEntityConfidence(text, pattern)

				entity := Entity{
					Text:       text,
					Label:      label,
					Confidence: confidence,
					StartPos:   startPos,
					EndPos:     endPos,
				}

				entities = append(entities, entity)
			}
		}
	}

	return entities
}

// validateEntity checks if an entity passes all validators
func (ee *EntityExtractor) validateEntity(text string, validators []EntityValidator) bool {
	for _, validator := range validators {
		if !validator(text) {
			return false
		}
	}
	return true
}

// calculateEntityConfidence calculates confidence score for an entity
func (ee *EntityExtractor) calculateEntityConfidence(text string, pattern *EntityPattern) float64 {
	confidence := 0.6 // Base confidence

	// Increase confidence based on pattern priority
	confidence += float64(pattern.Priority) * 0.1

	// Increase confidence for capitalized text (likely proper nouns)
	if unicode.IsUpper(rune(text[0])) {
		confidence += 0.1
	}

	// Increase confidence for longer entities (more specific)
	if len(text) > 10 {
		confidence += 0.1
	}

	// Decrease confidence for very short entities
	if len(text) < 3 {
		confidence -= 0.2
	}

	// Ensure confidence is within valid range
	if confidence > 1.0 {
		confidence = 1.0
	} else if confidence < 0.0 {
		confidence = 0.0
	}

	return confidence
}

// initializeEntityPatterns sets up predefined entity extraction patterns
func (ee *EntityExtractor) initializeEntityPatterns() {
	// Person names
	ee.patterns["PERSON"] = &EntityPattern{
		Label: "PERSON",
		Patterns: []*regexp.Regexp{
			// Full names (First Last, First Middle Last)
			regexp.MustCompile(`\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b`),
			// Names with titles
			regexp.MustCompile(`\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?: [A-Z][a-z]+)?\b`),
		},
		Validators: []EntityValidator{
			ee.isValidPersonName,
		},
		Priority:  3,
		MinLength: 2,
		MaxLength: 50,
	}

	// Organizations
	ee.patterns["ORG"] = &EntityPattern{
		Label: "ORG",
		Patterns: []*regexp.Regexp{
			// Companies with suffixes
			regexp.MustCompile(`\b[A-Z][A-Za-z\s&]+(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Co\.?|Company|Corporation)\b`),
			// Universities
			regexp.MustCompile(`\b(?:University of [A-Z][a-z]+|[A-Z][a-z]+ University)\b`),
			// Common organization patterns
			regexp.MustCompile(`\b[A-Z][A-Za-z\s]+ (?:Institute|Foundation|Association|Society|Organization)\b`),
		},
		Validators: []EntityValidator{
			ee.isValidOrganization,
		},
		Priority:  2,
		MinLength: 3,
		MaxLength: 100,
	}

	// Locations
	ee.patterns["LOC"] = &EntityPattern{
		Label: "LOC",
		Patterns: []*regexp.Regexp{
			// Cities and states
			regexp.MustCompile(`\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2}\b`),
			// Countries and major cities
			regexp.MustCompile(`\b(?:United States|United Kingdom|New York|Los Angeles|Chicago|London|Paris|Tokyo|Beijing|Mumbai|São Paulo)\b`),
			// Geographic features
			regexp.MustCompile(`\b(?:Mount|Mt\.?|Lake|River|Bay|Ocean|Sea|Desert|Mountain)\s+[A-Z][a-z]+\b`),
		},
		Validators: []EntityValidator{
			ee.isValidLocation,
		},
		Priority:  2,
		MinLength: 2,
		MaxLength: 50,
	}

	// Dates
	ee.patterns["DATE"] = &EntityPattern{
		Label: "DATE",
		Patterns: []*regexp.Regexp{
			// MM/DD/YYYY or MM-DD-YYYY
			regexp.MustCompile(`\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b`),
			// Month Day, Year
			regexp.MustCompile(`\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b`),
			// Month Year (no day)
			regexp.MustCompile(`\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b`),
			// Day Month Year
			regexp.MustCompile(`\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b`),
			// Year only
			regexp.MustCompile(`\b(?:19|20)\d{2}\b`),
		},
		Validators: []EntityValidator{
			ee.isValidDate,
		},
		Priority:  1,
		MinLength: 4,
		MaxLength: 30,
	}

	// Email addresses
	ee.patterns["EMAIL"] = &EntityPattern{
		Label: "EMAIL",
		Patterns: []*regexp.Regexp{
			regexp.MustCompile(`\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b`),
		},
		Validators: []EntityValidator{
			ee.isValidEmail,
		},
		Priority:  3,
		MinLength: 5,
		MaxLength: 100,
	}

	// Phone numbers
	ee.patterns["PHONE"] = &EntityPattern{
		Label: "PHONE",
		Patterns: []*regexp.Regexp{
			// US phone numbers
			regexp.MustCompile(`\b\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b`),
			// International format
			regexp.MustCompile(`\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b`),
		},
		Validators: []EntityValidator{
			ee.isValidPhone,
		},
		Priority:  2,
		MinLength: 10,
		MaxLength: 20,
	}

	// URLs
	ee.patterns["URL"] = &EntityPattern{
		Label: "URL",
		Patterns: []*regexp.Regexp{
			regexp.MustCompile(`https?://[^\s<>"{}|\\^` + "`" + `\[\]]+`),
			regexp.MustCompile(`www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?`),
		},
		Validators: []EntityValidator{
			ee.isValidURL,
		},
		Priority:  3,
		MinLength: 7,
		MaxLength: 200,
	}

	// Money amounts
	ee.patterns["MONEY"] = &EntityPattern{
		Label: "MONEY",
		Patterns: []*regexp.Regexp{
			regexp.MustCompile(`\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b`),
			regexp.MustCompile(`\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP|yen|JPY)\b`),
		},
		Validators: []EntityValidator{
			ee.isValidMoney,
		},
		Priority:  2,
		MinLength: 2,
		MaxLength: 20,
	}
}

// Validator functions

func (ee *EntityExtractor) isValidPersonName(text string) bool {
	// Check if it's not a common word
	commonWords := []string{"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
	textLower := strings.ToLower(text)
	
	for _, word := range commonWords {
		if textLower == word {
			return false
		}
	}

	// Check if it has proper capitalization
	words := strings.Fields(text)
	for _, word := range words {
		if len(word) > 0 && !unicode.IsUpper(rune(word[0])) {
			return false
		}
	}

	return true
}

func (ee *EntityExtractor) isValidOrganization(text string) bool {
	// Check minimum length
	if len(text) < 3 {
		return false
	}

	// Check if it contains at least one capital letter
	hasCapital := false
	for _, char := range text {
		if unicode.IsUpper(char) {
			hasCapital = true
			break
		}
	}

	return hasCapital
}

func (ee *EntityExtractor) isValidLocation(text string) bool {
	// Check minimum length
	if len(text) < 2 {
		return false
	}

	// Check if it starts with a capital letter
	if len(text) > 0 && !unicode.IsUpper(rune(text[0])) {
		return false
	}

	return true
}

func (ee *EntityExtractor) isValidDate(text string) bool {
	// Basic validation - ensure it's not just a number
	return len(text) >= 4 && (strings.Contains(text, "/") || strings.Contains(text, "-") || strings.Contains(text, " "))
}

func (ee *EntityExtractor) isValidEmail(text string) bool {
	// Basic email validation
	return strings.Contains(text, "@") && strings.Contains(text, ".")
}

func (ee *EntityExtractor) isValidPhone(text string) bool {
	// Count digits
	digitCount := 0
	for _, char := range text {
		if unicode.IsDigit(char) {
			digitCount++
		}
	}
	
	// Should have at least 10 digits
	return digitCount >= 10
}

func (ee *EntityExtractor) isValidURL(text string) bool {
	// Basic URL validation
	return strings.Contains(text, ".") && (strings.HasPrefix(text, "http") || strings.HasPrefix(text, "www"))
}

func (ee *EntityExtractor) isValidMoney(text string) bool {
	// Should contain digits
	hasDigit := false
	for _, char := range text {
		if unicode.IsDigit(char) {
			hasDigit = true
			break
		}
	}
	
	return hasDigit
}

// GetSupportedEntityTypes returns a list of all supported entity types
func (ee *EntityExtractor) GetSupportedEntityTypes() []string {
	var types []string
	for label := range ee.patterns {
		types = append(types, label)
	}
	return types
}