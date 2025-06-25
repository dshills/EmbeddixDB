package ai

import (
	"context"
	"regexp"
	"strings"
	"unicode"
)

// LanguageDetector detects the language of text content
type LanguageDetector struct {
	languagePatterns map[string]*LanguagePattern
	stopWords        map[string][]string
}

// LanguagePattern contains patterns and characteristics for language detection
type LanguagePattern struct {
	Code             string
	Name             string
	CommonWords      []string
	CharacterPattern *regexp.Regexp
	ScriptRanges     []UnicodeRange
	TypicalRatio     float64 // Expected ratio of common words
}

// UnicodeRange represents a Unicode character range for script detection
type UnicodeRange struct {
	Start rune
	End   rune
	Name  string
}

// NewLanguageDetector creates a new language detector with predefined patterns
func NewLanguageDetector() *LanguageDetector {
	detector := &LanguageDetector{
		languagePatterns: make(map[string]*LanguagePattern),
		stopWords:        make(map[string][]string),
	}

	detector.initializeLanguagePatterns()
	return detector
}

// DetectLanguage identifies the language of the given content
func (ld *LanguageDetector) DetectLanguage(ctx context.Context, content string) (LanguageInfo, error) {
	if content == "" {
		return LanguageInfo{
			Code:       "unknown",
			Name:       "Unknown",
			Confidence: 0.0,
		}, nil
	}

	// Normalize content for analysis
	normalizedContent := ld.normalizeText(content)
	words := strings.Fields(normalizedContent)

	if len(words) == 0 {
		return LanguageInfo{
			Code:       "unknown",
			Name:       "Unknown",
			Confidence: 0.0,
		}, nil
	}

	// Calculate scores for each language
	scores := make(map[string]float64)

	for langCode, pattern := range ld.languagePatterns {
		score := ld.calculateLanguageScore(normalizedContent, words, pattern)
		scores[langCode] = score
	}

	// Find the language with the highest score
	bestLang := "en" // Default to English
	bestScore := 0.0

	for langCode, score := range scores {
		if score > bestScore {
			bestScore = score
			bestLang = langCode
		}
	}

	// Convert confidence to 0-1 range
	confidence := bestScore
	if confidence > 1.0 {
		confidence = 1.0
	}

	pattern := ld.languagePatterns[bestLang]
	return LanguageInfo{
		Code:       bestLang,
		Name:       pattern.Name,
		Confidence: confidence,
	}, nil
}

// calculateLanguageScore calculates a score for how likely the content is in a specific language
func (ld *LanguageDetector) calculateLanguageScore(content string, words []string, pattern *LanguagePattern) float64 {
	if len(words) == 0 {
		return 0.0
	}

	score := 0.0

	// Score based on common words
	commonWordMatches := 0
	contentLower := strings.ToLower(content)

	for _, commonWord := range pattern.CommonWords {
		if strings.Contains(contentLower, strings.ToLower(commonWord)) {
			commonWordMatches++
		}
	}

	if len(pattern.CommonWords) > 0 {
		commonWordScore := float64(commonWordMatches) / float64(len(pattern.CommonWords))
		score += commonWordScore * 0.6 // 60% weight for common words
	}

	// Score based on character patterns
	if pattern.CharacterPattern != nil {
		matches := pattern.CharacterPattern.FindAllString(content, -1)
		characterScore := float64(len(matches)) / float64(len(content))
		score += characterScore * 0.2 // 20% weight for character patterns
	}

	// Score based on Unicode script ranges
	scriptScore := ld.calculateScriptScore(content, pattern.ScriptRanges)
	score += scriptScore * 0.2 // 20% weight for script detection

	return score
}

// calculateScriptScore calculates score based on Unicode script ranges
func (ld *LanguageDetector) calculateScriptScore(content string, scriptRanges []UnicodeRange) float64 {
	if len(scriptRanges) == 0 {
		return 0.0
	}

	matchingChars := 0
	totalChars := 0

	for _, char := range content {
		if unicode.IsLetter(char) {
			totalChars++
			for _, scriptRange := range scriptRanges {
				if char >= scriptRange.Start && char <= scriptRange.End {
					matchingChars++
					break
				}
			}
		}
	}

	if totalChars == 0 {
		return 0.0
	}

	return float64(matchingChars) / float64(totalChars)
}

// normalizeText performs basic text normalization for language detection
func (ld *LanguageDetector) normalizeText(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove punctuation and numbers for word-based analysis
	re := regexp.MustCompile(`[^\p{L}\s]`)
	text = re.ReplaceAllString(text, " ")

	// Normalize whitespace
	re = regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// initializeLanguagePatterns sets up language detection patterns
func (ld *LanguageDetector) initializeLanguagePatterns() {
	// English
	ld.languagePatterns["en"] = &LanguagePattern{
		Code: "en",
		Name: "English",
		CommonWords: []string{
			"the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
			"it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
			"this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
			"or", "an", "will", "my", "one", "all", "would", "there", "their",
		},
		CharacterPattern: regexp.MustCompile(`[a-zA-Z]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'A', End: 'Z', Name: "Latin uppercase"},
			{Start: 'a', End: 'z', Name: "Latin lowercase"},
		},
		TypicalRatio: 0.4,
	}

	// Spanish
	ld.languagePatterns["es"] = &LanguagePattern{
		Code: "es",
		Name: "Spanish",
		CommonWords: []string{
			"de", "la", "que", "el", "en", "y", "a", "un", "ser", "se",
			"no", "te", "lo", "le", "da", "su", "por", "son", "con", "para",
			"al", "una", "del", "todo", "está", "muy", "fue", "han", "era",
		},
		CharacterPattern: regexp.MustCompile(`[a-zA-ZñáéíóúüÑÁÉÍÓÚÜ]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'A', End: 'Z', Name: "Latin uppercase"},
			{Start: 'a', End: 'z', Name: "Latin lowercase"},
			{Start: 'À', End: 'ÿ', Name: "Latin extended"},
		},
		TypicalRatio: 0.35,
	}

	// French
	ld.languagePatterns["fr"] = &LanguagePattern{
		Code: "fr",
		Name: "French",
		CommonWords: []string{
			"de", "le", "et", "à", "un", "il", "être", "et", "en", "avoir",
			"que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne",
			"se", "pas", "tout", "plus", "par", "grand", "quand", "même",
		},
		CharacterPattern: regexp.MustCompile(`[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'A', End: 'Z', Name: "Latin uppercase"},
			{Start: 'a', End: 'z', Name: "Latin lowercase"},
			{Start: 'À', End: 'ÿ', Name: "Latin extended"},
		},
		TypicalRatio: 0.35,
	}

	// German
	ld.languagePatterns["de"] = &LanguagePattern{
		Code: "de",
		Name: "German",
		CommonWords: []string{
			"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
			"des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine",
			"als", "auch", "es", "an", "werden", "aus", "er", "hat", "dass",
		},
		CharacterPattern: regexp.MustCompile(`[a-zA-ZäöüßÄÖÜ]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'A', End: 'Z', Name: "Latin uppercase"},
			{Start: 'a', End: 'z', Name: "Latin lowercase"},
			{Start: 'À', End: 'ÿ', Name: "Latin extended"},
		},
		TypicalRatio: 0.35,
	}

	// Portuguese
	ld.languagePatterns["pt"] = &LanguagePattern{
		Code: "pt",
		Name: "Portuguese",
		CommonWords: []string{
			"de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
			"é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais",
			"as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à",
		},
		CharacterPattern: regexp.MustCompile(`[a-zA-ZãáàâéêíóôõúçÃÁÀÂÉÊÍÓÔÕÚÇ]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'A', End: 'Z', Name: "Latin uppercase"},
			{Start: 'a', End: 'z', Name: "Latin lowercase"},
			{Start: 'À', End: 'ÿ', Name: "Latin extended"},
		},
		TypicalRatio: 0.35,
	}

	// Italian
	ld.languagePatterns["it"] = &LanguagePattern{
		Code: "it",
		Name: "Italian",
		CommonWords: []string{
			"di", "a", "da", "in", "con", "su", "per", "tra", "fra", "il",
			"lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "che",
			"non", "si", "è", "ci", "sono", "ha", "ho", "hai", "abbiamo",
		},
		CharacterPattern: regexp.MustCompile(`[a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'A', End: 'Z', Name: "Latin uppercase"},
			{Start: 'a', End: 'z', Name: "Latin lowercase"},
			{Start: 'À', End: 'ÿ', Name: "Latin extended"},
		},
		TypicalRatio: 0.35,
	}

	// Russian
	ld.languagePatterns["ru"] = &LanguagePattern{
		Code: "ru",
		Name: "Russian",
		CommonWords: []string{
			"в", "и", "не", "на", "я", "быть", "тот", "этот", "он", "с",
			"что", "а", "по", "весь", "это", "так", "же", "к", "у", "из",
			"за", "от", "мы", "вы", "они", "она", "оно", "мой", "как", "для",
		},
		CharacterPattern: regexp.MustCompile(`[а-яё]`),
		ScriptRanges: []UnicodeRange{
			{Start: 'А', End: 'я', Name: "Cyrillic"},
			{Start: 'ё', End: 'ё', Name: "Cyrillic yo"},
		},
		TypicalRatio: 0.4,
	}

	// Chinese (Simplified)
	ld.languagePatterns["zh"] = &LanguagePattern{
		Code: "zh",
		Name: "Chinese",
		CommonWords: []string{
			"的", "一", "是", "在", "不", "了", "有", "和", "人", "这",
			"中", "大", "为", "上", "个", "国", "我", "以", "要", "他",
			"时", "来", "用", "们", "生", "到", "作", "地", "于", "出",
		},
		CharacterPattern: regexp.MustCompile(`[\x{4e00}-\x{9fff}]`),
		ScriptRanges: []UnicodeRange{
			{Start: '\u4e00', End: '\u9fff', Name: "CJK Unified Ideographs"},
		},
		TypicalRatio: 0.3,
	}

	// Japanese
	ld.languagePatterns["ja"] = &LanguagePattern{
		Code: "ja",
		Name: "Japanese",
		CommonWords: []string{
			"の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
			"れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として",
		},
		CharacterPattern: regexp.MustCompile(`[\x{3040}-\x{309f}\x{30a0}-\x{30ff}\x{4e00}-\x{9fff}]`),
		ScriptRanges: []UnicodeRange{
			{Start: '\u3040', End: '\u309f', Name: "Hiragana"},
			{Start: '\u30a0', End: '\u30ff', Name: "Katakana"},
			{Start: '\u4e00', End: '\u9fff', Name: "CJK Unified Ideographs"},
		},
		TypicalRatio: 0.25,
	}

	// Arabic
	ld.languagePatterns["ar"] = &LanguagePattern{
		Code: "ar",
		Name: "Arabic",
		CommonWords: []string{
			"في", "من", "إلى", "على", "هذا", "هذه", "التي", "الذي", "كان",
			"أن", "قد", "لا", "ما", "أو", "كل", "بين", "عند", "غير", "بعد",
		},
		CharacterPattern: regexp.MustCompile(`[\x{0600}-\x{06ff}]`),
		ScriptRanges: []UnicodeRange{
			{Start: '\u0600', End: '\u06ff', Name: "Arabic"},
		},
		TypicalRatio: 0.3,
	}
}

// GetSupportedLanguages returns a list of all supported language codes
func (ld *LanguageDetector) GetSupportedLanguages() []string {
	var languages []string
	for code := range ld.languagePatterns {
		languages = append(languages, code)
	}
	return languages
}

// GetLanguageName returns the full name for a language code
func (ld *LanguageDetector) GetLanguageName(code string) string {
	if pattern, exists := ld.languagePatterns[code]; exists {
		return pattern.Name
	}
	return "Unknown"
}