package feedback

import (
	"time"
)

// InteractionType represents the type of user interaction
type InteractionType string

const (
	InteractionTypeClick      InteractionType = "click"
	InteractionTypeDwell      InteractionType = "dwell"
	InteractionTypeRating     InteractionType = "rating"
	InteractionTypeBookmark   InteractionType = "bookmark"
	InteractionTypeShare      InteractionType = "share"
	InteractionTypeDownload   InteractionType = "download"
	InteractionTypeIgnore     InteractionType = "ignore"
	InteractionTypeNegative   InteractionType = "negative"
)

// Interaction represents a user interaction with a search result
type Interaction struct {
	ID           string          `json:"id"`
	UserID       string          `json:"user_id"`
	SessionID    string          `json:"session_id"`
	QueryID      string          `json:"query_id"`
	Query        string          `json:"query"`
	DocumentID   string          `json:"document_id"`
	CollectionID string          `json:"collection_id"`
	Type         InteractionType `json:"type"`
	Value        float64         `json:"value"` // Rating value, dwell time in seconds, etc.
	Position     int             `json:"position"` // Position in search results
	Timestamp    time.Time       `json:"timestamp"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// Session represents a user search session
type Session struct {
	ID         string    `json:"id"`
	UserID     string    `json:"user_id"`
	StartTime  time.Time `json:"start_time"`
	EndTime    *time.Time `json:"end_time,omitempty"`
	QueryCount int       `json:"query_count"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// UserProfile represents a user's search preferences and history
type UserProfile struct {
	ID              string                 `json:"id"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	SearchCount     int64                  `json:"search_count"`
	InteractionCount int64                 `json:"interaction_count"`
	Preferences     UserPreferences        `json:"preferences"`
	TopicInterests  map[string]float64     `json:"topic_interests"` // Topic -> Interest score
	EntityInterests map[string]float64     `json:"entity_interests"` // Entity -> Interest score
	TermFrequency   map[string]int         `json:"term_frequency"`   // Search term -> frequency
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// UserPreferences stores user-specific search preferences
type UserPreferences struct {
	PreferredLanguages []string           `json:"preferred_languages"`
	SearchRecency      float64            `json:"search_recency"`      // Weight for recent documents
	SearchDiversity    float64            `json:"search_diversity"`    // Weight for diverse results
	TopicWeights       map[string]float64 `json:"topic_weights"`       // Topic-specific weights
	SourceWeights      map[string]float64 `json:"source_weights"`      // Source-specific weights
	VectorWeight       float64            `json:"vector_weight"`       // Weight for vector search (0-1)
	TextWeight         float64            `json:"text_weight"`         // Weight for text search (0-1)
}

// QueryFeedback aggregates feedback for a specific query
type QueryFeedback struct {
	QueryID          string    `json:"query_id"`
	Query            string    `json:"query"`
	UserID           string    `json:"user_id"`
	SessionID        string    `json:"session_id"`
	Timestamp        time.Time `json:"timestamp"`
	ResultCount      int       `json:"result_count"`
	ClickCount       int       `json:"click_count"`
	AvgDwellTime     float64   `json:"avg_dwell_time"`
	AvgClickPosition float64   `json:"avg_click_position"`
	Satisfied        bool      `json:"satisfied"` // Based on interactions
}

// DocumentFeedback aggregates feedback for a specific document
type DocumentFeedback struct {
	DocumentID       string    `json:"document_id"`
	CollectionID     string    `json:"collection_id"`
	TotalViews       int64     `json:"total_views"`
	TotalClicks      int64     `json:"total_clicks"`
	TotalRatings     int64     `json:"total_ratings"`
	AvgRating        float64   `json:"avg_rating"`
	AvgDwellTime     float64   `json:"avg_dwell_time"`
	ClickThroughRate float64   `json:"click_through_rate"`
	LastUpdated      time.Time `json:"last_updated"`
}

// LearningSignal represents a signal for training re-ranking models
type LearningSignal struct {
	Query        string                 `json:"query"`
	DocumentID   string                 `json:"document_id"`
	Features     map[string]float64     `json:"features"`     // Feature vector for ML
	Label        float64                `json:"label"`        // Relevance label (0-1)
	Weight       float64                `json:"weight"`       // Importance weight
	Timestamp    time.Time              `json:"timestamp"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ClickModel represents click behavior patterns
type ClickModel struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        string                 `json:"type"` // "position_bias", "cascade", "dynamic_bayesian"
	Parameters  map[string]float64     `json:"parameters"`
	LastUpdated time.Time              `json:"last_updated"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}