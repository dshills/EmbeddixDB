package feedback

import (
	"context"
	"time"
)

// Collector defines the interface for collecting user feedback
type Collector interface {
	// RecordInteraction records a user interaction with a search result
	RecordInteraction(ctx context.Context, interaction *Interaction) error

	// RecordQuery records a search query and its context
	RecordQuery(ctx context.Context, feedback *QueryFeedback) error

	// GetInteractions retrieves interactions for a specific query or user
	GetInteractions(ctx context.Context, filter InteractionFilter) ([]*Interaction, error)

	// GetQueryFeedback retrieves aggregated feedback for queries
	GetQueryFeedback(ctx context.Context, queryID string) (*QueryFeedback, error)

	// GetDocumentFeedback retrieves aggregated feedback for a document
	GetDocumentFeedback(ctx context.Context, documentID string) (*DocumentFeedback, error)

	// UpdateDocumentFeedback updates aggregated feedback for a document
	UpdateDocumentFeedback(ctx context.Context, feedback *DocumentFeedback) error
}

// SessionManager manages user search sessions
type SessionManager interface {
	// CreateSession creates a new search session
	CreateSession(ctx context.Context, userID string, metadata map[string]interface{}) (*Session, error)

	// GetSession retrieves a session by ID
	GetSession(ctx context.Context, sessionID string) (*Session, error)

	// UpdateSession updates session information
	UpdateSession(ctx context.Context, session *Session) error

	// EndSession marks a session as ended
	EndSession(ctx context.Context, sessionID string) error

	// GetActiveSessions retrieves active sessions for a user
	GetActiveSessions(ctx context.Context, userID string) ([]*Session, error)

	// GetSessionHistory retrieves session history for a user
	GetSessionHistory(ctx context.Context, userID string, limit int) ([]*Session, error)
}

// ProfileManager manages user profiles and preferences
type ProfileManager interface {
	// GetProfile retrieves a user profile
	GetProfile(ctx context.Context, userID string) (*UserProfile, error)

	// CreateProfile creates a new user profile
	CreateProfile(ctx context.Context, userID string) (*UserProfile, error)

	// UpdateProfile updates a user profile
	UpdateProfile(ctx context.Context, profile *UserProfile) error

	// UpdatePreferences updates user preferences
	UpdatePreferences(ctx context.Context, userID string, preferences UserPreferences) error

	// IncrementInterests increments interest scores based on interactions
	IncrementInterests(ctx context.Context, userID string, topics map[string]float64, entities map[string]float64) error

	// GetTopInterests retrieves top interests for a user
	GetTopInterests(ctx context.Context, userID string, limit int) (topics map[string]float64, entities map[string]float64, err error)
}

// LearningEngine processes feedback to improve search quality
type LearningEngine interface {
	// GenerateLearningSignals converts interactions to learning signals
	GenerateLearningSignals(ctx context.Context, interactions []*Interaction) ([]*LearningSignal, error)

	// UpdateClickModel updates the click model based on new data
	UpdateClickModel(ctx context.Context, interactions []*Interaction) error

	// GetClickProbability estimates click probability for a result position
	GetClickProbability(ctx context.Context, position int, features map[string]float64) (float64, error)

	// GetRelevanceScore computes learned relevance score
	GetRelevanceScore(ctx context.Context, query string, documentID string, features map[string]float64) (float64, error)

	// TrainModel trains the learning model with new signals
	TrainModel(ctx context.Context, signals []*LearningSignal) error

	// ExportModel exports the trained model
	ExportModel(ctx context.Context) ([]byte, error)

	// ImportModel imports a pre-trained model
	ImportModel(ctx context.Context, modelData []byte) error
}

// InteractionFilter defines filters for querying interactions
type InteractionFilter struct {
	UserID       string
	SessionID    string
	QueryID      string
	DocumentID   string
	CollectionID string
	Type         InteractionType
	StartTime    *time.Time
	EndTime      *time.Time
	Limit        int
}

// FeedbackAnalyzer analyzes feedback patterns
type FeedbackAnalyzer interface {
	// AnalyzeQuerySatisfaction determines if a query was satisfied
	AnalyzeQuerySatisfaction(ctx context.Context, interactions []*Interaction) (satisfied bool, confidence float64)

	// AnalyzeClickPattern analyzes click patterns for bias
	AnalyzeClickPattern(ctx context.Context, interactions []*Interaction) (pattern map[string]interface{}, err error)

	// ComputeDocumentQuality computes quality score based on feedback
	ComputeDocumentQuality(ctx context.Context, feedback *DocumentFeedback) (quality float64, signals map[string]float64)

	// DetectAnomalies detects anomalous feedback patterns
	DetectAnomalies(ctx context.Context, interactions []*Interaction) (anomalies []string, err error)
}

// MetricsCollector collects search quality metrics
type MetricsCollector interface {
	// RecordSearchMetrics records metrics for a search operation
	RecordSearchMetrics(ctx context.Context, metrics SearchMetrics) error

	// GetAggregatedMetrics retrieves aggregated metrics
	GetAggregatedMetrics(ctx context.Context, timeRange TimeRange) (*AggregatedMetrics, error)

	// GetUserMetrics retrieves metrics for a specific user
	GetUserMetrics(ctx context.Context, userID string, timeRange TimeRange) (*UserMetrics, error)
}

// SearchMetrics represents metrics for a search operation
type SearchMetrics struct {
	QueryID       string
	UserID        string
	SessionID     string
	SearchType    string // "vector", "text", "hybrid"
	ResultCount   int
	ResponseTime  time.Duration
	RetrievalTime time.Duration
	RerankingTime time.Duration
	TotalTime     time.Duration
	CacheHit      bool
	Timestamp     time.Time
}

// AggregatedMetrics represents aggregated search metrics
type AggregatedMetrics struct {
	TotalSearches   int64
	AvgResponseTime time.Duration
	AvgResultCount  float64
	CacheHitRate    float64
	SearchTypes     map[string]int64
	TopQueries      []string
	ErrorRate       float64
}

// UserMetrics represents user-specific metrics
type UserMetrics struct {
	UserID           string
	TotalSearches    int64
	AvgResponseTime  time.Duration
	SatisfactionRate float64
	TopTopics        []string
	TopEntities      []string
	ActiveSessions   int
}

// TimeRange represents a time range for queries
type TimeRange struct {
	Start time.Time
	End   time.Time
}
