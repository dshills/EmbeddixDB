package feedback

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CTRTracker tracks click-through rates and optimizes based on them
type CTRTracker interface {
	// RecordImpression records that a document was shown in search results
	RecordImpression(ctx context.Context, impression Impression) error
	
	// RecordClick records that a document was clicked
	RecordClick(ctx context.Context, click Click) error
	
	// GetDocumentCTR gets the CTR for a specific document
	GetDocumentCTR(ctx context.Context, documentID string) (CTRMetrics, error)
	
	// GetQueryCTR gets the CTR for a specific query
	GetQueryCTR(ctx context.Context, query string) (CTRMetrics, error)
	
	// GetPositionCTR gets the CTR by position
	GetPositionCTR(ctx context.Context, position int) (CTRMetrics, error)
	
	// GetOptimalRanking suggests optimal ranking based on CTR data
	GetOptimalRanking(ctx context.Context, documentIDs []string, query string) ([]string, error)
	
	// ExportMetrics exports CTR metrics for analysis
	ExportMetrics(ctx context.Context) (CTRReport, error)
}

// Impression represents a document shown in search results
type Impression struct {
	QueryID      string    `json:"query_id"`
	Query        string    `json:"query"`
	DocumentID   string    `json:"document_id"`
	Position     int       `json:"position"`
	UserID       string    `json:"user_id"`
	SessionID    string    `json:"session_id"`
	Timestamp    time.Time `json:"timestamp"`
	ResultCount  int       `json:"result_count"`
	SearchMode   string    `json:"search_mode"`
}

// Click represents a click on a search result
type Click struct {
	QueryID    string    `json:"query_id"`
	DocumentID string    `json:"document_id"`
	Position   int       `json:"position"`
	UserID     string    `json:"user_id"`
	SessionID  string    `json:"session_id"`
	Timestamp  time.Time `json:"timestamp"`
	DwellTime  float64   `json:"dwell_time"` // in seconds
}

// CTRMetrics contains CTR statistics
type CTRMetrics struct {
	Impressions      int64     `json:"impressions"`
	Clicks           int64     `json:"clicks"`
	CTR              float64   `json:"ctr"`
	AvgPosition      float64   `json:"avg_position"`
	AvgDwellTime     float64   `json:"avg_dwell_time"`
	LastUpdated      time.Time `json:"last_updated"`
	ConfidenceScore  float64   `json:"confidence_score"` // Based on sample size
}

// CTRReport contains comprehensive CTR analytics
type CTRReport struct {
	GeneratedAt      time.Time                    `json:"generated_at"`
	TotalImpressions int64                        `json:"total_impressions"`
	TotalClicks      int64                        `json:"total_clicks"`
	OverallCTR       float64                      `json:"overall_ctr"`
	TopDocuments     []DocumentCTR                `json:"top_documents"`
	TopQueries       []QueryCTR                   `json:"top_queries"`
	PositionCTR      map[int]CTRMetrics           `json:"position_ctr"`
	HourlyTrends     map[int]CTRMetrics           `json:"hourly_trends"`
	UserSegments     map[string]CTRMetrics        `json:"user_segments"`
}

// DocumentCTR represents CTR data for a document
type DocumentCTR struct {
	DocumentID string     `json:"document_id"`
	Title      string     `json:"title,omitempty"`
	CTRMetrics CTRMetrics `json:"metrics"`
}

// QueryCTR represents CTR data for a query
type QueryCTR struct {
	Query      string     `json:"query"`
	CTRMetrics CTRMetrics `json:"metrics"`
}

// memoryCTRTracker implements CTRTracker with in-memory storage
type memoryCTRTracker struct {
	mu sync.RWMutex
	
	// Metrics storage
	documentMetrics  map[string]*CTRMetrics
	queryMetrics     map[string]*CTRMetrics
	positionMetrics  map[int]*CTRMetrics
	
	// Raw data for detailed analysis
	impressions []Impression
	clicks      []Click
	
	// Indexes for fast lookup
	queryImpressions    map[string][]int // queryID -> impression indices
	documentImpressions map[string][]int // documentID -> impression indices
	
	// Configuration
	maxDataSize int
	decayFactor float64
}

// NewMemoryCTRTracker creates a new in-memory CTR tracker
func NewMemoryCTRTracker() CTRTracker {
	return &memoryCTRTracker{
		documentMetrics:     make(map[string]*CTRMetrics),
		queryMetrics:        make(map[string]*CTRMetrics),
		positionMetrics:     make(map[int]*CTRMetrics),
		impressions:         make([]Impression, 0),
		clicks:              make([]Click, 0),
		queryImpressions:    make(map[string][]int),
		documentImpressions: make(map[string][]int),
		maxDataSize:         100000,
		decayFactor:         0.95,
	}
}

func (t *memoryCTRTracker) RecordImpression(ctx context.Context, impression Impression) error {
	if impression.Timestamp.IsZero() {
		impression.Timestamp = time.Now()
	}
	
	t.mu.Lock()
	defer t.mu.Unlock()
	
	// Add impression
	idx := len(t.impressions)
	t.impressions = append(t.impressions, impression)
	
	// Update indexes
	t.queryImpressions[impression.QueryID] = append(t.queryImpressions[impression.QueryID], idx)
	t.documentImpressions[impression.DocumentID] = append(t.documentImpressions[impression.DocumentID], idx)
	
	// Update metrics
	t.updateDocumentMetrics(impression.DocumentID, &impression, nil)
	t.updateQueryMetrics(impression.Query, &impression, nil)
	t.updatePositionMetrics(impression.Position, &impression, nil)
	
	// Clean up old data if needed
	if len(t.impressions) > t.maxDataSize {
		t.cleanupOldData()
	}
	
	return nil
}

func (t *memoryCTRTracker) RecordClick(ctx context.Context, click Click) error {
	if click.Timestamp.IsZero() {
		click.Timestamp = time.Now()
	}
	
	t.mu.Lock()
	defer t.mu.Unlock()
	
	// Add click
	t.clicks = append(t.clicks, click)
	
	// Find corresponding impression
	impressionIndices := t.queryImpressions[click.QueryID]
	var impression *Impression
	for _, idx := range impressionIndices {
		if idx < len(t.impressions) && t.impressions[idx].DocumentID == click.DocumentID {
			impression = &t.impressions[idx]
			break
		}
	}
	
	// Update metrics
	t.updateDocumentMetrics(click.DocumentID, nil, &click)
	if impression != nil {
		t.updateQueryMetrics(impression.Query, nil, &click)
	}
	t.updatePositionMetrics(click.Position, nil, &click)
	
	return nil
}

func (t *memoryCTRTracker) GetDocumentCTR(ctx context.Context, documentID string) (CTRMetrics, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	metrics, exists := t.documentMetrics[documentID]
	if !exists {
		return CTRMetrics{}, fmt.Errorf("no CTR data for document: %s", documentID)
	}
	
	return *metrics, nil
}

func (t *memoryCTRTracker) GetQueryCTR(ctx context.Context, query string) (CTRMetrics, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	metrics, exists := t.queryMetrics[query]
	if !exists {
		return CTRMetrics{}, fmt.Errorf("no CTR data for query: %s", query)
	}
	
	return *metrics, nil
}

func (t *memoryCTRTracker) GetPositionCTR(ctx context.Context, position int) (CTRMetrics, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	metrics, exists := t.positionMetrics[position]
	if !exists {
		return CTRMetrics{}, fmt.Errorf("no CTR data for position: %d", position)
	}
	
	return *metrics, nil
}

func (t *memoryCTRTracker) GetOptimalRanking(ctx context.Context, documentIDs []string, query string) ([]string, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	// Score each document based on CTR and position bias
	type docScore struct {
		id    string
		score float64
	}
	
	scores := make([]docScore, len(documentIDs))
	for i, docID := range documentIDs {
		score := 0.0
		
		// Get document CTR
		if metrics, exists := t.documentMetrics[docID]; exists {
			score += metrics.CTR * 0.5
		}
		
		// Get position bias correction
		position := i + 1
		if posMetrics, exists := t.positionMetrics[position]; exists {
			// Correct for position bias
			expectedCTR := posMetrics.CTR
			if expectedCTR > 0 {
				score = score / expectedCTR
			}
		}
		
		// Boost based on query-specific CTR if available
		// (This would require more sophisticated tracking)
		
		scores[i] = docScore{id: docID, score: score}
	}
	
	// Sort by score
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}
	
	// Return optimized ordering
	result := make([]string, len(documentIDs))
	for i, ds := range scores {
		result[i] = ds.id
	}
	
	return result, nil
}

func (t *memoryCTRTracker) ExportMetrics(ctx context.Context) (CTRReport, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	report := CTRReport{
		GeneratedAt:      time.Now(),
		TotalImpressions: int64(len(t.impressions)),
		TotalClicks:      int64(len(t.clicks)),
		PositionCTR:      make(map[int]CTRMetrics),
		HourlyTrends:     make(map[int]CTRMetrics),
		UserSegments:     make(map[string]CTRMetrics),
	}
	
	if report.TotalImpressions > 0 {
		report.OverallCTR = float64(report.TotalClicks) / float64(report.TotalImpressions)
	}
	
	// Top documents by CTR
	for docID, metrics := range t.documentMetrics {
		if metrics.Impressions >= 10 { // Minimum threshold
			report.TopDocuments = append(report.TopDocuments, DocumentCTR{
				DocumentID: docID,
				CTRMetrics: *metrics,
			})
		}
	}
	
	// Sort top documents by CTR
	for i := 0; i < len(report.TopDocuments) && i < 10; i++ {
		for j := i + 1; j < len(report.TopDocuments); j++ {
			if report.TopDocuments[j].CTRMetrics.CTR > report.TopDocuments[i].CTRMetrics.CTR {
				report.TopDocuments[i], report.TopDocuments[j] = report.TopDocuments[j], report.TopDocuments[i]
			}
		}
	}
	if len(report.TopDocuments) > 10 {
		report.TopDocuments = report.TopDocuments[:10]
	}
	
	// Top queries by CTR
	for query, metrics := range t.queryMetrics {
		if metrics.Impressions >= 5 { // Minimum threshold
			report.TopQueries = append(report.TopQueries, QueryCTR{
				Query:      query,
				CTRMetrics: *metrics,
			})
		}
	}
	
	// Position CTR
	for pos, metrics := range t.positionMetrics {
		report.PositionCTR[pos] = *metrics
	}
	
	// Calculate hourly trends
	hourlyImpressions := make(map[int]int64)
	hourlyClicks := make(map[int]int64)
	
	for _, imp := range t.impressions {
		hour := imp.Timestamp.Hour()
		hourlyImpressions[hour]++
	}
	
	for _, click := range t.clicks {
		hour := click.Timestamp.Hour()
		hourlyClicks[hour]++
	}
	
	for hour := 0; hour < 24; hour++ {
		if impressions := hourlyImpressions[hour]; impressions > 0 {
			ctr := 0.0
			if clicks := hourlyClicks[hour]; clicks > 0 {
				ctr = float64(clicks) / float64(impressions)
			}
			report.HourlyTrends[hour] = CTRMetrics{
				Impressions: impressions,
				Clicks:      hourlyClicks[hour],
				CTR:         ctr,
			}
		}
	}
	
	return report, nil
}

// Helper methods

func (t *memoryCTRTracker) updateDocumentMetrics(documentID string, impression *Impression, click *Click) {
	metrics, exists := t.documentMetrics[documentID]
	if !exists {
		metrics = &CTRMetrics{
			LastUpdated: time.Now(),
		}
		t.documentMetrics[documentID] = metrics
	}
	
	if impression != nil {
		metrics.Impressions++
		// Update average position
		metrics.AvgPosition = (metrics.AvgPosition*float64(metrics.Impressions-1) + float64(impression.Position)) / float64(metrics.Impressions)
	}
	
	if click != nil {
		metrics.Clicks++
		// Update average dwell time
		if click.DwellTime > 0 {
			totalDwell := metrics.AvgDwellTime * float64(metrics.Clicks-1)
			metrics.AvgDwellTime = (totalDwell + click.DwellTime) / float64(metrics.Clicks)
		}
	}
	
	// Update CTR
	if metrics.Impressions > 0 {
		metrics.CTR = float64(metrics.Clicks) / float64(metrics.Impressions)
		// Confidence score based on sample size
		metrics.ConfidenceScore = 1.0 - 1.0/float64(metrics.Impressions+1)
	}
	
	metrics.LastUpdated = time.Now()
}

func (t *memoryCTRTracker) updateQueryMetrics(query string, impression *Impression, click *Click) {
	if query == "" {
		return
	}
	
	metrics, exists := t.queryMetrics[query]
	if !exists {
		metrics = &CTRMetrics{
			LastUpdated: time.Now(),
		}
		t.queryMetrics[query] = metrics
	}
	
	if impression != nil {
		metrics.Impressions++
	}
	
	if click != nil {
		metrics.Clicks++
		if click.DwellTime > 0 {
			totalDwell := metrics.AvgDwellTime * float64(metrics.Clicks-1)
			metrics.AvgDwellTime = (totalDwell + click.DwellTime) / float64(metrics.Clicks)
		}
	}
	
	// Update CTR
	if metrics.Impressions > 0 {
		metrics.CTR = float64(metrics.Clicks) / float64(metrics.Impressions)
		metrics.ConfidenceScore = 1.0 - 1.0/float64(metrics.Impressions+1)
	}
	
	metrics.LastUpdated = time.Now()
}

func (t *memoryCTRTracker) updatePositionMetrics(position int, impression *Impression, click *Click) {
	if position <= 0 {
		return
	}
	
	metrics, exists := t.positionMetrics[position]
	if !exists {
		metrics = &CTRMetrics{
			AvgPosition: float64(position),
			LastUpdated: time.Now(),
		}
		t.positionMetrics[position] = metrics
	}
	
	if impression != nil {
		metrics.Impressions++
	}
	
	if click != nil {
		metrics.Clicks++
		if click.DwellTime > 0 {
			totalDwell := metrics.AvgDwellTime * float64(metrics.Clicks-1)
			metrics.AvgDwellTime = (totalDwell + click.DwellTime) / float64(metrics.Clicks)
		}
	}
	
	// Update CTR
	if metrics.Impressions > 0 {
		metrics.CTR = float64(metrics.Clicks) / float64(metrics.Impressions)
		metrics.ConfidenceScore = 1.0 - 1.0/float64(metrics.Impressions+1)
	}
	
	metrics.LastUpdated = time.Now()
}

func (t *memoryCTRTracker) cleanupOldData() {
	// Keep only recent data (last 90% of data)
	keepSize := int(float64(t.maxDataSize) * 0.9)
	
	if len(t.impressions) > keepSize {
		// Remove oldest impressions
		removeCount := len(t.impressions) - keepSize
		t.impressions = t.impressions[removeCount:]
		
		// Rebuild indexes
		t.queryImpressions = make(map[string][]int)
		t.documentImpressions = make(map[string][]int)
		
		for i, imp := range t.impressions {
			t.queryImpressions[imp.QueryID] = append(t.queryImpressions[imp.QueryID], i)
			t.documentImpressions[imp.DocumentID] = append(t.documentImpressions[imp.DocumentID], i)
		}
	}
	
	// Apply decay to old metrics
	for _, metrics := range t.documentMetrics {
		age := time.Since(metrics.LastUpdated)
		if age > 24*time.Hour {
			decayFactor := t.decayFactor
			metrics.Impressions = int64(float64(metrics.Impressions) * decayFactor)
			metrics.Clicks = int64(float64(metrics.Clicks) * decayFactor)
			if metrics.Impressions > 0 {
				metrics.CTR = float64(metrics.Clicks) / float64(metrics.Impressions)
			}
		}
	}
}