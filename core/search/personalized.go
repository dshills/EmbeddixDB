package search

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/feedback"
	"github.com/dshills/EmbeddixDB/core/text"
)

// PersonalizedSearchManager extends hybrid search with personalization and learning
type PersonalizedSearchManager struct {
	vectorStore     core.VectorStore
	textEngine      *text.EnhancedBM25Index
	feedbackManager *FeedbackManager
	reranker        feedback.Reranker
	modelManager    ai.ModelManager

	// Configuration
	config PersonalizedSearchConfig
}

// PersonalizedSearchConfig contains configuration for personalized search
type PersonalizedSearchConfig struct {
	EnablePersonalization bool
	EnableLearning        bool
	EnableSessionContext  bool
	DefaultVectorWeight   float64
	DefaultTextWeight     float64
	MaxResults            int
	MinScore              float64
}

// FeedbackManager manages all feedback-related components
type FeedbackManager struct {
	Collector      feedback.Collector
	SessionManager feedback.SessionManager
	ProfileManager feedback.ProfileManager
	LearningEngine feedback.LearningEngine
	Analyzer       feedback.FeedbackAnalyzer
	CTRTracker     feedback.CTRTracker
}

// PersonalizedSearchRequest extends the basic search request
type PersonalizedSearchRequest struct {
	// Base search parameters
	CollectionID string                 `json:"collection_id"`
	Query        string                 `json:"query"`
	Vector       []float32              `json:"vector,omitempty"`
	K            int                    `json:"k"`
	Filter       map[string]interface{} `json:"filter,omitempty"`

	// Personalization parameters
	UserID     string `json:"user_id,omitempty"`
	SessionID  string `json:"session_id,omitempty"`
	SearchMode string `json:"search_mode,omitempty"` // "vector", "text", "hybrid", "auto"

	// Weight overrides
	VectorWeight *float64 `json:"vector_weight,omitempty"`
	TextWeight   *float64 `json:"text_weight,omitempty"`

	// Advanced options
	EnableReranking       bool               `json:"enable_reranking"`
	EnablePersonalization bool               `json:"enable_personalization"`
	CollectFeedback       bool               `json:"collect_feedback"`
	BoostFields           map[string]float64 `json:"boost_fields,omitempty"`
}

// PersonalizedSearchResult extends the basic search result
type PersonalizedSearchResult struct {
	ID                string                 `json:"id"`
	Score             float64                `json:"score"`
	PersonalizedScore float64                `json:"personalized_score"`
	Vector            []float32              `json:"vector,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
	Explanation       map[string]interface{} `json:"explanation,omitempty"`
	Position          int                    `json:"position"`
}

// PersonalizedSearchResponse contains the search response with additional context
type PersonalizedSearchResponse struct {
	Results      []PersonalizedSearchResult `json:"results"`
	QueryID      string                     `json:"query_id"`
	SessionID    string                     `json:"session_id"`
	SearchMode   string                     `json:"search_mode"`
	TotalResults int                        `json:"total_results"`
	SearchTime   time.Duration              `json:"search_time"`
	Personalized bool                       `json:"personalized"`
}

// NewPersonalizedSearchManager creates a new personalized search manager
func NewPersonalizedSearchManager(
	vectorStore core.VectorStore,
	textEngine *text.EnhancedBM25Index,
	modelManager ai.ModelManager,
	feedbackManager *FeedbackManager,
	config PersonalizedSearchConfig,
) *PersonalizedSearchManager {

	// Create reranker
	reranker := feedback.NewContextualReranker(
		feedbackManager.Collector,
		feedbackManager.ProfileManager,
		feedbackManager.LearningEngine,
	)

	return &PersonalizedSearchManager{
		vectorStore:     vectorStore,
		textEngine:      textEngine,
		modelManager:    modelManager,
		feedbackManager: feedbackManager,
		reranker:        reranker,
		config:          config,
	}
}

// Search performs personalized hybrid search
func (m *PersonalizedSearchManager) Search(ctx context.Context, req PersonalizedSearchRequest) (*PersonalizedSearchResponse, error) {
	startTime := time.Now()

	// Generate query ID for tracking
	queryID := generateQueryID()

	// Get or create session
	session, err := m.getOrCreateSession(ctx, req.UserID, req.SessionID)
	if err != nil {
		return nil, fmt.Errorf("session error: %w", err)
	}

	// Get user profile if personalization is enabled
	var userProfile *feedback.UserProfile
	if req.EnablePersonalization && req.UserID != "" {
		userProfile, _ = m.feedbackManager.ProfileManager.GetProfile(ctx, req.UserID)
		if userProfile == nil && m.config.EnablePersonalization {
			// Auto-create profile
			userProfile, _ = m.feedbackManager.ProfileManager.CreateProfile(ctx, req.UserID)
		}
	}

	// Determine search weights
	vectorWeight, textWeight := m.determineSearchWeights(req, userProfile)

	// Determine search mode
	searchMode := m.determineSearchMode(req, vectorWeight, textWeight)

	// Perform base search
	results, err := m.performBaseSearch(ctx, req, searchMode, vectorWeight, textWeight)
	if err != nil {
		return nil, fmt.Errorf("search error: %w", err)
	}

	// Convert to feedback search results for reranking
	feedbackResults := m.convertToFeedbackResults(results, req.CollectionID)

	// Apply re-ranking if enabled
	if req.EnableReranking || m.config.EnableLearning {
		rerankCtx := feedback.RerankingContext{
			UserID:      req.UserID,
			SessionID:   session.ID,
			Query:       req.Query,
			Timestamp:   time.Now(),
			UserProfile: userProfile,
		}

		// Get session history for context
		if m.config.EnableSessionContext {
			history, _ := m.getSessionHistory(ctx, session.ID)
			rerankCtx.SessionHistory = history
		}

		// Rerank results
		rerankedResults, err := m.reranker.Rerank(ctx, feedbackResults, rerankCtx)
		if err == nil {
			feedbackResults = rerankedResults
		}
	}

	// Convert back to personalized results
	personalizedResults := m.convertToPersonalizedResults(feedbackResults)

	// Record query feedback if enabled
	if req.CollectFeedback {
		m.recordQueryFeedback(ctx, queryID, req, session.ID, len(personalizedResults))
	}

	// Update user profile based on query
	if userProfile != nil {
		m.updateUserProfileFromQuery(ctx, req.UserID, req.Query, personalizedResults)
	}

	// Record impressions for CTR tracking
	if m.feedbackManager.CTRTracker != nil && req.CollectFeedback {
		m.recordImpressions(ctx, queryID, req, session.ID, personalizedResults)
	}

	// Build response
	response := &PersonalizedSearchResponse{
		Results:      personalizedResults,
		QueryID:      queryID,
		SessionID:    session.ID,
		SearchMode:   searchMode,
		TotalResults: len(personalizedResults),
		SearchTime:   time.Since(startTime),
		Personalized: req.EnablePersonalization && userProfile != nil,
	}

	return response, nil
}

// RecordInteraction records user interaction with search results
func (m *PersonalizedSearchManager) RecordInteraction(ctx context.Context, interaction *feedback.Interaction) error {
	// Record the interaction
	if err := m.feedbackManager.Collector.RecordInteraction(ctx, interaction); err != nil {
		return err
	}

	// Record click for CTR tracking
	if m.feedbackManager.CTRTracker != nil && interaction.Type == feedback.InteractionTypeClick {
		click := feedback.Click{
			QueryID:    interaction.QueryID,
			DocumentID: interaction.DocumentID,
			Position:   interaction.Position,
			UserID:     interaction.UserID,
			SessionID:  interaction.SessionID,
			Timestamp:  interaction.Timestamp,
			DwellTime:  interaction.Value, // If tracking dwell time
		}
		m.feedbackManager.CTRTracker.RecordClick(ctx, click)
	}

	// Update user interests based on interaction
	if interaction.Type == feedback.InteractionTypeClick ||
		interaction.Type == feedback.InteractionTypeDwell ||
		interaction.Type == feedback.InteractionTypeRating {
		m.updateUserInterestsFromInteraction(ctx, interaction)
	}

	// Trigger model update if enough feedback collected
	m.triggerModelUpdateIfNeeded(ctx)

	return nil
}

// Helper methods

func (m *PersonalizedSearchManager) getOrCreateSession(ctx context.Context, userID, sessionID string) (*feedback.Session, error) {
	if sessionID != "" {
		session, err := m.feedbackManager.SessionManager.GetSession(ctx, sessionID)
		if err == nil {
			return session, nil
		}
	}

	// Create new session
	if userID == "" {
		userID = "anonymous"
	}

	return m.feedbackManager.SessionManager.CreateSession(ctx, userID, nil)
}

func (m *PersonalizedSearchManager) determineSearchWeights(req PersonalizedSearchRequest, profile *feedback.UserProfile) (float64, float64) {
	vectorWeight := m.config.DefaultVectorWeight
	textWeight := m.config.DefaultTextWeight

	// Use request overrides if provided
	if req.VectorWeight != nil {
		vectorWeight = *req.VectorWeight
	}
	if req.TextWeight != nil {
		textWeight = *req.TextWeight
	}

	// Apply user preferences if available
	if profile != nil && req.EnablePersonalization {
		if req.VectorWeight == nil {
			vectorWeight = profile.Preferences.VectorWeight
		}
		if req.TextWeight == nil {
			textWeight = profile.Preferences.TextWeight
		}
	}

	// Normalize weights
	total := vectorWeight + textWeight
	if total > 0 {
		vectorWeight /= total
		textWeight /= total
	}

	return vectorWeight, textWeight
}

func (m *PersonalizedSearchManager) determineSearchMode(req PersonalizedSearchRequest, vectorWeight, textWeight float64) string {
	if req.SearchMode != "" && req.SearchMode != "auto" {
		return req.SearchMode
	}

	// Auto-determine based on inputs and weights
	hasVector := len(req.Vector) > 0
	hasQuery := req.Query != ""

	if hasVector && hasQuery {
		return "hybrid"
	} else if hasVector {
		return "vector"
	} else if hasQuery {
		return "text"
	}

	// Default based on weights
	if vectorWeight > textWeight {
		return "vector"
	}
	return "text"
}

func (m *PersonalizedSearchManager) performBaseSearch(
	ctx context.Context,
	req PersonalizedSearchRequest,
	searchMode string,
	vectorWeight, textWeight float64,
) ([]ai.SearchResult, error) {

	switch searchMode {
	case "vector":
		return m.performVectorSearch(ctx, req)
	case "text":
		return m.performTextSearch(ctx, req)
	case "hybrid":
		return m.performHybridSearch(ctx, req, vectorWeight, textWeight)
	default:
		return nil, fmt.Errorf("unknown search mode: %s", searchMode)
	}
}

func (m *PersonalizedSearchManager) performVectorSearch(ctx context.Context, req PersonalizedSearchRequest) ([]ai.SearchResult, error) {
	if len(req.Vector) == 0 {
		return nil, fmt.Errorf("vector required for vector search")
	}

	searchReq := core.SearchRequest{
		Query:          req.Vector,
		TopK:           req.K,
		Filter:         convertStringMap(req.Filter),
		IncludeVectors: false,
	}

	results, err := m.vectorStore.Search(ctx, req.CollectionID, searchReq)
	if err != nil {
		return nil, err
	}

	// Convert to search results
	searchResults := make([]ai.SearchResult, len(results))
	for i, r := range results {
		searchResults[i] = ai.SearchResult{
			ID:       r.ID,
			Score:    float64(r.Score),
			Metadata: convertMetadata(r.Metadata),
			Explanation: &ai.SearchExplanation{
				VectorScore: float64(r.Score),
			},
		}
	}

	return searchResults, nil
}

func (m *PersonalizedSearchManager) performTextSearch(ctx context.Context, req PersonalizedSearchRequest) ([]ai.SearchResult, error) {
	if req.Query == "" {
		return nil, fmt.Errorf("query required for text search")
	}

	results, err := m.textEngine.Search(ctx, req.Query, req.K)
	if err != nil {
		return nil, err
	}

	// Convert to search results
	searchResults := make([]ai.SearchResult, 0, len(results))
	for _, r := range results {
		// Get vector and metadata from vector store
		vector, err := m.vectorStore.GetVector(ctx, req.CollectionID, r.ID)
		if err != nil {
			continue
		}

		searchResults = append(searchResults, ai.SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Metadata: convertMetadata(vector.Metadata),
			Explanation: &ai.SearchExplanation{
				TextScore: r.Score,
			},
		})
	}

	return searchResults, nil
}

func (m *PersonalizedSearchManager) performHybridSearch(
	ctx context.Context,
	req PersonalizedSearchRequest,
	vectorWeight, textWeight float64,
) ([]ai.SearchResult, error) {

	// Parallel search
	var vectorResults, textResults []ai.SearchResult
	var vectorErr, textErr error
	var wg sync.WaitGroup

	if len(req.Vector) > 0 && vectorWeight > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vectorResults, vectorErr = m.performVectorSearch(ctx, req)
		}()
	}

	if req.Query != "" && textWeight > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			textResults, textErr = m.performTextSearch(ctx, req)
		}()
	}

	wg.Wait()

	// Check errors
	if vectorErr != nil && textErr != nil {
		return nil, fmt.Errorf("both searches failed: vector=%v, text=%v", vectorErr, textErr)
	}

	// Fuse results using RRF
	results := m.fuseResults(vectorResults, textResults, vectorWeight, textWeight)

	// Limit results
	if len(results) > req.K {
		results = results[:req.K]
	}

	return results, nil
}

func (m *PersonalizedSearchManager) convertToFeedbackResults(results []ai.SearchResult, collectionID string) []feedback.SearchResult {
	feedbackResults := make([]feedback.SearchResult, len(results))
	for i, r := range results {
		feedbackResults[i] = feedback.SearchResult{
			ID:           r.ID,
			Score:        r.Score,
			VectorScore:  getAIVectorScore(r),
			TextScore:    getAITextScore(r),
			Metadata:     r.Metadata,
			Position:     i + 1,
			CollectionID: collectionID,
		}
	}
	return feedbackResults
}

func (m *PersonalizedSearchManager) convertToPersonalizedResults(results []feedback.SearchResult) []PersonalizedSearchResult {
	personalizedResults := make([]PersonalizedSearchResult, len(results))
	for i, r := range results {
		personalizedResults[i] = PersonalizedSearchResult{
			ID:                r.ID,
			Score:             r.Score,
			PersonalizedScore: r.Score,
			Metadata:          r.Metadata,
			Position:          r.Position,
			Explanation: map[string]interface{}{
				"vector_score": r.VectorScore,
				"text_score":   r.TextScore,
				"reranked":     true,
			},
		}
	}
	return personalizedResults
}

func (m *PersonalizedSearchManager) recordQueryFeedback(
	ctx context.Context,
	queryID string,
	req PersonalizedSearchRequest,
	sessionID string,
	resultCount int,
) {
	feedback := &feedback.QueryFeedback{
		QueryID:     queryID,
		Query:       req.Query,
		UserID:      req.UserID,
		SessionID:   sessionID,
		Timestamp:   time.Now(),
		ResultCount: resultCount,
	}

	m.feedbackManager.Collector.RecordQuery(ctx, feedback)
}

func (m *PersonalizedSearchManager) updateUserProfileFromQuery(
	ctx context.Context,
	userID string,
	query string,
	results []PersonalizedSearchResult,
) {
	// Extract topics and entities from top results
	topics := make(map[string]float64)
	entities := make(map[string]float64)

	for i, result := range results {
		if i >= 5 { // Only consider top 5 results
			break
		}

		weight := 1.0 / float64(i+1) // Weight by position

		if resultTopics, ok := result.Metadata["topics"].([]string); ok {
			for _, topic := range resultTopics {
				topics[topic] += weight * 0.1 // Small increment for query
			}
		}

		if resultEntities, ok := result.Metadata["entities"].([]string); ok {
			for _, entity := range resultEntities {
				entities[entity] += weight * 0.1
			}
		}
	}

	// Update profile
	m.feedbackManager.ProfileManager.IncrementInterests(ctx, userID, topics, entities)
}

func (m *PersonalizedSearchManager) updateUserInterestsFromInteraction(ctx context.Context, interaction *feedback.Interaction) {
	if interaction.UserID == "" {
		return
	}

	// Get document to extract topics/entities
	vector, err := m.vectorStore.GetVector(ctx, interaction.CollectionID, interaction.DocumentID)
	if err != nil || vector.Metadata == nil {
		return
	}

	// Weight based on interaction type
	weight := 0.0
	switch interaction.Type {
	case feedback.InteractionTypeClick:
		weight = 0.3
	case feedback.InteractionTypeDwell:
		weight = interaction.Value / 60.0 // Normalize by minute
	case feedback.InteractionTypeBookmark:
		weight = 1.0
	case feedback.InteractionTypeShare:
		weight = 0.8
	case feedback.InteractionTypeRating:
		weight = interaction.Value / 5.0
	}

	if weight <= 0 {
		return
	}

	// Extract and update interests
	topics := make(map[string]float64)
	entities := make(map[string]float64)

	// Parse topics from metadata (assuming comma-separated string)
	if topicsStr, ok := vector.Metadata["topics"]; ok {
		for _, topic := range strings.Split(topicsStr, ",") {
			if topic = strings.TrimSpace(topic); topic != "" {
				topics[topic] = weight
			}
		}
	}

	// Parse entities from metadata (assuming comma-separated string)
	if entitiesStr, ok := vector.Metadata["entities"]; ok {
		for _, entity := range strings.Split(entitiesStr, ",") {
			if entity = strings.TrimSpace(entity); entity != "" {
				entities[entity] = weight
			}
		}
	}

	m.feedbackManager.ProfileManager.IncrementInterests(ctx, interaction.UserID, topics, entities)
}

func (m *PersonalizedSearchManager) getSessionHistory(ctx context.Context, sessionID string) ([]*feedback.Interaction, error) {
	filter := feedback.InteractionFilter{
		SessionID: sessionID,
		Limit:     20, // Last 20 interactions
	}

	return m.feedbackManager.Collector.GetInteractions(ctx, filter)
}

func (m *PersonalizedSearchManager) triggerModelUpdateIfNeeded(ctx context.Context) {
	// This would typically check if enough feedback has been collected
	// and trigger a background model update
	// For now, this is a placeholder
}

func generateQueryID() string {
	return fmt.Sprintf("q_%d_%d", time.Now().Unix(), time.Now().Nanosecond())
}

// convertMetadata converts map[string]string to map[string]interface{}
func convertMetadata(metadata map[string]string) map[string]interface{} {
	if metadata == nil {
		return nil
	}
	result := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		result[k] = v
	}
	return result
}

// getVectorScore extracts vector score from explanation
func getVectorScore(r feedback.SearchResult) float64 {
	if r.VectorScore > 0 {
		return r.VectorScore
	}
	return 0.0
}

// getTextScore extracts text score from explanation
func getTextScore(r feedback.SearchResult) float64 {
	if r.TextScore > 0 {
		return r.TextScore
	}
	return 0.0
}

// getAIVectorScore extracts vector score from ai.SearchResult
func getAIVectorScore(r ai.SearchResult) float64 {
	if r.Explanation != nil && r.Explanation.VectorScore > 0 {
		return r.Explanation.VectorScore
	}
	return 0.0
}

// getAITextScore extracts text score from ai.SearchResult
func getAITextScore(r ai.SearchResult) float64 {
	if r.Explanation != nil && r.Explanation.TextScore > 0 {
		return r.Explanation.TextScore
	}
	return 0.0
}

// fuseResults combines vector and text search results
func (m *PersonalizedSearchManager) fuseResults(vectorResults, textResults []ai.SearchResult, vectorWeight, textWeight float64) []ai.SearchResult {
	// Create a map to track document scores
	docScores := make(map[string]*fusedScore)

	// Process vector results
	for i, result := range vectorResults {
		docScores[result.ID] = &fusedScore{
			result:      result,
			vectorRank:  i + 1,
			vectorScore: result.Score,
		}
	}

	// Process text results
	for i, result := range textResults {
		if fs, exists := docScores[result.ID]; exists {
			fs.textRank = i + 1
			fs.textScore = result.Score
		} else {
			docScores[result.ID] = &fusedScore{
				result:    result,
				textRank:  i + 1,
				textScore: result.Score,
			}
		}
	}

	// Calculate fused scores using RRF
	k := 60.0 // RRF constant
	for _, fs := range docScores {
		vectorRRF := 0.0
		textRRF := 0.0

		if fs.vectorRank > 0 {
			vectorRRF = 1.0 / (k + float64(fs.vectorRank))
		}
		if fs.textRank > 0 {
			textRRF = 1.0 / (k + float64(fs.textRank))
		}

		fs.fusedScore = vectorWeight*vectorRRF + textWeight*textRRF
	}

	// Convert to slice and sort
	results := make([]ai.SearchResult, 0, len(docScores))
	for _, fs := range docScores {
		result := fs.result
		result.Score = fs.fusedScore
		if result.Explanation == nil {
			result.Explanation = &ai.SearchExplanation{}
		}
		result.Explanation.FusionScore = fs.fusedScore
		result.Explanation.VectorScore = fs.vectorScore
		result.Explanation.TextScore = fs.textScore
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

type fusedScore struct {
	result      ai.SearchResult
	vectorRank  int
	textRank    int
	vectorScore float64
	textScore   float64
	fusedScore  float64
}

// recordImpressions records search result impressions for CTR tracking
func (m *PersonalizedSearchManager) recordImpressions(
	ctx context.Context,
	queryID string,
	req PersonalizedSearchRequest,
	sessionID string,
	results []PersonalizedSearchResult,
) {
	for _, result := range results {
		impression := feedback.Impression{
			QueryID:     queryID,
			Query:       req.Query,
			DocumentID:  result.ID,
			Position:    result.Position,
			UserID:      req.UserID,
			SessionID:   sessionID,
			Timestamp:   time.Now(),
			ResultCount: len(results),
			SearchMode:  req.SearchMode,
		}

		// Record impression (ignore errors for non-critical operation)
		m.feedbackManager.CTRTracker.RecordImpression(ctx, impression)
	}
}

// GetCTROptimizedRanking re-ranks results based on CTR data
func (m *PersonalizedSearchManager) GetCTROptimizedRanking(
	ctx context.Context,
	results []PersonalizedSearchResult,
	query string,
) ([]PersonalizedSearchResult, error) {
	if m.feedbackManager.CTRTracker == nil {
		return results, nil
	}

	// Extract document IDs
	docIDs := make([]string, len(results))
	for i, r := range results {
		docIDs[i] = r.ID
	}

	// Get optimal ranking based on CTR
	optimalIDs, err := m.feedbackManager.CTRTracker.GetOptimalRanking(ctx, docIDs, query)
	if err != nil {
		return results, err
	}

	// Create a map for quick lookup
	resultMap := make(map[string]PersonalizedSearchResult)
	for _, r := range results {
		resultMap[r.ID] = r
	}

	// Reorder results
	reorderedResults := make([]PersonalizedSearchResult, 0, len(results))
	for i, id := range optimalIDs {
		if r, exists := resultMap[id]; exists {
			r.Position = i + 1
			reorderedResults = append(reorderedResults, r)
		}
	}

	return reorderedResults, nil
}

// GetCTRReport generates a CTR report
func (m *PersonalizedSearchManager) GetCTRReport(ctx context.Context) (feedback.CTRReport, error) {
	if m.feedbackManager.CTRTracker == nil {
		return feedback.CTRReport{}, fmt.Errorf("CTR tracking not enabled")
	}

	return m.feedbackManager.CTRTracker.ExportMetrics(ctx)
}

// Helper functions

func convertStringMap(input map[string]interface{}) map[string]string {
	if input == nil {
		return nil
	}
	result := make(map[string]string)
	for k, v := range input {
		if str, ok := v.(string); ok {
			result[k] = str
		}
	}
	return result
}
