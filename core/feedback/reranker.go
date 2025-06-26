package feedback

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"
)

// SearchResult represents a search result to be re-ranked
type SearchResult struct {
	ID           string                 `json:"id"`
	Score        float64                `json:"score"`
	VectorScore  float64                `json:"vector_score,omitempty"`
	TextScore    float64                `json:"text_score,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Position     int                    `json:"position"`
	CollectionID string                 `json:"collection_id"`
}

// RerankingContext contains context for re-ranking
type RerankingContext struct {
	UserID         string
	SessionID      string
	Query          string
	Timestamp      time.Time
	UserProfile    *UserProfile
	SessionHistory []*Interaction
}

// Reranker defines the interface for re-ranking search results
type Reranker interface {
	// Rerank re-ranks search results based on context and learned patterns
	Rerank(ctx context.Context, results []SearchResult, rerankCtx RerankingContext) ([]SearchResult, error)

	// UpdateModel updates the re-ranking model based on feedback
	UpdateModel(ctx context.Context, feedback []*Interaction) error

	// GetFeatures extracts features for a result in context
	GetFeatures(result SearchResult, rerankCtx RerankingContext) map[string]float64
}

// contextualReranker implements learning-based re-ranking
type contextualReranker struct {
	mu             sync.RWMutex
	collector      Collector
	profileManager ProfileManager
	learningEngine LearningEngine

	// Feature weights learned from interactions
	featureWeights map[string]float64

	// Position bias model
	positionBias []float64

	// Click models
	clickModel ClickModel
}

// NewContextualReranker creates a new contextual re-ranker
func NewContextualReranker(collector Collector, profileManager ProfileManager, learningEngine LearningEngine) Reranker {
	return &contextualReranker{
		collector:      collector,
		profileManager: profileManager,
		learningEngine: learningEngine,
		featureWeights: initializeFeatureWeights(),
		positionBias:   initializePositionBias(),
		clickModel: ClickModel{
			Type: "cascade",
			Parameters: map[string]float64{
				"click_probability": 0.5,
				"stop_probability":  0.3,
			},
		},
	}
}

func (r *contextualReranker) Rerank(ctx context.Context, results []SearchResult, rerankCtx RerankingContext) ([]SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	// Extract features for each result
	resultFeatures := make([]map[string]float64, len(results))
	for i, result := range results {
		resultFeatures[i] = r.GetFeatures(result, rerankCtx)
	}

	// Compute re-ranking scores
	rerankedResults := make([]SearchResult, len(results))
	copy(rerankedResults, results)

	for i := range rerankedResults {
		features := resultFeatures[i]

		// Base score from original search
		score := rerankedResults[i].Score

		// Apply feature-based adjustments
		r.mu.RLock()
		for feature, value := range features {
			if weight, exists := r.featureWeights[feature]; exists {
				score += weight * value
			}
		}
		r.mu.RUnlock()

		// Apply personalization if profile exists
		if rerankCtx.UserProfile != nil {
			score = r.applyPersonalization(score, rerankedResults[i], rerankCtx.UserProfile)
		}

		// Apply temporal boost for recent content
		score = r.applyTemporalBoost(score, rerankedResults[i])

		// Apply diversity penalty if needed
		score = r.applyDiversityPenalty(score, rerankedResults[i], rerankedResults[:i])

		rerankedResults[i].Score = score
	}

	// Sort by new scores
	sort.Slice(rerankedResults, func(i, j int) bool {
		return rerankedResults[i].Score > rerankedResults[j].Score
	})

	// Update positions
	for i := range rerankedResults {
		rerankedResults[i].Position = i + 1
	}

	return rerankedResults, nil
}

func (r *contextualReranker) GetFeatures(result SearchResult, rerankCtx RerankingContext) map[string]float64 {
	features := make(map[string]float64)

	// Query-document features
	features["original_score"] = result.Score
	features["vector_score"] = result.VectorScore
	features["text_score"] = result.TextScore
	features["score_variance"] = math.Abs(result.VectorScore - result.TextScore)

	// Position features
	features["original_position"] = float64(result.Position)
	features["inverse_position"] = 1.0 / float64(result.Position)

	// Metadata features
	if result.Metadata != nil {
		// Document quality signals
		if views, ok := result.Metadata["view_count"].(float64); ok {
			features["document_popularity"] = math.Log1p(views)
		}
		if rating, ok := result.Metadata["avg_rating"].(float64); ok {
			features["document_rating"] = rating
		}

		// Freshness
		if created, ok := result.Metadata["created_at"].(time.Time); ok {
			age := time.Since(created).Hours()
			features["document_freshness"] = 1.0 / (1.0 + age/24.0) // Decay over days
		}

		// Source authority
		if source, ok := result.Metadata["source"].(string); ok {
			features["source_authority"] = r.getSourceAuthority(source)
		}

		// Topic relevance
		if topics, ok := result.Metadata["topics"].([]string); ok && rerankCtx.UserProfile != nil {
			features["topic_relevance"] = r.computeTopicRelevance(topics, rerankCtx.UserProfile)
		}
	}

	// User-document features (if we have historical data)
	if rerankCtx.UserID != "" {
		features["user_doc_clicks"] = r.getUserDocumentClicks(rerankCtx.UserID, result.ID)
		features["user_doc_dwell"] = r.getUserDocumentDwellTime(rerankCtx.UserID, result.ID)
	}

	// Session features
	if len(rerankCtx.SessionHistory) > 0 {
		features["session_topic_consistency"] = r.computeSessionConsistency(result, rerankCtx.SessionHistory)
		features["session_diversity_needed"] = r.computeDiversityNeed(rerankCtx.SessionHistory)
	}

	// Query features
	features["query_length"] = float64(len(rerankCtx.Query))
	features["query_complexity"] = r.estimateQueryComplexity(rerankCtx.Query)

	return features
}

func (r *contextualReranker) UpdateModel(ctx context.Context, feedback []*Interaction) error {
	// Generate learning signals from feedback
	signals, err := r.learningEngine.GenerateLearningSignals(ctx, feedback)
	if err != nil {
		return err
	}

	// Update feature weights using gradient descent
	r.updateFeatureWeights(signals)

	// Update click model
	if err := r.learningEngine.UpdateClickModel(ctx, feedback); err != nil {
		return err
	}

	// Update position bias model
	r.updatePositionBias(feedback)

	return nil
}

func (r *contextualReranker) applyPersonalization(score float64, result SearchResult, profile *UserProfile) float64 {
	// Apply topic preferences
	if topics, ok := result.Metadata["topics"].([]string); ok {
		for _, topic := range topics {
			if interest, exists := profile.TopicInterests[topic]; exists {
				score *= (1.0 + interest*0.2) // Up to 20% boost per topic
			}
		}
	}

	// Apply entity preferences
	if entities, ok := result.Metadata["entities"].([]string); ok {
		for _, entity := range entities {
			if interest, exists := profile.EntityInterests[entity]; exists {
				score *= (1.0 + interest*0.1) // Up to 10% boost per entity
			}
		}
	}

	// Apply source preferences
	if source, ok := result.Metadata["source"].(string); ok {
		if weight, exists := profile.Preferences.SourceWeights[source]; exists {
			score *= weight
		}
	}

	return score
}

func (r *contextualReranker) applyTemporalBoost(score float64, result SearchResult) float64 {
	if created, ok := result.Metadata["created_at"].(time.Time); ok {
		age := time.Since(created).Hours() / 24.0 // Age in days
		temporalBoost := math.Exp(-age / 30.0)    // Exponential decay over 30 days
		score *= (0.8 + 0.2*temporalBoost)        // 80% base + up to 20% boost
	}
	return score
}

func (r *contextualReranker) applyDiversityPenalty(score float64, result SearchResult, previousResults []SearchResult) float64 {
	if len(previousResults) == 0 {
		return score
	}

	// Check similarity with previous results
	maxSimilarity := 0.0
	resultTopics, _ := result.Metadata["topics"].([]string)

	for _, prev := range previousResults {
		prevTopics, _ := prev.Metadata["topics"].([]string)
		similarity := r.computeTopicSimilarity(resultTopics, prevTopics)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
		}
	}

	// Apply penalty for high similarity
	diversityPenalty := 1.0 - maxSimilarity*0.3 // Up to 30% penalty
	return score * diversityPenalty
}

func (r *contextualReranker) computeTopicRelevance(topics []string, profile *UserProfile) float64 {
	if len(topics) == 0 {
		return 0.0
	}

	totalRelevance := 0.0
	for _, topic := range topics {
		if interest, exists := profile.TopicInterests[topic]; exists {
			totalRelevance += interest
		}
	}

	return totalRelevance / float64(len(topics))
}

func (r *contextualReranker) computeTopicSimilarity(topics1, topics2 []string) float64 {
	if len(topics1) == 0 || len(topics2) == 0 {
		return 0.0
	}

	// Convert to sets
	set1 := make(map[string]bool)
	for _, t := range topics1 {
		set1[t] = true
	}

	// Count intersection
	intersection := 0
	for _, t := range topics2 {
		if set1[t] {
			intersection++
		}
	}

	// Jaccard similarity
	union := len(set1) + len(topics2) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

func (r *contextualReranker) getSourceAuthority(source string) float64 {
	// Hardcoded authority scores for demo
	// In production, this would be learned from data
	authorities := map[string]float64{
		"wikipedia":     0.9,
		"arxiv":         0.85,
		"github":        0.8,
		"stackoverflow": 0.75,
		"medium":        0.6,
		"blog":          0.5,
	}

	if auth, exists := authorities[source]; exists {
		return auth
	}
	return 0.5 // Default authority
}

func (r *contextualReranker) getUserDocumentClicks(userID, documentID string) float64 {
	// Query historical clicks
	filter := InteractionFilter{
		UserID:     userID,
		DocumentID: documentID,
		Type:       InteractionTypeClick,
	}

	interactions, err := r.collector.GetInteractions(context.Background(), filter)
	if err != nil {
		return 0.0
	}

	return float64(len(interactions))
}

func (r *contextualReranker) getUserDocumentDwellTime(userID, documentID string) float64 {
	filter := InteractionFilter{
		UserID:     userID,
		DocumentID: documentID,
		Type:       InteractionTypeDwell,
	}

	interactions, err := r.collector.GetInteractions(context.Background(), filter)
	if err != nil {
		return 0.0
	}

	totalDwell := 0.0
	for _, interaction := range interactions {
		totalDwell += interaction.Value
	}

	if len(interactions) > 0 {
		return totalDwell / float64(len(interactions))
	}
	return 0.0
}

func (r *contextualReranker) computeSessionConsistency(result SearchResult, history []*Interaction) float64 {
	// Measure how consistent this result is with the session history
	// Higher score means more consistent
	if len(history) == 0 {
		return 0.5 // Neutral if no history
	}

	resultTopics, ok := result.Metadata["topics"].([]string)
	if !ok || len(resultTopics) == 0 {
		return 0.5 // Neutral if no topics
	}

	// Collect all topics from session history
	historyTopics := make(map[string]bool)
	for _, interaction := range history {
		if interaction.Metadata != nil {
			if topics, exists := interaction.Metadata["topics"]; exists {
				if topicList, ok := topics.([]string); ok {
					for _, topic := range topicList {
						historyTopics[topic] = true
					}
				}
			}
		}
	}

	if len(historyTopics) == 0 {
		return 0.5 // Neutral if no history topics
	}

	// Calculate overlap between result topics and history topics
	matchingTopics := 0
	for _, topic := range resultTopics {
		if historyTopics[topic] {
			matchingTopics++
		}
	}

	// Return consistency score (0.0 to 1.0)
	if len(resultTopics) == 0 {
		return 0.5
	}

	consistency := float64(matchingTopics) / float64(len(resultTopics))
	// Boost consistency to make it more influential in ranking
	return 0.3 + (consistency * 0.7) // Scale from 0.3 to 1.0
}

func (r *contextualReranker) computeDiversityNeed(history []*Interaction) float64 {
	// Estimate how much diversity is needed based on session history
	// Higher score means more diversity needed
	if len(history) > 5 {
		return 0.8 // Need more diversity after many interactions
	}
	return 0.2
}

func (r *contextualReranker) estimateQueryComplexity(query string) float64 {
	// Simple complexity estimation based on query characteristics
	words := len(query) / 5 // Rough word count
	return math.Min(float64(words)/10.0, 1.0)
}

func (r *contextualReranker) updateFeatureWeights(signals []*LearningSignal) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Simple gradient descent update
	learningRate := 0.01

	for _, signal := range signals {
		for feature, value := range signal.Features {
			gradient := (signal.Label - r.predictScore(signal.Features)) * value
			r.featureWeights[feature] += learningRate * gradient * signal.Weight
		}
	}
}

func (r *contextualReranker) updatePositionBias(feedback []*Interaction) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Update position bias based on click patterns
	positionClicks := make(map[int]int)
	positionViews := make(map[int]int)

	for _, interaction := range feedback {
		pos := interaction.Position
		positionViews[pos]++
		if interaction.Type == InteractionTypeClick {
			positionClicks[pos]++
		}
	}

	// Update bias estimates
	for pos := 1; pos <= 20; pos++ {
		if views := positionViews[pos]; views > 0 {
			ctr := float64(positionClicks[pos]) / float64(views)
			// Smooth update
			if pos <= len(r.positionBias) {
				r.positionBias[pos-1] = 0.9*r.positionBias[pos-1] + 0.1*ctr
			}
		}
	}
}

func (r *contextualReranker) predictScore(features map[string]float64) float64 {
	score := 0.0
	for feature, value := range features {
		if weight, exists := r.featureWeights[feature]; exists {
			score += weight * value
		}
	}
	return score
}

func initializeFeatureWeights() map[string]float64 {
	return map[string]float64{
		"original_score":            1.0,
		"vector_score":              0.5,
		"text_score":                0.5,
		"score_variance":            -0.1,
		"document_popularity":       0.2,
		"document_rating":           0.3,
		"document_freshness":        0.15,
		"source_authority":          0.25,
		"topic_relevance":           0.4,
		"user_doc_clicks":           0.3,
		"user_doc_dwell":            0.2,
		"session_topic_consistency": 0.2,
		"session_diversity_needed":  0.1,
		"query_complexity":          0.1,
	}
}

func initializePositionBias() []float64 {
	// Initialize with typical position bias curve
	bias := make([]float64, 20)
	for i := 0; i < 20; i++ {
		bias[i] = 1.0 / math.Pow(float64(i+1), 0.5)
	}
	return bias
}
