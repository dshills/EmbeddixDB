package feedback

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// simpleLearningEngine implements LearningEngine with basic ML algorithms
type simpleLearningEngine struct {
	mu              sync.RWMutex
	clickModel      *ClickModel
	relevanceModel  map[string]float64 // Simple linear model weights
	trainingData    []*LearningSignal
	maxTrainingSize int
}

// NewSimpleLearningEngine creates a basic learning engine
func NewSimpleLearningEngine() LearningEngine {
	return &simpleLearningEngine{
		clickModel: &ClickModel{
			Type: "position_bias",
			Parameters: map[string]float64{
				"alpha": 1.0,
				"beta":  0.5,
			},
			LastUpdated: time.Now(),
		},
		relevanceModel:  make(map[string]float64),
		trainingData:    make([]*LearningSignal, 0),
		maxTrainingSize: 10000,
	}
}

func (e *simpleLearningEngine) GenerateLearningSignals(ctx context.Context, interactions []*Interaction) ([]*LearningSignal, error) {
	signals := make([]*LearningSignal, 0)

	// Group interactions by query
	queryGroups := make(map[string][]*Interaction)
	for _, interaction := range interactions {
		queryGroups[interaction.QueryID] = append(queryGroups[interaction.QueryID], interaction)
	}

	// Generate signals for each query group
	for queryID, queryInteractions := range queryGroups {
		// Sort by position
		sort.Slice(queryInteractions, func(i, j int) bool {
			return queryInteractions[i].Position < queryInteractions[j].Position
		})

		// Extract click pattern
		clickedPositions := make(map[int]bool)
		maxClickPosition := 0
		for _, interaction := range queryInteractions {
			if interaction.Type == InteractionTypeClick {
				clickedPositions[interaction.Position] = true
				if interaction.Position > maxClickPosition {
					maxClickPosition = interaction.Position
				}
			}
		}

		// Generate pairwise preferences
		for i := 0; i < len(queryInteractions); i++ {
			for j := i + 1; j < len(queryInteractions); j++ {
				doc1 := queryInteractions[i]
				doc2 := queryInteractions[j]

				// Skip if neither document was clicked
				clicked1 := clickedPositions[doc1.Position]
				clicked2 := clickedPositions[doc2.Position]
				if !clicked1 && !clicked2 {
					continue
				}

				// Generate preference signal
				var label float64
				if clicked1 && !clicked2 {
					label = 1.0
				} else if !clicked1 && clicked2 {
					label = 0.0
				} else {
					// Both clicked, use dwell time or position as tiebreaker
					if doc1.Value > doc2.Value {
						label = 1.0
					} else if doc1.Value < doc2.Value {
						label = 0.0
					} else {
						label = 0.5 // No preference
					}
				}

				// Extract features for the pair
				features := e.extractPairwiseFeatures(doc1, doc2)
				
				signal := &LearningSignal{
					Query:      doc1.Query,
					DocumentID: fmt.Sprintf("%s_vs_%s", doc1.DocumentID, doc2.DocumentID),
					Features:   features,
					Label:      label,
					Weight:     1.0,
					Timestamp:  time.Now(),
					Metadata: map[string]interface{}{
						"doc1_id": doc1.DocumentID,
						"doc2_id": doc2.DocumentID,
						"queryID": queryID,
					},
				}

				signals = append(signals, signal)
			}
		}
	}

	return signals, nil
}

func (e *simpleLearningEngine) UpdateClickModel(ctx context.Context, interactions []*Interaction) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Count clicks and impressions by position
	positionClicks := make(map[int]float64)
	positionImpressions := make(map[int]float64)

	for _, interaction := range interactions {
		pos := interaction.Position
		positionImpressions[pos]++
		
		if interaction.Type == InteractionTypeClick {
			positionClicks[pos]++
		}
	}

	// Update position bias model using Maximum Likelihood Estimation
	alpha := 0.0
	beta := 0.0
	n := 0

	for pos := 1; pos <= 20; pos++ {
		impressions := positionImpressions[pos]
		if impressions > 0 {
			clicks := positionClicks[pos]
			ctr := clicks / impressions
			
			// Log-likelihood contribution
			if ctr > 0 {
				alpha += math.Log(ctr) * clicks
				beta += math.Log(float64(pos)) * clicks
				n++
			}
		}
	}

	if n > 0 {
		// Update parameters with smoothing
		newAlpha := math.Exp(alpha / float64(n))
		newBeta := -beta / float64(n)
		
		e.clickModel.Parameters["alpha"] = 0.9*e.clickModel.Parameters["alpha"] + 0.1*newAlpha
		e.clickModel.Parameters["beta"] = 0.9*e.clickModel.Parameters["beta"] + 0.1*newBeta
	}

	e.clickModel.LastUpdated = time.Now()
	return nil
}

func (e *simpleLearningEngine) GetClickProbability(ctx context.Context, position int, features map[string]float64) (float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Position bias model: P(click) = alpha * position^(-beta)
	alpha := e.clickModel.Parameters["alpha"]
	beta := e.clickModel.Parameters["beta"]
	
	positionBias := alpha * math.Pow(float64(position), -beta)
	
	// Adjust by relevance features if available
	relevance := 0.5 // Default relevance
	if features != nil {
		relevance = e.computeRelevance(features)
	}
	
	// Combine position bias and relevance
	clickProbability := positionBias * relevance
	
	// Ensure probability is in [0, 1]
	return math.Min(math.Max(clickProbability, 0.0), 1.0), nil
}

func (e *simpleLearningEngine) GetRelevanceScore(ctx context.Context, query string, documentID string, features map[string]float64) (float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.computeRelevance(features), nil
}

func (e *simpleLearningEngine) TrainModel(ctx context.Context, signals []*LearningSignal) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Add new signals to training data
	e.trainingData = append(e.trainingData, signals...)
	
	// Limit training data size
	if len(e.trainingData) > e.maxTrainingSize {
		// Keep only recent data
		e.trainingData = e.trainingData[len(e.trainingData)-e.maxTrainingSize:]
	}

	// Train using stochastic gradient descent
	learningRate := 0.01
	regularization := 0.001
	epochs := 10

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		
		for _, signal := range e.trainingData {
			// Compute prediction
			prediction := e.computeRelevance(signal.Features)
			
			// Compute loss (squared error)
			loss := math.Pow(prediction-signal.Label, 2)
			totalLoss += loss
			
			// Update weights
			for feature, value := range signal.Features {
				gradient := 2 * (prediction - signal.Label) * value
				
				// Apply gradient with L2 regularization
				oldWeight := e.relevanceModel[feature]
				e.relevanceModel[feature] = oldWeight - learningRate*(gradient+regularization*oldWeight)
			}
		}
		
		// Decay learning rate
		learningRate *= 0.95
	}

	return nil
}

func (e *simpleLearningEngine) ExportModel(ctx context.Context) ([]byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	model := struct {
		ClickModel     *ClickModel        `json:"click_model"`
		RelevanceModel map[string]float64 `json:"relevance_model"`
		Timestamp      time.Time          `json:"timestamp"`
	}{
		ClickModel:     e.clickModel,
		RelevanceModel: e.relevanceModel,
		Timestamp:      time.Now(),
	}

	return json.Marshal(model)
}

func (e *simpleLearningEngine) ImportModel(ctx context.Context, modelData []byte) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var model struct {
		ClickModel     *ClickModel        `json:"click_model"`
		RelevanceModel map[string]float64 `json:"relevance_model"`
		Timestamp      time.Time          `json:"timestamp"`
	}

	if err := json.Unmarshal(modelData, &model); err != nil {
		return err
	}

	e.clickModel = model.ClickModel
	e.relevanceModel = model.RelevanceModel

	return nil
}

func (e *simpleLearningEngine) extractPairwiseFeatures(doc1, doc2 *Interaction) map[string]float64 {
	features := make(map[string]float64)

	// Position features
	features["pos_diff"] = float64(doc2.Position - doc1.Position)
	features["pos_ratio"] = float64(doc1.Position) / float64(doc2.Position)
	
	// Metadata features (if available)
	if doc1.Metadata != nil && doc2.Metadata != nil {
		// Score differences
		if score1, ok1 := doc1.Metadata["score"].(float64); ok1 {
			if score2, ok2 := doc2.Metadata["score"].(float64); ok2 {
				features["score_diff"] = score1 - score2
			}
		}
		
		// View count differences
		if views1, ok1 := doc1.Metadata["views"].(float64); ok1 {
			if views2, ok2 := doc2.Metadata["views"].(float64); ok2 {
				features["views_diff"] = math.Log1p(views1) - math.Log1p(views2)
			}
		}
	}

	return features
}

func (e *simpleLearningEngine) computeRelevance(features map[string]float64) float64 {
	score := 0.5 // Base relevance
	
	for feature, value := range features {
		if weight, exists := e.relevanceModel[feature]; exists {
			score += weight * value
		}
	}
	
	// Apply sigmoid to get probability
	return 1.0 / (1.0 + math.Exp(-score))
}

// feedbackAnalyzer implements FeedbackAnalyzer
type feedbackAnalyzer struct {
	collector Collector
}

// NewFeedbackAnalyzer creates a new feedback analyzer
func NewFeedbackAnalyzer(collector Collector) FeedbackAnalyzer {
	return &feedbackAnalyzer{
		collector: collector,
	}
}

func (a *feedbackAnalyzer) AnalyzeQuerySatisfaction(ctx context.Context, interactions []*Interaction) (satisfied bool, confidence float64) {
	if len(interactions) == 0 {
		return false, 0.0
	}

	// Satisfaction signals
	clickCount := 0
	totalDwellTime := 0.0
	maxClickPosition := 0
	hasNegativeFeedback := false

	for _, interaction := range interactions {
		switch interaction.Type {
		case InteractionTypeClick:
			clickCount++
			if interaction.Position > maxClickPosition {
				maxClickPosition = interaction.Position
			}
		case InteractionTypeDwell:
			totalDwellTime += interaction.Value
		case InteractionTypeRating:
			if interaction.Value < 3.0 {
				hasNegativeFeedback = true
			}
		case InteractionTypeNegative:
			hasNegativeFeedback = true
		}
	}

	// Compute satisfaction score
	satisfactionScore := 0.0
	confidence = 0.0

	// Click-based satisfaction
	if clickCount > 0 {
		// Clicks on top results indicate satisfaction
		if maxClickPosition <= 3 {
			satisfactionScore += 0.4
			confidence += 0.3
		} else if maxClickPosition <= 5 {
			satisfactionScore += 0.2
			confidence += 0.2
		}
		
		// Multiple clicks might indicate exploration
		if clickCount >= 3 {
			satisfactionScore -= 0.1
		}
	}

	// Dwell time satisfaction  
	dwellCount := 0
	for _, interaction := range interactions {
		if interaction.Type == InteractionTypeDwell {
			dwellCount++
		}
	}
	
	if dwellCount > 0 {
		avgDwellTime := totalDwellTime / float64(dwellCount)
		if avgDwellTime > 30 {
			satisfactionScore += 0.3
			confidence += 0.3
		} else if avgDwellTime > 10 {
			satisfactionScore += 0.1
			confidence += 0.2
		}
	}

	// Negative feedback override
	if hasNegativeFeedback {
		satisfactionScore = 0.0
		confidence = 1.0
	}

	satisfied = satisfactionScore > 0.3
	confidence = math.Min(confidence, 1.0)
	
	return satisfied, confidence
}

func (a *feedbackAnalyzer) AnalyzeClickPattern(ctx context.Context, interactions []*Interaction) (pattern map[string]interface{}, err error) {
	pattern = make(map[string]interface{})

	// Extract click interactions
	var clicks []*Interaction
	for _, interaction := range interactions {
		if interaction.Type == InteractionTypeClick {
			clicks = append(clicks, interaction)
		}
	}

	if len(clicks) == 0 {
		pattern["pattern_type"] = "no_clicks"
		return pattern, nil
	}

	// Sort by position
	sort.Slice(clicks, func(i, j int) bool {
		return clicks[i].Position < clicks[j].Position
	})

	// Analyze patterns
	positions := make([]int, len(clicks))
	for i, click := range clicks {
		positions[i] = click.Position
	}

	// Check for position bias
	avgPosition := 0.0
	for _, pos := range positions {
		avgPosition += float64(pos)
	}
	avgPosition /= float64(len(positions))

	pattern["avg_click_position"] = avgPosition
	pattern["click_count"] = len(clicks)
	pattern["positions"] = positions

	// Determine pattern type
	if avgPosition <= 2.0 {
		pattern["pattern_type"] = "top_heavy"
		pattern["position_bias"] = "high"
	} else if avgPosition <= 5.0 {
		pattern["pattern_type"] = "moderate"
		pattern["position_bias"] = "medium"
	} else {
		pattern["pattern_type"] = "exploratory"
		pattern["position_bias"] = "low"
	}

	// Check for cascading pattern
	cascading := true
	for i := 1; i < len(positions); i++ {
		if positions[i] < positions[i-1] {
			cascading = false
			break
		}
	}
	pattern["cascading"] = cascading

	return pattern, nil
}

func (a *feedbackAnalyzer) ComputeDocumentQuality(ctx context.Context, feedback *DocumentFeedback) (quality float64, signals map[string]float64) {
	signals = make(map[string]float64)

	// CTR signal
	if feedback.TotalViews > 0 {
		signals["ctr"] = feedback.ClickThroughRate
		quality += feedback.ClickThroughRate * 0.3
	}

	// Rating signal
	if feedback.TotalRatings > 0 {
		normalizedRating := feedback.AvgRating / 5.0
		signals["rating"] = normalizedRating
		quality += normalizedRating * 0.4
	}

	// Dwell time signal
	if feedback.AvgDwellTime > 0 {
		// Normalize dwell time (assume 60s is good)
		normalizedDwell := math.Min(feedback.AvgDwellTime/60.0, 1.0)
		signals["dwell"] = normalizedDwell
		quality += normalizedDwell * 0.3
	}

	// Ensure quality is in [0, 1]
	quality = math.Min(math.Max(quality, 0.0), 1.0)

	return quality, signals
}

func (a *feedbackAnalyzer) DetectAnomalies(ctx context.Context, interactions []*Interaction) (anomalies []string, err error) {
	anomalies = make([]string, 0)

	if len(interactions) == 0 {
		return anomalies, nil
	}

	// Group by user
	userInteractions := make(map[string][]*Interaction)
	for _, interaction := range interactions {
		userInteractions[interaction.UserID] = append(userInteractions[interaction.UserID], interaction)
	}

	// Check for anomalous patterns
	for userID, userInts := range userInteractions {
		// Rapid clicking (potential bot)
		clickTimes := make([]time.Time, 0)
		for _, interaction := range userInts {
			if interaction.Type == InteractionTypeClick {
				clickTimes = append(clickTimes, interaction.Timestamp)
			}
		}
		
		if len(clickTimes) > 10 {
			// Check click rate
			duration := clickTimes[len(clickTimes)-1].Sub(clickTimes[0])
			clickRate := float64(len(clickTimes)) / duration.Seconds()
			
			if clickRate > 1.0 { // More than 1 click per second
				anomalies = append(anomalies, fmt.Sprintf("user:%s:rapid_clicking", userID))
			}
		}

		// All negative ratings (potential abuse)
		negativeCount := 0
		for _, interaction := range userInts {
			if interaction.Type == InteractionTypeRating && interaction.Value < 2.0 {
				negativeCount++
			}
		}
		
		if negativeCount > 5 {
			anomalies = append(anomalies, fmt.Sprintf("user:%s:excessive_negative_ratings", userID))
		}
	}

	return anomalies, nil
}