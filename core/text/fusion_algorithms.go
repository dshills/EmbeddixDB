package text

import (
	"math"
	"sort"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// WeightedBordaFusion implements the Weighted Borda Count fusion algorithm
type WeightedBordaFusion struct {
	positionWeights []float64 // Weight for each rank position
}

// NewWeightedBordaFusion creates a new Weighted Borda Count fusion algorithm
func NewWeightedBordaFusion() *WeightedBordaFusion {
	// Default exponential decay weights
	weights := make([]float64, 100)
	for i := 0; i < 100; i++ {
		weights[i] = math.Exp(-0.1 * float64(i))
	}

	return &WeightedBordaFusion{
		positionWeights: weights,
	}
}

// Fuse combines results using Weighted Borda Count
func (wbf *WeightedBordaFusion) Fuse(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Calculate Borda scores
	docScores := make(map[string]float64)
	docContent := make(map[string]*ai.SearchResult)

	// Process vector results
	for rank, result := range vectorResults {
		weight := weights.Vector
		if rank < len(wbf.positionWeights) {
			weight *= wbf.positionWeights[rank]
		}
		docScores[result.ID] = weight
		r := result
		docContent[result.ID] = &r
	}

	// Process text results
	for rank, result := range textResults {
		weight := weights.Text
		if rank < len(wbf.positionWeights) {
			weight *= wbf.positionWeights[rank]
		}

		if existing, exists := docScores[result.ID]; exists {
			docScores[result.ID] = existing + weight
			// Update content if missing
			if docContent[result.ID].Content == "" {
				docContent[result.ID].Content = result.Content
			}
		} else {
			docScores[result.ID] = weight
			r := result
			docContent[result.ID] = &r
		}
	}

	// Convert to results and sort
	results := make([]ai.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		result := *docContent[docID]
		result.Score = score
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// GetName returns the algorithm name
func (wbf *WeightedBordaFusion) GetName() string {
	return "weighted_borda_fusion"
}

// GetParameters returns algorithm parameters
func (wbf *WeightedBordaFusion) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"position_weights": len(wbf.positionWeights),
		"decay_function":   "exponential",
	}
}

// CombSumFusion implements the CombSUM fusion algorithm with normalization
type CombSumFusion struct {
	normalizeScores bool
}

// NewCombSumFusion creates a new CombSUM fusion algorithm
func NewCombSumFusion(normalize bool) *CombSumFusion {
	return &CombSumFusion{
		normalizeScores: normalize,
	}
}

// Fuse combines results using CombSUM
func (csf *CombSumFusion) Fuse(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Normalize scores if requested
	if csf.normalizeScores {
		vectorResults = csf.normalizeResultScores(vectorResults)
		textResults = csf.normalizeResultScores(textResults)
	}

	// Combine scores
	docScores := make(map[string]float64)
	docContent := make(map[string]*ai.SearchResult)

	// Add vector scores
	for _, result := range vectorResults {
		docScores[result.ID] = result.Score * weights.Vector
		r := result
		docContent[result.ID] = &r
	}

	// Add text scores
	for _, result := range textResults {
		if existing, exists := docScores[result.ID]; exists {
			docScores[result.ID] = existing + (result.Score * weights.Text)
			if docContent[result.ID].Content == "" {
				docContent[result.ID].Content = result.Content
			}
		} else {
			docScores[result.ID] = result.Score * weights.Text
			r := result
			docContent[result.ID] = &r
		}
	}

	// Convert to results
	results := make([]ai.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		result := *docContent[docID]
		result.Score = score
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// normalizeResultScores normalizes scores to [0, 1] range
func (csf *CombSumFusion) normalizeResultScores(results []ai.SearchResult) []ai.SearchResult {
	if len(results) == 0 {
		return results
	}

	// Find min and max scores
	minScore := results[0].Score
	maxScore := results[0].Score

	for _, result := range results {
		if result.Score < minScore {
			minScore = result.Score
		}
		if result.Score > maxScore {
			maxScore = result.Score
		}
	}

	// Avoid division by zero
	scoreRange := maxScore - minScore
	if scoreRange == 0 {
		return results
	}

	// Normalize scores
	normalized := make([]ai.SearchResult, len(results))
	for i, result := range results {
		normalized[i] = result
		normalized[i].Score = (result.Score - minScore) / scoreRange
	}

	return normalized
}

// GetName returns the algorithm name
func (csf *CombSumFusion) GetName() string {
	if csf.normalizeScores {
		return "combsum_normalized"
	}
	return "combsum"
}

// GetParameters returns algorithm parameters
func (csf *CombSumFusion) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"normalize_scores": csf.normalizeScores,
	}
}

// ISRFusion implements the Inverse Square Rank fusion algorithm
type ISRFusion struct {
	constant float64
}

// NewISRFusion creates a new ISR fusion algorithm
func NewISRFusion() *ISRFusion {
	return &ISRFusion{
		constant: 60.0, // Similar to RRF default
	}
}

// Fuse combines results using Inverse Square Rank
func (isr *ISRFusion) Fuse(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Calculate ISR scores
	docScores := make(map[string]float64)
	docContent := make(map[string]*ai.SearchResult)

	// Process vector results
	for rank, result := range vectorResults {
		score := weights.Vector / math.Pow(float64(rank+1)+isr.constant, 2)
		docScores[result.ID] = score
		r := result
		docContent[result.ID] = &r
	}

	// Process text results
	for rank, result := range textResults {
		score := weights.Text / math.Pow(float64(rank+1)+isr.constant, 2)

		if existing, exists := docScores[result.ID]; exists {
			docScores[result.ID] = existing + score
			if docContent[result.ID].Content == "" {
				docContent[result.ID].Content = result.Content
			}
		} else {
			docScores[result.ID] = score
			r := result
			docContent[result.ID] = &r
		}
	}

	// Convert to results
	results := make([]ai.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		result := *docContent[docID]
		result.Score = score
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// GetName returns the algorithm name
func (isr *ISRFusion) GetName() string {
	return "inverse_square_rank"
}

// GetParameters returns algorithm parameters
func (isr *ISRFusion) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"constant": isr.constant,
	}
}

// RelativeScoreFusion implements fusion based on relative score differences
type RelativeScoreFusion struct {
	threshold float64 // Minimum relative score to include
}

// NewRelativeScoreFusion creates a new relative score fusion algorithm
func NewRelativeScoreFusion(threshold float64) *RelativeScoreFusion {
	return &RelativeScoreFusion{
		threshold: threshold,
	}
}

// Fuse combines results based on relative scores
func (rsf *RelativeScoreFusion) Fuse(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Get max scores for each result set
	maxVectorScore := float64(0)
	if len(vectorResults) > 0 {
		maxVectorScore = vectorResults[0].Score
	}

	maxTextScore := float64(0)
	if len(textResults) > 0 {
		maxTextScore = textResults[0].Score
	}

	// Combine results with relative scoring
	docScores := make(map[string]float64)
	docContent := make(map[string]*ai.SearchResult)

	// Process vector results
	for _, result := range vectorResults {
		if maxVectorScore > 0 {
			relativeScore := result.Score / maxVectorScore
			if relativeScore >= rsf.threshold {
				docScores[result.ID] = relativeScore * weights.Vector
				r := result
				docContent[result.ID] = &r
			}
		}
	}

	// Process text results
	for _, result := range textResults {
		if maxTextScore > 0 {
			relativeScore := result.Score / maxTextScore
			if relativeScore >= rsf.threshold {
				score := relativeScore * weights.Text
				if existing, exists := docScores[result.ID]; exists {
					docScores[result.ID] = existing + score
					if docContent[result.ID].Content == "" {
						docContent[result.ID].Content = result.Content
					}
				} else {
					docScores[result.ID] = score
					r := result
					docContent[result.ID] = &r
				}
			}
		}
	}

	// Convert to results
	results := make([]ai.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		result := *docContent[docID]
		result.Score = score
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// GetName returns the algorithm name
func (rsf *RelativeScoreFusion) GetName() string {
	return "relative_score_fusion"
}

// GetParameters returns algorithm parameters
func (rsf *RelativeScoreFusion) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"threshold": rsf.threshold,
	}
}

// ProbabilisticFusion implements fusion based on probability theory
type ProbabilisticFusion struct {
	epsilon float64 // Small constant to avoid log(0)
}

// NewProbabilisticFusion creates a new probabilistic fusion algorithm
func NewProbabilisticFusion() *ProbabilisticFusion {
	return &ProbabilisticFusion{
		epsilon: 1e-10,
	}
}

// Fuse combines results using probabilistic fusion
func (pf *ProbabilisticFusion) Fuse(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Convert scores to probabilities (assuming scores are similarities in [0,1])
	vectorProbs := pf.scoresToProbabilities(vectorResults)
	textProbs := pf.scoresToProbabilities(textResults)

	// Combine probabilities
	docProbs := make(map[string]float64)
	docContent := make(map[string]*ai.SearchResult)

	// Process vector results
	for i, result := range vectorResults {
		prob := vectorProbs[i]
		// Use log probabilities to avoid underflow
		logProb := math.Log(prob + pf.epsilon)
		docProbs[result.ID] = logProb * weights.Vector
		r := result
		docContent[result.ID] = &r
	}

	// Process text results
	for i, result := range textResults {
		prob := textProbs[i]
		logProb := math.Log(prob + pf.epsilon)

		if existing, exists := docProbs[result.ID]; exists {
			// Combine log probabilities
			docProbs[result.ID] = existing + (logProb * weights.Text)
			if docContent[result.ID].Content == "" {
				docContent[result.ID].Content = result.Content
			}
		} else {
			docProbs[result.ID] = logProb * weights.Text
			r := result
			docContent[result.ID] = &r
		}
	}

	// Convert back from log probabilities and create results
	results := make([]ai.SearchResult, 0, len(docProbs))
	for docID, logProb := range docProbs {
		result := *docContent[docID]
		// Convert back from log probability
		result.Score = math.Exp(logProb)
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// scoresToProbabilities converts scores to probability distribution
func (pf *ProbabilisticFusion) scoresToProbabilities(results []ai.SearchResult) []float64 {
	if len(results) == 0 {
		return []float64{}
	}

	// Calculate sum of scores
	sum := float64(0)
	for _, result := range results {
		sum += result.Score
	}

	// Convert to probabilities
	probs := make([]float64, len(results))
	if sum > 0 {
		for i, result := range results {
			probs[i] = result.Score / sum
		}
	} else {
		// Uniform distribution if all scores are 0
		uniformProb := 1.0 / float64(len(results))
		for i := range probs {
			probs[i] = uniformProb
		}
	}

	return probs
}

// GetName returns the algorithm name
func (pf *ProbabilisticFusion) GetName() string {
	return "probabilistic_fusion"
}

// GetParameters returns algorithm parameters
func (pf *ProbabilisticFusion) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"epsilon": pf.epsilon,
	}
}
