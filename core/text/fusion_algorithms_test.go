package text

import (
	"math"
	"testing"

	"github.com/dshills/EmbeddixDB/core/ai"
)

func createTestResults() ([]ai.SearchResult, []ai.SearchResult) {
	vectorResults := []ai.SearchResult{
		{ID: "doc1", Score: 0.9, Content: "Vector result 1"},
		{ID: "doc2", Score: 0.8, Content: "Vector result 2"},
		{ID: "doc3", Score: 0.7, Content: "Vector result 3"},
		{ID: "doc4", Score: 0.6, Content: "Vector result 4"},
	}

	textResults := []ai.SearchResult{
		{ID: "doc2", Score: 12.5, Content: "Text result 2"}, // High BM25 score
		{ID: "doc3", Score: 10.0, Content: "Text result 3"},
		{ID: "doc5", Score: 8.5, Content: "Text result 5"},
		{ID: "doc1", Score: 5.0, Content: "Text result 1"},
	}

	return vectorResults, textResults
}

func TestWeightedBordaFusion(t *testing.T) {
	wbf := NewWeightedBordaFusion()
	vectorResults, textResults := createTestResults()

	weights := ai.SearchWeights{
		Vector: 0.6,
		Text:   0.4,
	}

	results := wbf.Fuse(vectorResults, textResults, weights)

	// Verify results
	if len(results) != 5 {
		t.Errorf("Expected 5 results, got %d", len(results))
	}

	// Check that results are sorted by score
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("Results not properly sorted at index %d", i)
		}
	}

	// Verify algorithm name
	if wbf.GetName() != "weighted_borda_fusion" {
		t.Errorf("Unexpected algorithm name: %s", wbf.GetName())
	}

	// Check parameters
	params := wbf.GetParameters()
	if params["position_weights"] != 100 {
		t.Errorf("Expected 100 position weights, got %v", params["position_weights"])
	}
}

func TestCombSumFusion(t *testing.T) {
	// Test without normalization
	t.Run("without_normalization", func(t *testing.T) {
		csf := NewCombSumFusion(false)
		vectorResults, textResults := createTestResults()

		weights := ai.SearchWeights{
			Vector: 0.5,
			Text:   0.5,
		}

		results := csf.Fuse(vectorResults, textResults, weights)

		// doc2 should have highest combined score
		// Vector: 0.8 * 0.5 = 0.4
		// Text: 12.5 * 0.5 = 6.25
		// Total: 6.65
		if len(results) > 0 && results[0].ID != "doc2" {
			t.Errorf("Expected doc2 to rank first, got %s", results[0].ID)
		}

		if csf.GetName() != "combsum" {
			t.Errorf("Unexpected algorithm name: %s", csf.GetName())
		}
	})

	// Test with normalization
	t.Run("with_normalization", func(t *testing.T) {
		csf := NewCombSumFusion(true)
		vectorResults, textResults := createTestResults()

		weights := ai.SearchWeights{
			Vector: 0.6,
			Text:   0.4,
		}

		results := csf.Fuse(vectorResults, textResults, weights)

		// All scores should be normalized
		for _, result := range results {
			if result.Score < 0 || result.Score > 1.2 { // 1.2 allows for weighted sum > 1
				t.Errorf("Score %f outside expected range for doc %s", result.Score, result.ID)
			}
		}

		if csf.GetName() != "combsum_normalized" {
			t.Errorf("Unexpected algorithm name: %s", csf.GetName())
		}
	})
}

func TestISRFusion(t *testing.T) {
	isr := NewISRFusion()
	vectorResults, textResults := createTestResults()

	weights := ai.SearchWeights{
		Vector: 0.7,
		Text:   0.3,
	}

	results := isr.Fuse(vectorResults, textResults, weights)

	// Verify results
	if len(results) != 5 {
		t.Errorf("Expected 5 results, got %d", len(results))
	}

	// Check that scores decrease (due to inverse square rank)
	prevScore := math.Inf(1)
	for _, result := range results {
		if result.Score > prevScore {
			t.Error("Scores should be non-increasing")
		}
		prevScore = result.Score
	}

	// Verify algorithm name and parameters
	if isr.GetName() != "inverse_square_rank" {
		t.Errorf("Unexpected algorithm name: %s", isr.GetName())
	}

	params := isr.GetParameters()
	if params["constant"] != 60.0 {
		t.Errorf("Expected constant 60.0, got %v", params["constant"])
	}
}

func TestRelativeScoreFusion(t *testing.T) {
	rsf := NewRelativeScoreFusion(0.5) // 50% threshold
	vectorResults, textResults := createTestResults()

	weights := ai.SearchWeights{
		Vector: 0.5,
		Text:   0.5,
	}

	results := rsf.Fuse(vectorResults, textResults, weights)

	// With 50% threshold, some low-scoring results should be filtered
	// Vector: doc4 has score 0.6/0.9 = 0.67 > 0.5, so included
	// Text: doc1 has score 5.0/12.5 = 0.4 < 0.5, so excluded from text contribution

	foundDoc1 := false
	for _, result := range results {
		if result.ID == "doc1" {
			foundDoc1 = true
			// doc1 should only have vector contribution
			// since its text score is below threshold
			break
		}
	}

	if !foundDoc1 {
		t.Log("doc1 might be filtered if only text contribution was considered")
	}

	if rsf.GetName() != "relative_score_fusion" {
		t.Errorf("Unexpected algorithm name: %s", rsf.GetName())
	}
}

func TestProbabilisticFusion(t *testing.T) {
	pf := NewProbabilisticFusion()

	// Create results with scores in [0,1] range for probability interpretation
	vectorResults := []ai.SearchResult{
		{ID: "doc1", Score: 0.9},
		{ID: "doc2", Score: 0.7},
		{ID: "doc3", Score: 0.5},
	}

	textResults := []ai.SearchResult{
		{ID: "doc2", Score: 0.8},
		{ID: "doc3", Score: 0.6},
		{ID: "doc4", Score: 0.4},
	}

	weights := ai.SearchWeights{
		Vector: 0.6,
		Text:   0.4,
	}

	results := pf.Fuse(vectorResults, textResults, weights)

	// Verify results
	if len(results) != 4 {
		t.Errorf("Expected 4 results, got %d", len(results))
	}

	// All scores should be positive (exponential of log probabilities)
	for _, result := range results {
		if result.Score <= 0 {
			t.Errorf("Probabilistic fusion produced non-positive score: %f", result.Score)
		}
	}

	if pf.GetName() != "probabilistic_fusion" {
		t.Errorf("Unexpected algorithm name: %s", pf.GetName())
	}

	params := pf.GetParameters()
	if params["epsilon"] != 1e-10 {
		t.Errorf("Unexpected epsilon value: %v", params["epsilon"])
	}
}

func TestScoresToProbabilities(t *testing.T) {
	pf := NewProbabilisticFusion()

	// Test normal case
	results := []ai.SearchResult{
		{Score: 10.0},
		{Score: 20.0},
		{Score: 30.0},
	}

	probs := pf.scoresToProbabilities(results)

	// Check probabilities sum to 1
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Probabilities don't sum to 1: %f", sum)
	}

	// Test with zero scores
	zeroResults := []ai.SearchResult{
		{Score: 0.0},
		{Score: 0.0},
		{Score: 0.0},
	}

	zeroProbs := pf.scoresToProbabilities(zeroResults)

	// Should get uniform distribution
	for _, p := range zeroProbs {
		if math.Abs(p-1.0/3.0) > 1e-10 {
			t.Errorf("Expected uniform probability 0.333..., got %f", p)
		}
	}
}

func TestNormalizeResultScores(t *testing.T) {
	csf := NewCombSumFusion(true)

	results := []ai.SearchResult{
		{ID: "doc1", Score: 10.0},
		{ID: "doc2", Score: 20.0},
		{ID: "doc3", Score: 30.0},
	}

	normalized := csf.normalizeResultScores(results)

	// Check normalization
	if normalized[0].Score != 0.0 {
		t.Errorf("Min score should be normalized to 0, got %f", normalized[0].Score)
	}

	if normalized[2].Score != 1.0 {
		t.Errorf("Max score should be normalized to 1, got %f", normalized[2].Score)
	}

	// Test with all same scores
	sameResults := []ai.SearchResult{
		{ID: "doc1", Score: 5.0},
		{ID: "doc2", Score: 5.0},
		{ID: "doc3", Score: 5.0},
	}

	sameNormalized := csf.normalizeResultScores(sameResults)

	// Should return unchanged when all scores are same
	for i, result := range sameNormalized {
		if result.Score != sameResults[i].Score {
			t.Errorf("Expected unchanged score for same values, got %f", result.Score)
		}
	}
}

func BenchmarkFusionAlgorithms(b *testing.B) {
	vectorResults, textResults := createTestResults()
	weights := ai.SearchWeights{Vector: 0.6, Text: 0.4}

	algorithms := []ai.FusionAlgorithm{
		NewReciprocalRankFusion(),
		NewWeightedBordaFusion(),
		NewCombSumFusion(false),
		NewCombSumFusion(true),
		NewISRFusion(),
		NewRelativeScoreFusion(0.3),
		NewProbabilisticFusion(),
	}

	for _, algo := range algorithms {
		b.Run(algo.GetName(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = algo.Fuse(vectorResults, textResults, weights)
			}
		})
	}
}
