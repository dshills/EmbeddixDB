package feedback

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestContextualReranker(t *testing.T) {
	ctx := context.Background()

	// Create test components
	collector := NewMemoryCollector()
	profileManager := NewMemoryProfileManager()
	learningEngine := NewSimpleLearningEngine()

	reranker := NewContextualReranker(collector, profileManager, learningEngine)

	t.Run("GetFeatures", func(t *testing.T) {
		result := SearchResult{
			ID:          "doc1",
			Score:       0.8,
			VectorScore: 0.85,
			TextScore:   0.75,
			Position:    2,
			Metadata: map[string]interface{}{
				"view_count": 100.0,
				"avg_rating": 4.5,
				"created_at": time.Now().Add(-24 * time.Hour),
				"source":     "wikipedia",
				"topics":     []string{"ai", "machine_learning"},
			},
			CollectionID: "collection1",
		}

		rerankCtx := RerankingContext{
			UserID:    "user1",
			SessionID: "session1",
			Query:     "machine learning tutorial",
			Timestamp: time.Now(),
		}

		features := reranker.GetFeatures(result, rerankCtx)

		// Verify basic features
		assert.Equal(t, 0.8, features["original_score"])
		assert.Equal(t, 0.85, features["vector_score"])
		assert.Equal(t, 0.75, features["text_score"])
		assert.InDelta(t, 0.1, features["score_variance"], 0.0001) // Use tolerance for floating point comparison
		assert.Equal(t, 2.0, features["original_position"])
		assert.Equal(t, 0.5, features["inverse_position"])

		// Verify metadata features
		assert.Greater(t, features["document_popularity"], 0.0)
		assert.Equal(t, 4.5, features["document_rating"])
		assert.Greater(t, features["document_freshness"], 0.0)
		assert.Equal(t, 0.9, features["source_authority"]) // wikipedia

		// Verify query features
		assert.Equal(t, float64(len("machine learning tutorial")), features["query_length"])
		assert.Greater(t, features["query_complexity"], 0.0)
	})

	t.Run("Rerank_BasicScoring", func(t *testing.T) {
		results := []SearchResult{
			{
				ID:          "doc1",
				Score:       0.6,
				VectorScore: 0.6,
				Position:    1,
				Metadata: map[string]interface{}{
					"view_count": 50.0,
					"avg_rating": 3.0,
				},
			},
			{
				ID:          "doc2",
				Score:       0.5,
				VectorScore: 0.5,
				Position:    2,
				Metadata: map[string]interface{}{
					"view_count": 200.0,
					"avg_rating": 4.8,
					"source":     "arxiv",
				},
			},
			{
				ID:          "doc3",
				Score:       0.55,
				VectorScore: 0.55,
				Position:    3,
				Metadata: map[string]interface{}{
					"view_count": 10.0,
					"avg_rating": 4.0,
					"created_at": time.Now(), // Very fresh
				},
			},
		}

		rerankCtx := RerankingContext{
			UserID:    "user1",
			SessionID: "session1",
			Query:     "test query",
			Timestamp: time.Now(),
		}

		reranked, err := reranker.Rerank(ctx, results, rerankCtx)
		assert.NoError(t, err)
		assert.Len(t, reranked, 3)

		// Verify positions are updated
		for i, result := range reranked {
			assert.Equal(t, i+1, result.Position)
		}

		// Doc2 should rank higher due to better metadata
		assert.NotEqual(t, results[0].ID, reranked[0].ID)
	})

	t.Run("Rerank_WithUserProfile", func(t *testing.T) {
		// Create a user profile with preferences
		profile, err := profileManager.CreateProfile(ctx, "user2")
		require.NoError(t, err)

		// Set topic interests
		topics := map[string]float64{
			"ai":         0.9,
			"healthcare": 0.2,
		}
		err = profileManager.IncrementInterests(ctx, "user2", topics, nil)
		require.NoError(t, err)

		results := []SearchResult{
			{
				ID:       "doc1",
				Score:    0.7,
				Position: 1,
				Metadata: map[string]interface{}{
					"topics": []string{"healthcare", "medicine"},
				},
			},
			{
				ID:       "doc2",
				Score:    0.7,
				Position: 2,
				Metadata: map[string]interface{}{
					"topics": []string{"ai", "machine_learning"},
				},
			},
		}

		profile, _ = profileManager.GetProfile(ctx, "user2")
		rerankCtx := RerankingContext{
			UserID:      "user2",
			SessionID:   "session2",
			Query:       "research papers",
			Timestamp:   time.Now(),
			UserProfile: profile,
		}

		reranked, err := reranker.Rerank(ctx, results, rerankCtx)
		assert.NoError(t, err)

		// Doc2 should rank higher due to user's AI interest
		assert.Equal(t, "doc2", reranked[0].ID)
		assert.Greater(t, reranked[0].Score, reranked[1].Score)
	})

	t.Run("Rerank_WithSessionHistory", func(t *testing.T) {
		// Create session history showing interest in AI topics
		sessionHistory := []*Interaction{
			{
				UserID:     "user3",
				SessionID:  "session3",
				DocumentID: "prev_doc1",
				Type:       InteractionTypeClick,
				Metadata: map[string]interface{}{
					"topics": []string{"ai", "deep_learning"},
				},
			},
			{
				UserID:     "user3",
				SessionID:  "session3",
				DocumentID: "prev_doc2",
				Type:       InteractionTypeClick,
				Metadata: map[string]interface{}{
					"topics": []string{"neural_networks"},
				},
			},
		}

		results := []SearchResult{
			{
				ID:       "doc1",
				Score:    0.6,
				Position: 1,
				Metadata: map[string]interface{}{
					"topics": []string{"databases", "sql"},
				},
			},
			{
				ID:       "doc2",
				Score:    0.6,
				Position: 2,
				Metadata: map[string]interface{}{
					"topics": []string{"ai", "neural_networks"},
				},
			},
		}

		rerankCtx := RerankingContext{
			UserID:         "user3",
			SessionID:      "session3",
			Query:          "technology",
			Timestamp:      time.Now(),
			SessionHistory: sessionHistory,
		}

		reranked, err := reranker.Rerank(ctx, results, rerankCtx)
		assert.NoError(t, err)

		// Doc2 should rank higher due to session consistency
		assert.Equal(t, "doc2", reranked[0].ID)
	})

	t.Run("Rerank_DiversityPenalty", func(t *testing.T) {
		results := []SearchResult{
			{
				ID:       "doc1",
				Score:    0.9,
				Position: 1,
				Metadata: map[string]interface{}{
					"topics": []string{"python", "programming"},
				},
			},
			{
				ID:       "doc2",
				Score:    0.85,
				Position: 2,
				Metadata: map[string]interface{}{
					"topics": []string{"python", "programming"}, // Same topics
				},
			},
			{
				ID:       "doc3",
				Score:    0.83,
				Position: 3,
				Metadata: map[string]interface{}{
					"topics": []string{"java", "programming"}, // Different topic
				},
			},
		}

		rerankCtx := RerankingContext{
			UserID:    "user4",
			SessionID: "session4",
			Query:     "programming languages",
			Timestamp: time.Now(),
		}

		reranked, err := reranker.Rerank(ctx, results, rerankCtx)
		assert.NoError(t, err)

		// Doc3 might rank higher than doc2 due to diversity
		// (depends on the exact implementation of diversity penalty)
		assert.Equal(t, "doc1", reranked[0].ID) // Top result should remain
	})

	t.Run("UpdateModel", func(t *testing.T) {
		// Create feedback data
		interactions := []*Interaction{
			{
				UserID:     "user5",
				SessionID:  "session5",
				QueryID:    "query5",
				DocumentID: "clicked_doc",
				Type:       InteractionTypeClick,
				Position:   3,
				Timestamp:  time.Now(),
			},
			{
				UserID:     "user5",
				SessionID:  "session5",
				QueryID:    "query5",
				DocumentID: "ignored_doc",
				Type:       InteractionTypeIgnore,
				Position:   1,
				Timestamp:  time.Now(),
			},
		}

		// Update model with feedback
		err := reranker.UpdateModel(ctx, interactions)
		assert.NoError(t, err)

		// Model should have learned from the feedback
		// (specific assertions depend on implementation details)
	})

	t.Run("TemporalBoost", func(t *testing.T) {
		now := time.Now()
		results := []SearchResult{
			{
				ID:       "old_doc",
				Score:    0.8,
				Position: 1,
				Metadata: map[string]interface{}{
					"created_at": now.Add(-60 * 24 * time.Hour), // 60 days old
				},
			},
			{
				ID:       "new_doc",
				Score:    0.75,
				Position: 2,
				Metadata: map[string]interface{}{
					"created_at": now.Add(-1 * time.Hour), // 1 hour old
				},
			},
		}

		rerankCtx := RerankingContext{
			UserID:    "user6",
			SessionID: "session6",
			Query:     "latest news",
			Timestamp: now,
		}

		reranked, err := reranker.Rerank(ctx, results, rerankCtx)
		assert.NoError(t, err)

		// New doc should get temporal boost
		// The exact ranking depends on boost magnitude
		assert.Len(t, reranked, 2)
	})
}

func TestLearningEngine(t *testing.T) {
	ctx := context.Background()
	engine := NewSimpleLearningEngine()

	t.Run("GenerateLearningSignals", func(t *testing.T) {
		interactions := []*Interaction{
			{
				QueryID:    "query1",
				DocumentID: "doc1",
				Type:       InteractionTypeClick,
				Position:   1,
				Value:      30.0, // dwell time
				Metadata: map[string]interface{}{
					"score": 0.8,
					"views": 100.0,
				},
			},
			{
				QueryID:    "query1",
				DocumentID: "doc2",
				Type:       InteractionTypeIgnore,
				Position:   2,
				Metadata: map[string]interface{}{
					"score": 0.9,
					"views": 50.0,
				},
			},
			{
				QueryID:    "query1",
				DocumentID: "doc3",
				Type:       InteractionTypeClick,
				Position:   3,
				Value:      60.0, // longer dwell time
				Metadata: map[string]interface{}{
					"score": 0.7,
					"views": 200.0,
				},
			},
		}

		signals, err := engine.GenerateLearningSignals(ctx, interactions)
		assert.NoError(t, err)
		assert.Greater(t, len(signals), 0)

		// Verify pairwise preferences were generated
		for _, signal := range signals {
			assert.NotEmpty(t, signal.Features)
			assert.GreaterOrEqual(t, signal.Label, 0.0)
			assert.LessOrEqual(t, signal.Label, 1.0)
		}
	})

	t.Run("UpdateClickModel", func(t *testing.T) {
		// Generate click data at different positions
		interactions := make([]*Interaction, 0)
		for pos := 1; pos <= 10; pos++ {
			// More clicks at higher positions
			numClicks := 11 - pos
			for i := 0; i < numClicks; i++ {
				interactions = append(interactions, &Interaction{
					Type:     InteractionTypeClick,
					Position: pos,
				})
			}
			// Add non-clicks
			for i := 0; i < 5; i++ {
				interactions = append(interactions, &Interaction{
					Type:     InteractionTypeIgnore,
					Position: pos,
				})
			}
		}

		err := engine.UpdateClickModel(ctx, interactions)
		assert.NoError(t, err)
	})

	t.Run("GetClickProbability", func(t *testing.T) {
		// Create fresh engine to avoid contamination from other tests
		freshEngine := NewSimpleLearningEngine()

		// Get click probabilities for different positions
		prob1, err := freshEngine.GetClickProbability(ctx, 1, nil)
		assert.NoError(t, err)
		assert.Greater(t, prob1, 0.0)
		assert.LessOrEqual(t, prob1, 1.0)

		prob10, err := freshEngine.GetClickProbability(ctx, 10, nil)
		assert.NoError(t, err)
		assert.Greater(t, prob10, 0.0)
		assert.Less(t, prob10, prob1) // Position 10 should have lower CTR
	})

	t.Run("TrainModel", func(t *testing.T) {
		// Create training signals
		signals := []*LearningSignal{
			{
				Query:      "test",
				DocumentID: "doc1",
				Features: map[string]float64{
					"score_diff": 0.5,
					"pos_diff":   2.0,
				},
				Label:  1.0,
				Weight: 1.0,
			},
			{
				Query:      "test",
				DocumentID: "doc2",
				Features: map[string]float64{
					"score_diff": -0.3,
					"pos_diff":   -1.0,
				},
				Label:  0.0,
				Weight: 1.0,
			},
		}

		err := engine.TrainModel(ctx, signals)
		assert.NoError(t, err)
	})

	t.Run("ExportImportModel", func(t *testing.T) {
		// Export model
		modelData, err := engine.ExportModel(ctx)
		assert.NoError(t, err)
		assert.NotEmpty(t, modelData)

		// Create new engine and import
		newEngine := NewSimpleLearningEngine()
		err = newEngine.ImportModel(ctx, modelData)
		assert.NoError(t, err)

		// Verify imported model works
		prob, err := newEngine.GetClickProbability(ctx, 1, nil)
		assert.NoError(t, err)
		assert.Greater(t, prob, 0.0)
	})
}

func TestFeedbackAnalyzer(t *testing.T) {
	ctx := context.Background()
	collector := NewMemoryCollector()
	analyzer := NewFeedbackAnalyzer(collector)

	t.Run("AnalyzeQuerySatisfaction", func(t *testing.T) {
		// Satisfied query - clicks on top results
		satisfiedInteractions := []*Interaction{
			{
				Type:     InteractionTypeClick,
				Position: 1,
			},
			{
				Type:     InteractionTypeDwell,
				Value:    45.0, // Good dwell time
				Position: 1,
			},
		}

		satisfied, confidence := analyzer.AnalyzeQuerySatisfaction(ctx, satisfiedInteractions)
		assert.True(t, satisfied)
		assert.Greater(t, confidence, 0.5)

		// Unsatisfied query - clicks on lower results
		unsatisfiedInteractions := []*Interaction{
			{
				Type:     InteractionTypeClick,
				Position: 8,
			},
			{
				Type:     InteractionTypeClick,
				Position: 9,
			},
			{
				Type:     InteractionTypeClick,
				Position: 10,
			},
		}

		satisfied, confidence = analyzer.AnalyzeQuerySatisfaction(ctx, unsatisfiedInteractions)
		assert.False(t, satisfied)

		// Negative feedback
		negativeInteractions := []*Interaction{
			{
				Type:  InteractionTypeRating,
				Value: 1.0, // Low rating
			},
		}

		satisfied, confidence = analyzer.AnalyzeQuerySatisfaction(ctx, negativeInteractions)
		assert.False(t, satisfied)
		assert.Equal(t, 1.0, confidence) // High confidence due to explicit negative feedback
	})

	t.Run("AnalyzeClickPattern", func(t *testing.T) {
		interactions := []*Interaction{
			{Type: InteractionTypeClick, Position: 1},
			{Type: InteractionTypeClick, Position: 2},
			{Type: InteractionTypeClick, Position: 3},
			{Type: InteractionTypeIgnore, Position: 4},
			{Type: InteractionTypeIgnore, Position: 5},
		}

		pattern, err := analyzer.AnalyzeClickPattern(ctx, interactions)
		assert.NoError(t, err)
		assert.Equal(t, "top_heavy", pattern["pattern_type"])
		assert.Equal(t, "high", pattern["position_bias"])
		assert.Equal(t, true, pattern["cascading"])
		assert.Equal(t, 3, pattern["click_count"])
		assert.InDelta(t, 2.0, pattern["avg_click_position"], 0.01)
	})

	t.Run("ComputeDocumentQuality", func(t *testing.T) {
		feedback := &DocumentFeedback{
			DocumentID:       "doc1",
			TotalViews:       100,
			TotalClicks:      30,
			TotalRatings:     10,
			AvgRating:        4.2,
			AvgDwellTime:     55.0,
			ClickThroughRate: 0.3,
		}

		quality, signals := analyzer.ComputeDocumentQuality(ctx, feedback)
		assert.Greater(t, quality, 0.0)
		assert.LessOrEqual(t, quality, 1.0)
		assert.Equal(t, 0.3, signals["ctr"])
		assert.InDelta(t, 0.84, signals["rating"], 0.01) // 4.2/5
		assert.Greater(t, signals["dwell"], 0.0)
	})

	t.Run("DetectAnomalies", func(t *testing.T) {
		// Rapid clicking pattern
		rapidClicks := make([]*Interaction, 15)
		baseTime := time.Now()
		for i := range rapidClicks {
			rapidClicks[i] = &Interaction{
				UserID:    "bot_user",
				Type:      InteractionTypeClick,
				Timestamp: baseTime.Add(time.Duration(i) * 500 * time.Millisecond), // 0.5s apart
			}
		}

		anomalies, err := analyzer.DetectAnomalies(ctx, rapidClicks)
		assert.NoError(t, err)
		assert.NotEmpty(t, anomalies)
		assert.Contains(t, anomalies[0], "rapid_clicking")

		// Excessive negative ratings
		negativeRatings := make([]*Interaction, 10)
		for i := range negativeRatings {
			negativeRatings[i] = &Interaction{
				UserID: "negative_user",
				Type:   InteractionTypeRating,
				Value:  1.0,
			}
		}

		anomalies, err = analyzer.DetectAnomalies(ctx, negativeRatings)
		assert.NoError(t, err)
		assert.NotEmpty(t, anomalies)
		assert.Contains(t, anomalies[0], "excessive_negative_ratings")
	})
}
