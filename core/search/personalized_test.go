package search

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	
	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/dshills/EmbeddixDB/core/feedback"
	"github.com/dshills/EmbeddixDB/core/text"
	"github.com/dshills/EmbeddixDB/persistence"
)

func setupTestComponents(t *testing.T) (*PersonalizedSearchManager, *FeedbackManager) {
	// Create in-memory vector store
	memStore := persistence.NewMemoryPersistence()
	indexFactory := &testIndexFactory{}
	vectorStore := core.NewVectorStore(memStore, indexFactory)
	
	// Create test collection
	ctx := context.Background()
	collection := core.Collection{
		Name:      "test",
		Dimension: 128,
		IndexType: "flat",
		Distance:  "cosine",
	}
	err := vectorStore.CreateCollection(ctx, collection)
	require.NoError(t, err)
	
	// Add test vectors
	testVectors := []core.Vector{
		{
			ID:     "doc1",
			Values: generateTestEmbedding(128),
			Metadata: map[string]string{
				"title":    "Machine Learning Basics",
				"category": "AI",
				"source":   "wikipedia",
				"views":    "1000",
				"rating":   "4.5",
			},
		},
		{
			ID:     "doc2",
			Values: generateTestEmbedding(128),
			Metadata: map[string]string{
				"title":    "Deep Learning Advanced",
				"category": "AI",
				"source":   "arxiv",
				"views":    "500",
				"rating":   "4.8",
			},
		},
		{
			ID:     "doc3",
			Values: generateTestEmbedding(128),
			Metadata: map[string]string{
				"title":    "Natural Language Processing",
				"category": "AI",
				"source":   "medium",
				"views":    "300",
				"rating":   "4.2",
			},
		},
	}
	
	// Add topics to vector metadata and store vectors
	for i, v := range testVectors {
		// Add topics to vector metadata as comma-separated string
		topics := []string{"ai", "technology"}
		if i == 0 {
			topics = []string{"machine_learning", "ai", "basics"}
		} else if i == 1 {
			topics = []string{"deep_learning", "neural_networks", "ai"}
		} else {
			topics = []string{"nlp", "language", "ai"}
		}
		v.Metadata["topics"] = strings.Join(topics, ",")
		
		err := vectorStore.AddVector(ctx, "test", v)
		require.NoError(t, err)
	}
	
	// Create text engine
	textEngine := text.NewEnhancedBM25Index()
	
	// Add documents to text engine
	for i, v := range testVectors {
		doc := ai.Document{
			ID:      v.ID,
			Content: v.Metadata["title"],
			Metadata: map[string]interface{}{
				"source": v.Metadata["source"],
				"views":  v.Metadata["views"],
				"rating": v.Metadata["rating"],
				"topics": []string{"ai", "technology"},
				"entities": []string{"AI", "ML"},
			},
		}
		if i == 0 {
			doc.Metadata["topics"] = []string{"machine_learning", "ai", "basics"}
		} else if i == 1 {
			doc.Metadata["topics"] = []string{"deep_learning", "neural_networks", "ai"}
		} else {
			doc.Metadata["topics"] = []string{"nlp", "language", "ai"}
		}
		
		err := textEngine.Index(ctx, []ai.Document{doc})
		require.NoError(t, err)
	}
	
	// Create mock model manager
	modelManager := &mockModelManager{}
	
	// Create feedback manager
	feedbackConfig := feedback.DefaultManagerConfig()
	feedbackConfig.EnablePersistence = false
	feedbackManager, err := feedback.NewManager(feedbackConfig)
	require.NoError(t, err)
	
	// Create feedback manager wrapper
	searchFeedbackManager := &FeedbackManager{
		Collector:      feedbackManager.Collector,
		SessionManager: feedbackManager.SessionManager,
		ProfileManager: feedbackManager.ProfileManager,
		LearningEngine: feedbackManager.LearningEngine,
		Analyzer:       feedbackManager.Analyzer,
		CTRTracker:     feedbackManager.CTRTracker,
	}
	
	// Create personalized search manager
	config := PersonalizedSearchConfig{
		EnablePersonalization: true,
		EnableLearning:        true,
		EnableSessionContext:  true,
		DefaultVectorWeight:   0.5,
		DefaultTextWeight:     0.5,
		MaxResults:            10,
		MinScore:              0.0,
	}
	
	searchManager := NewPersonalizedSearchManager(
		vectorStore,
		textEngine,
		modelManager,
		searchFeedbackManager,
		config,
	)
	
	return searchManager, searchFeedbackManager
}

func TestPersonalizedSearchManager_Search(t *testing.T) {
	ctx := context.Background()
	searchManager, feedbackManager := setupTestComponents(t)

	t.Run("BasicSearch", func(t *testing.T) {
		req := PersonalizedSearchRequest{
			CollectionID: "test",
			Query:        "machine learning",
			K:            3,
			SearchMode:   "text",
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.Greater(t, len(response.Results), 0)
		assert.Equal(t, "text", response.SearchMode)
		assert.False(t, response.Personalized)
	})

	t.Run("VectorSearch", func(t *testing.T) {
		req := PersonalizedSearchRequest{
			CollectionID: "test",
			Vector:       generateTestEmbedding(128),
			K:            3,
			SearchMode:   "vector",
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.Greater(t, len(response.Results), 0)
		assert.Equal(t, "vector", response.SearchMode)
	})

	t.Run("HybridSearch", func(t *testing.T) {
		req := PersonalizedSearchRequest{
			CollectionID: "test",
			Query:        "deep learning",
			Vector:       generateTestEmbedding(128),
			K:            3,
			SearchMode:   "hybrid",
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.Greater(t, len(response.Results), 0)
		assert.Equal(t, "hybrid", response.SearchMode)
	})

	t.Run("PersonalizedSearch", func(t *testing.T) {
		// Create user profile
		userID := "test_user"
		_, err := feedbackManager.ProfileManager.CreateProfile(ctx, userID)
		require.NoError(t, err)
		
		// Set user interests
		topics := map[string]float64{
			"deep_learning": 0.9,
			"machine_learning": 0.5,
		}
		err = feedbackManager.ProfileManager.IncrementInterests(ctx, userID, topics, nil)
		require.NoError(t, err)

		req := PersonalizedSearchRequest{
			CollectionID:          "test",
			Query:                 "learning",
			K:                     3,
			UserID:                userID,
			EnablePersonalization: true,
			EnableReranking:       true,
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.True(t, response.Personalized)
		assert.Greater(t, len(response.Results), 0)
		
		// Deep learning doc should rank higher due to user interest
		foundDeepLearning := false
		for i, result := range response.Results {
			if result.ID == "doc2" && i < 2 { // Should be in top 2
				foundDeepLearning = true
				break
			}
		}
		assert.True(t, foundDeepLearning, "Deep learning doc should rank high for this user")
	})

	t.Run("SessionAwareSearch", func(t *testing.T) {
		userID := "session_user"
		
		// Create session
		session, err := feedbackManager.SessionManager.CreateSession(ctx, userID, nil)
		require.NoError(t, err)

		// Record some interactions in the session
		interaction := &feedback.Interaction{
			UserID:       userID,
			SessionID:    session.ID,
			QueryID:      "prev_query",
			Query:        "neural networks",
			DocumentID:   "doc2",
			CollectionID: "test",
			Type:         feedback.InteractionTypeClick,
			Position:     1,
			Timestamp:    time.Now(),
		}
		err = searchManager.RecordInteraction(ctx, interaction)
		require.NoError(t, err)

		// Search with session context
		req := PersonalizedSearchRequest{
			CollectionID:          "test",
			Query:                 "advanced topics",
			K:                     3,
			UserID:                userID,
			SessionID:             session.ID,
			EnablePersonalization: true,
			EnableReranking:       true,
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, session.ID, response.SessionID)
	})

	t.Run("WeightCustomization", func(t *testing.T) {
		vectorWeight := 0.8
		textWeight := 0.2
		
		req := PersonalizedSearchRequest{
			CollectionID: "test",
			Query:        "learning",
			Vector:       generateTestEmbedding(128),
			K:            3,
			VectorWeight: &vectorWeight,
			TextWeight:   &textWeight,
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.Greater(t, len(response.Results), 0)
		
		// Results should be more influenced by vector similarity
	})

	t.Run("FeedbackCollection", func(t *testing.T) {
		req := PersonalizedSearchRequest{
			CollectionID:    "test",
			Query:           "machine learning",
			K:               3,
			UserID:          "feedback_user",
			CollectFeedback: true,
		}

		response, err := searchManager.Search(ctx, req)
		assert.NoError(t, err)
		assert.NotEmpty(t, response.QueryID)
		
		// Verify query feedback was recorded
		queryFeedback, err := feedbackManager.Collector.GetQueryFeedback(ctx, response.QueryID)
		assert.NoError(t, err)
		assert.Equal(t, req.Query, queryFeedback.Query)
		assert.Equal(t, len(response.Results), queryFeedback.ResultCount)
	})
}

func TestPersonalizedSearchManager_RecordInteraction(t *testing.T) {
	ctx := context.Background()
	searchManager, feedbackManager := setupTestComponents(t)

	t.Run("RecordClick", func(t *testing.T) {
		interaction := &feedback.Interaction{
			UserID:       "user1",
			SessionID:    "session1",
			QueryID:      "query1",
			Query:        "test query",
			DocumentID:   "doc1",
			CollectionID: "test",
			Type:         feedback.InteractionTypeClick,
			Position:     1,
			Timestamp:    time.Now(),
		}

		err := searchManager.RecordInteraction(ctx, interaction)
		assert.NoError(t, err)
		
		// Verify interaction was recorded
		filter := feedback.InteractionFilter{
			UserID: "user1",
			Type:   feedback.InteractionTypeClick,
		}
		interactions, err := feedbackManager.Collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 1)
		assert.Equal(t, "doc1", interactions[0].DocumentID)
		
		// Verify CTR was updated
		if feedbackManager.CTRTracker != nil {
			metrics, err := feedbackManager.CTRTracker.GetDocumentCTR(ctx, "doc1")
			assert.NoError(t, err)
			assert.Equal(t, int64(1), metrics.Clicks)
		}
	})

	t.Run("RecordDwell", func(t *testing.T) {
		// Create user profile first (ignore error if already exists)
		_, err := feedbackManager.ProfileManager.CreateProfile(ctx, "user1")
		if err != nil && !strings.Contains(err.Error(), "already exists") {
			require.NoError(t, err)
		}
		
		interaction := &feedback.Interaction{
			UserID:       "user1",
			SessionID:    "session1",
			QueryID:      "query1",
			DocumentID:   "doc2",
			CollectionID: "test",
			Type:         feedback.InteractionTypeDwell,
			Value:        45.5, // dwell time in seconds
			Position:     2,
			Timestamp:    time.Now(),
		}

		err = searchManager.RecordInteraction(ctx, interaction)
		assert.NoError(t, err)
		
		// Add a small delay to allow for async processing
		time.Sleep(100 * time.Millisecond)
		
		// Verify user interests were updated
		topics, _, err := feedbackManager.ProfileManager.GetTopInterests(ctx, "user1", 5)
		assert.NoError(t, err)
		if len(topics) == 0 {
			// Debug: check if profile exists
			profile, profileErr := feedbackManager.ProfileManager.GetProfile(ctx, "user1")
			t.Logf("Profile exists: %v, error: %v", profile != nil, profileErr)
			if profile != nil {
				t.Logf("Profile topic interests: %+v", profile.TopicInterests)
			}
		}
		assert.NotEmpty(t, topics) // Should have learned from the interaction
	})

	t.Run("RecordRating", func(t *testing.T) {
		// Create user profile first
		_, err := feedbackManager.ProfileManager.CreateProfile(ctx, "user2")
		require.NoError(t, err)
		
		interaction := &feedback.Interaction{
			UserID:       "user2",
			SessionID:    "session2",
			QueryID:      "query2",
			DocumentID:   "doc3",
			CollectionID: "test",
			Type:         feedback.InteractionTypeRating,
			Value:        4.5, // rating
			Position:     1,
			Timestamp:    time.Now(),
		}

		err = searchManager.RecordInteraction(ctx, interaction)
		assert.NoError(t, err)
		
		// Add a small delay to allow for async processing
		time.Sleep(100 * time.Millisecond)
		
		// High rating should increase user interest
		topics, _, err := feedbackManager.ProfileManager.GetTopInterests(ctx, "user2", 5)
		assert.NoError(t, err)
		if len(topics) == 0 {
			// Debug: check if profile exists
			profile, profileErr := feedbackManager.ProfileManager.GetProfile(ctx, "user2")
			t.Logf("Profile exists: %v, error: %v", profile != nil, profileErr)
			if profile != nil {
				t.Logf("Profile topic interests: %+v", profile.TopicInterests)
			}
		}
		assert.NotEmpty(t, topics)
	})
}

func TestPersonalizedSearchManager_CTROptimization(t *testing.T) {
	ctx := context.Background()
	searchManager, feedbackManager := setupTestComponents(t)

	// Simulate impressions and clicks to build CTR data
	documents := []string{"doc1", "doc2", "doc3"}
	clickRates := []float64{0.5, 0.2, 0.8} // doc3 has highest CTR
	
	for i, docID := range documents {
		// Record 100 impressions per document across various positions
		for j := 0; j < 100; j++ {
			impression := feedback.Impression{
				QueryID:    "ctr_query" + docID + "_" + string(rune('a'+j)),
				Query:      "test",
				DocumentID: docID,
				Position:   (j % 3) + 1, // Distribute across positions 1-3
				UserID:     "ctr_user" + string(rune('a'+j)),
				SessionID:  "ctr_session" + string(rune('a'+j)),
				Timestamp:  time.Now(),
			}
			err := feedbackManager.CTRTracker.RecordImpression(ctx, impression)
			require.NoError(t, err)
			
			// Record clicks based on CTR - use number of clicks that gives exact CTR
			clickCount := int(float64(100) * clickRates[i])
			if j < clickCount {
				click := feedback.Click{
					QueryID:    impression.QueryID,
					DocumentID: impression.DocumentID,
					Position:   impression.Position,
					UserID:     impression.UserID,
					SessionID:  impression.SessionID,
					Timestamp:  time.Now(),
				}
				err = feedbackManager.CTRTracker.RecordClick(ctx, click)
				require.NoError(t, err)
			}
		}
	}

	t.Run("GetCTROptimizedRanking", func(t *testing.T) {
		// Create mock results
		results := []PersonalizedSearchResult{
			{ID: "doc1", Score: 0.7, Position: 1},
			{ID: "doc2", Score: 0.8, Position: 2},
			{ID: "doc3", Score: 0.6, Position: 3},
		}

		optimized, err := searchManager.GetCTROptimizedRanking(ctx, results, "test")
		assert.NoError(t, err)
		assert.Len(t, optimized, 3)
		
		// doc3 should rank first due to highest CTR
		assert.Equal(t, "doc3", optimized[0].ID)
		assert.Equal(t, 1, optimized[0].Position)
	})

	t.Run("GetCTRReport", func(t *testing.T) {
		report, err := searchManager.GetCTRReport(ctx)
		assert.NoError(t, err)
		assert.Greater(t, report.TotalImpressions, int64(0))
		assert.Greater(t, report.TotalClicks, int64(0))
		assert.NotEmpty(t, report.TopDocuments)
		
		// Find doc3 in top documents
		found := false
		for _, doc := range report.TopDocuments {
			if doc.DocumentID == "doc3" {
				found = true
				assert.InDelta(t, 0.8, doc.CTRMetrics.CTR, 0.1)
				break
			}
		}
		assert.True(t, found, "doc3 should be in top documents")
	})
}

// Helper functions

func generateTestEmbedding(dim int) []float32 {
	embedding := make([]float32, dim)
	for i := range embedding {
		embedding[i] = float32(i) / float32(dim) // Simple pattern for testing
	}
	return embedding
}

// Mock model manager for testing
type mockModelManager struct{}

func (m *mockModelManager) LoadModel(ctx context.Context, modelName string, config ai.ModelConfig) error {
	return nil
}

func (m *mockModelManager) UnloadModel(modelName string) error {
	return nil
}

func (m *mockModelManager) GetEngine(modelName string) (ai.EmbeddingEngine, error) {
	return &mockEmbeddingEngine{}, nil
}

func (m *mockModelManager) ListModels() []ai.ModelInfo {
	return []ai.ModelInfo{{Name: "test-model", Version: "1.0", Dimension: 128}}
}

func (m *mockModelManager) GetModelHealth(modelName string) (ai.ModelHealth, error) {
	return ai.ModelHealth{ModelName: modelName, Status: "healthy"}, nil
}

type mockEmbeddingEngine struct{}

func (e *mockEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {
	embeddings := make([][]float32, len(content))
	for i := range content {
		embeddings[i] = generateTestEmbedding(128)
	}
	return embeddings, nil
}

func (e *mockEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	embeddings := make([][]float32, len(content))
	for i := range content {
		embeddings[i] = generateTestEmbedding(128)
	}
	return embeddings, nil
}

func (e *mockEmbeddingEngine) Warm(ctx context.Context) error {
	return nil
}

func (e *mockEmbeddingEngine) GetDimension() int {
	return 128
}

func (e *mockEmbeddingEngine) GetModelInfo() ai.ModelInfo {
	return ai.ModelInfo{Name: "test-model", Version: "1.0", Dimension: 128}
}

func (e *mockEmbeddingEngine) Close() error {
	return nil
}

// Test index factory
type testIndexFactory struct{}

func (f *testIndexFactory) CreateIndex(indexType string, dimension int, distanceMetric core.DistanceMetric) (core.Index, error) {
	return &testIndex{}, nil
}

// Test index implementation
type testIndex struct {
	vectors []core.Vector
}

func (i *testIndex) Add(vector core.Vector) error { 
	i.vectors = append(i.vectors, vector)
	return nil 
}

func (i *testIndex) Search(query []float32, k int, filter map[string]string) ([]core.SearchResult, error) {
	// Return mock results for testing
	results := []core.SearchResult{}
	for j, v := range i.vectors {
		if j >= k {
			break
		}
		results = append(results, core.SearchResult{
			ID:       v.ID,
			Score:    float32(0.8 - float32(j)*0.1), // Decreasing scores
			Metadata: v.Metadata,
		})
	}
	return results, nil
}
func (i *testIndex) RangeSearch(query []float32, radius float32, filter map[string]string, limit int) ([]core.SearchResult, error) {
	return []core.SearchResult{}, nil
}
func (i *testIndex) Delete(id string) error { return nil }
func (i *testIndex) Rebuild() error { return nil }
func (i *testIndex) Size() int { return 0 }
func (i *testIndex) Type() string { return "test" }
func (i *testIndex) Serialize() ([]byte, error) { return []byte{}, nil }
func (i *testIndex) Deserialize(data []byte) error { return nil }