package feedback

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFeedbackSystemIntegration(t *testing.T) {
	ctx := context.Background()

	// Create feedback manager with all components
	tempDir := t.TempDir()
	config := ManagerConfig{
		SessionTimeout:      30 * time.Minute,
		MaxCacheSize:        1000,
		EnableCTRTracking:   true,
		CTRMaxDataSize:      10000,
		CTRDecayFactor:      0.95,
		EnableLearning:      true,
		MaxTrainingDataSize: 1000,
		EnablePersistence:   true,
		StoragePath:         filepath.Join(tempDir, "feedback_test.db"),
	}

	manager, err := NewManager(config)
	require.NoError(t, err)
	defer manager.Close()

	t.Run("CompleteUserJourney", func(t *testing.T) {
		userID := "test_user"

		// Step 1: Create user profile
		profile, err := manager.ProfileManager.CreateProfile(ctx, userID)
		require.NoError(t, err)
		assert.Equal(t, userID, profile.ID)

		// Step 2: Start a session
		session, err := manager.SessionManager.CreateSession(ctx, userID, map[string]interface{}{
			"device": "desktop",
		})
		require.NoError(t, err)
		assert.Equal(t, userID, session.UserID)

		// Step 3: Record search and impressions
		queryID := "query_001"
		query := "machine learning tutorial"

		// Record query
		queryFeedback := &QueryFeedback{
			QueryID:     queryID,
			Query:       query,
			UserID:      userID,
			SessionID:   session.ID,
			Timestamp:   time.Now(),
			ResultCount: 5,
		}
		err = manager.Collector.RecordQuery(ctx, queryFeedback)
		require.NoError(t, err)

		// Record impressions
		documents := []string{"doc1", "doc2", "doc3", "doc4", "doc5"}
		for i, docID := range documents {
			impression := Impression{
				QueryID:     queryID,
				Query:       query,
				DocumentID:  docID,
				Position:    i + 1,
				UserID:      userID,
				SessionID:   session.ID,
				Timestamp:   time.Now(),
				ResultCount: 5,
				SearchMode:  "hybrid",
			}
			err = manager.CTRTracker.RecordImpression(ctx, impression)
			require.NoError(t, err)
		}

		// Step 4: User clicks on results
		clickedDocs := []string{"doc2", "doc3"}
		for _, docID := range clickedDocs {
			// Record interaction
			interaction := &Interaction{
				UserID:       userID,
				SessionID:    session.ID,
				QueryID:      queryID,
				Query:        query,
				DocumentID:   docID,
				CollectionID: "test_collection",
				Type:         InteractionTypeClick,
				Position:     2, // Simplified
				Timestamp:    time.Now(),
				Metadata: map[string]interface{}{
					"topics": []string{"machine_learning", "ai"},
				},
			}
			err = manager.Collector.RecordInteraction(ctx, interaction)
			require.NoError(t, err)

			// Record click for CTR
			click := Click{
				QueryID:    queryID,
				DocumentID: docID,
				Position:   2,
				UserID:     userID,
				SessionID:  session.ID,
				Timestamp:  time.Now(),
				DwellTime:  30.0,
			}
			err = manager.CTRTracker.RecordClick(ctx, click)
			require.NoError(t, err)
		}

		// Step 5: Update user interests based on interactions
		topics := map[string]float64{
			"machine_learning": 0.8,
			"ai":               0.6,
		}
		err = manager.ProfileManager.IncrementInterests(ctx, userID, topics, nil)
		require.NoError(t, err)

		// Step 6: Verify everything was recorded correctly

		// Check interactions
		filter := InteractionFilter{
			UserID:    userID,
			SessionID: session.ID,
		}
		interactions, err := manager.Collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 2) // Two clicks

		// Check CTR metrics
		for _, docID := range clickedDocs {
			metrics, err := manager.CTRTracker.GetDocumentCTR(ctx, docID)
			assert.NoError(t, err)
			assert.Equal(t, int64(1), metrics.Impressions)
			assert.Equal(t, int64(1), metrics.Clicks)
			assert.Equal(t, 1.0, metrics.CTR)
		}

		// Check user profile
		updatedProfile, err := manager.ProfileManager.GetProfile(ctx, userID)
		assert.NoError(t, err)
		assert.Greater(t, updatedProfile.TopicInterests["machine_learning"], 0.0)

		// Check query satisfaction
		allInteractions, err := manager.Collector.GetInteractions(ctx, InteractionFilter{QueryID: queryID})
		assert.NoError(t, err)
		satisfied, confidence := manager.Analyzer.AnalyzeQuerySatisfaction(ctx, allInteractions)
		assert.True(t, satisfied) // Clicks on top results indicate satisfaction
		assert.Greater(t, confidence, 0.0)

		// Step 7: Generate learning signals
		signals, err := manager.LearningEngine.GenerateLearningSignals(ctx, interactions)
		assert.NoError(t, err)
		assert.Greater(t, len(signals), 0)

		// Step 8: End session
		err = manager.SessionManager.EndSession(ctx, session.ID)
		assert.NoError(t, err)

		// Verify session was ended
		endedSession, err := manager.SessionManager.GetSession(ctx, session.ID)
		assert.NoError(t, err)
		assert.NotNil(t, endedSession.EndTime)
	})

	t.Run("MultiUserScenario", func(t *testing.T) {
		// Simulate multiple users with different preferences
		users := []struct {
			id         string
			interests  []string
			clickRates map[string]float64 // doc -> click probability
		}{
			{
				id:        "user_ai_focused",
				interests: []string{"ai", "machine_learning", "deep_learning"},
				clickRates: map[string]float64{
					"ai_doc1": 0.9,
					"ai_doc2": 0.8,
					"db_doc1": 0.1,
				},
			},
			{
				id:        "user_db_focused",
				interests: []string{"databases", "sql", "nosql"},
				clickRates: map[string]float64{
					"ai_doc1": 0.1,
					"ai_doc2": 0.2,
					"db_doc1": 0.9,
				},
			},
		}

		// Simulate behavior for each user
		for _, user := range users {
			// Create profile
			_, err := manager.ProfileManager.CreateProfile(ctx, user.id)
			require.NoError(t, err)

			// Create session
			session, err := manager.SessionManager.CreateSession(ctx, user.id, nil)
			require.NoError(t, err)

			// Simulate searches and clicks
			for i := 0; i < 5; i++ {
				queryID := fmt.Sprintf("%s_query_%d", user.id, i)

				// Record impressions for all docs
				for _, docID := range []string{"ai_doc1", "ai_doc2", "db_doc1"} {
					impression := Impression{
						QueryID:    queryID,
						Query:      "technology",
						DocumentID: docID,
						Position:   1,
						UserID:     user.id,
						SessionID:  session.ID,
						Timestamp:  time.Now(),
					}
					err = manager.CTRTracker.RecordImpression(ctx, impression)
					require.NoError(t, err)

					// Simulate clicks based on user preferences
					if clickProb, exists := user.clickRates[docID]; exists {
						if float64(i)/5.0 < clickProb {
							click := Click{
								QueryID:    queryID,
								DocumentID: docID,
								Position:   1,
								UserID:     user.id,
								SessionID:  session.ID,
								Timestamp:  time.Now(),
							}
							err = manager.CTRTracker.RecordClick(ctx, click)
							require.NoError(t, err)
						}
					}
				}
			}

			// Update interests based on behavior
			topicInterests := make(map[string]float64)
			for _, interest := range user.interests {
				topicInterests[interest] = 0.8
			}
			err = manager.ProfileManager.IncrementInterests(ctx, user.id, topicInterests, nil)
			require.NoError(t, err)
		}

		// Verify users have different profiles
		aiProfile, err := manager.ProfileManager.GetProfile(ctx, "user_ai_focused")
		assert.NoError(t, err)
		dbProfile, err := manager.ProfileManager.GetProfile(ctx, "user_db_focused")
		assert.NoError(t, err)

		// AI user should have high AI interest
		assert.Greater(t, aiProfile.TopicInterests["ai"], dbProfile.TopicInterests["ai"])
		// DB user should have high database interest
		assert.Greater(t, dbProfile.TopicInterests["databases"], aiProfile.TopicInterests["databases"])

		// Check CTR differences
		aiDoc1CTR, err := manager.CTRTracker.GetDocumentCTR(ctx, "ai_doc1")
		assert.NoError(t, err)
		dbDoc1CTR, err := manager.CTRTracker.GetDocumentCTR(ctx, "db_doc1")
		assert.NoError(t, err)

		// Both documents should have impressions
		assert.Greater(t, aiDoc1CTR.Impressions, int64(0))
		assert.Greater(t, dbDoc1CTR.Impressions, int64(0))

		// Generate CTR report
		report, err := manager.CTRTracker.ExportMetrics(ctx)
		assert.NoError(t, err)
		assert.Greater(t, report.TotalImpressions, int64(0))
		assert.Greater(t, report.TotalClicks, int64(0))
	})

	t.Run("LearningFromFeedback", func(t *testing.T) {
		// Create a scenario where the system learns from user feedback
		userID := "learning_user"

		// Create profile
		_, err := manager.ProfileManager.CreateProfile(ctx, userID)
		require.NoError(t, err)

		// Simulate multiple search sessions with consistent behavior
		for sessionNum := 0; sessionNum < 3; sessionNum++ {
			session, err := manager.SessionManager.CreateSession(ctx, userID, nil)
			require.NoError(t, err)

			queryID := fmt.Sprintf("learn_query_%d", sessionNum)

			// User consistently clicks on lower-ranked results
			// This simulates a case where initial ranking is poor
			interactions := []*Interaction{
				{
					UserID:     userID,
					SessionID:  session.ID,
					QueryID:    queryID,
					DocumentID: "relevant_doc",
					Type:       InteractionTypeClick,
					Position:   5, // Clicked result at position 5
					Timestamp:  time.Now(),
					Value:      60.0, // Long dwell time
				},
				{
					UserID:     userID,
					SessionID:  session.ID,
					QueryID:    queryID,
					DocumentID: "irrelevant_doc",
					Type:       InteractionTypeIgnore,
					Position:   1, // Ignored top result
					Timestamp:  time.Now(),
				},
			}

			for _, interaction := range interactions {
				err = manager.Collector.RecordInteraction(ctx, interaction)
				require.NoError(t, err)
			}

			// Generate learning signals
			signals, err := manager.LearningEngine.GenerateLearningSignals(ctx, interactions)
			assert.NoError(t, err)
			assert.Greater(t, len(signals), 0)

			// Train model
			err = manager.LearningEngine.TrainModel(ctx, signals)
			assert.NoError(t, err)
		}

		// Export trained model
		modelData, err := manager.LearningEngine.ExportModel(ctx)
		assert.NoError(t, err)
		assert.NotEmpty(t, modelData)

		// The system should have learned that "relevant_doc" is preferred over "irrelevant_doc"
		// In a real implementation, we would verify this through re-ranking
	})

	t.Run("PersistenceAcrossRestarts", func(t *testing.T) {
		userID := "persistent_user"
		sessionID := ""

		// Create initial data
		{
			// Create profile
			_, err := manager.ProfileManager.CreateProfile(ctx, userID)
			require.NoError(t, err)

			// Create session
			session, err := manager.SessionManager.CreateSession(ctx, userID, nil)
			require.NoError(t, err)
			sessionID = session.ID

			// Record interactions
			interaction := &Interaction{
				ID:         "persist_int",
				UserID:     userID,
				SessionID:  sessionID,
				QueryID:    "persist_query",
				DocumentID: "persist_doc",
				Type:       InteractionTypeClick,
				Timestamp:  time.Now(),
			}
			err = manager.Collector.RecordInteraction(ctx, interaction)
			require.NoError(t, err)
		}

		// Close manager
		err = manager.Close()
		require.NoError(t, err)

		// Create new manager with same storage path
		newManager, err := NewManager(config)
		require.NoError(t, err)
		defer newManager.Close()

		// Verify data persisted
		{
			// Check profile
			profile, err := newManager.ProfileManager.GetProfile(ctx, userID)
			assert.NoError(t, err)
			assert.Equal(t, userID, profile.ID)

			// Check session
			session, err := newManager.SessionManager.GetSession(ctx, sessionID)
			assert.NoError(t, err)
			assert.Equal(t, userID, session.UserID)

			// Check interactions
			filter := InteractionFilter{
				UserID: userID,
			}
			interactions, err := newManager.Collector.GetInteractions(ctx, filter)
			assert.NoError(t, err)
			assert.Len(t, interactions, 1)
			assert.Equal(t, "persist_int", interactions[0].ID)
		}
	})
}

func TestConcurrentFeedbackOperations(t *testing.T) {
	ctx := context.Background()

	// Create manager with in-memory storage for speed
	config := DefaultManagerConfig()
	config.EnablePersistence = false

	manager, err := NewManager(config)
	require.NoError(t, err)
	defer manager.Close()

	// Test concurrent operations
	numUsers := 10
	numOperationsPerUser := 20

	var wg sync.WaitGroup
	errors := make(chan error, numUsers*numOperationsPerUser)

	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(userNum int) {
			defer wg.Done()

			userID := fmt.Sprintf("concurrent_user_%d", userNum)

			// Create profile
			_, err := manager.ProfileManager.CreateProfile(ctx, userID)
			if err != nil {
				errors <- err
				return
			}

			// Create session
			session, err := manager.SessionManager.CreateSession(ctx, userID, nil)
			if err != nil {
				errors <- err
				return
			}

			// Perform multiple operations
			for j := 0; j < numOperationsPerUser; j++ {
				// Record interaction
				interaction := &Interaction{
					UserID:       userID,
					SessionID:    session.ID,
					QueryID:      fmt.Sprintf("query_%d_%d", userNum, j),
					DocumentID:   fmt.Sprintf("doc_%d_%d", userNum, j),
					CollectionID: "test",
					Type:         InteractionTypeClick,
					Position:     1,
					Timestamp:    time.Now(),
				}

				if err := manager.Collector.RecordInteraction(ctx, interaction); err != nil {
					errors <- err
					continue
				}

				// Record impression
				impression := Impression{
					QueryID:    interaction.QueryID,
					Query:      "test query",
					DocumentID: interaction.DocumentID,
					Position:   1,
					UserID:     userID,
					SessionID:  session.ID,
					Timestamp:  time.Now(),
				}

				if err := manager.CTRTracker.RecordImpression(ctx, impression); err != nil {
					errors <- err
					continue
				}

				// Update interests
				topics := map[string]float64{
					fmt.Sprintf("topic_%d", j%5): 0.1,
				}
				if err := manager.ProfileManager.IncrementInterests(ctx, userID, topics, nil); err != nil {
					errors <- err
					continue
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for errors
	errorCount := 0
	for err := range errors {
		t.Errorf("Concurrent operation error: %v", err)
		errorCount++
	}
	assert.Equal(t, 0, errorCount, "Should have no errors in concurrent operations")

	// Verify all operations completed
	for i := 0; i < numUsers; i++ {
		userID := fmt.Sprintf("concurrent_user_%d", i)

		// Check profile exists
		profile, err := manager.ProfileManager.GetProfile(ctx, userID)
		assert.NoError(t, err)
		assert.NotNil(t, profile)

		// Check interactions
		filter := InteractionFilter{
			UserID: userID,
		}
		interactions, err := manager.Collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, numOperationsPerUser)
	}
}
