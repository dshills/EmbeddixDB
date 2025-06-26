package feedback

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBoltFeedbackStore(t *testing.T) {
	// Create temporary database file
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_feedback.db")

	store, err := NewBoltFeedbackStore(dbPath)
	require.NoError(t, err)
	defer store.Close()

	ctx := context.Background()

	t.Run("SaveAndLoadInteraction", func(t *testing.T) {
		interaction := &Interaction{
			ID:           "test_interaction_1",
			UserID:       "user1",
			SessionID:    "session1",
			QueryID:      "query1",
			Query:        "test query",
			DocumentID:   "doc1",
			CollectionID: "collection1",
			Type:         InteractionTypeClick,
			Value:        0,
			Position:     1,
			Timestamp:    time.Now(),
			Metadata: map[string]interface{}{
				"device": "mobile",
			},
		}

		// Save interaction
		err := store.SaveInteraction(ctx, interaction)
		assert.NoError(t, err)

		// Load by user
		filter := InteractionFilter{
			UserID: "user1",
		}
		interactions, err := store.LoadInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 1)
		assert.Equal(t, interaction.ID, interactions[0].ID)
		assert.Equal(t, interaction.Query, interactions[0].Query)
		assert.Equal(t, "mobile", interactions[0].Metadata["device"])
	})

	t.Run("LoadInteractions_BySession", func(t *testing.T) {
		// Save multiple interactions for a session
		for i := 0; i < 5; i++ {
			interaction := &Interaction{
				ID:           "session_int_" + string(rune('1'+i)),
				UserID:       "user2",
				SessionID:    "session2",
				QueryID:      "query2",
				DocumentID:   "doc" + string(rune('1'+i)),
				CollectionID: "collection1",
				Type:         InteractionTypeClick,
				Position:     i + 1,
				Timestamp:    time.Now(),
			}
			err := store.SaveInteraction(ctx, interaction)
			require.NoError(t, err)
		}

		// Load by session
		filter := InteractionFilter{
			SessionID: "session2",
		}
		interactions, err := store.LoadInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 5)

		// Verify all belong to the session
		for _, interaction := range interactions {
			assert.Equal(t, "session2", interaction.SessionID)
		}
	})

	t.Run("LoadInteractions_WithFilters", func(t *testing.T) {
		// Save interaction with specific type
		dwellInteraction := &Interaction{
			ID:           "dwell_int_1",
			UserID:       "user3",
			SessionID:    "session3",
			QueryID:      "query3",
			DocumentID:   "doc_dwell",
			CollectionID: "collection1",
			Type:         InteractionTypeDwell,
			Value:        45.5,
			Position:     1,
			Timestamp:    time.Now(),
		}
		err := store.SaveInteraction(ctx, dwellInteraction)
		require.NoError(t, err)

		// Filter by type
		filter := InteractionFilter{
			UserID: "user3",
			Type:   InteractionTypeDwell,
		}
		interactions, err := store.LoadInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 1)
		assert.Equal(t, InteractionTypeDwell, interactions[0].Type)
		assert.Equal(t, 45.5, interactions[0].Value)
	})

	t.Run("SaveAndLoadQueryFeedback", func(t *testing.T) {
		queryFeedback := &QueryFeedback{
			QueryID:          "query_feedback_1",
			Query:            "machine learning",
			UserID:           "user1",
			SessionID:        "session1",
			Timestamp:        time.Now(),
			ResultCount:      10,
			ClickCount:       3,
			AvgDwellTime:     35.5,
			AvgClickPosition: 2.3,
			Satisfied:        true,
		}

		// Save
		err := store.SaveQueryFeedback(ctx, queryFeedback)
		assert.NoError(t, err)

		// Load
		loaded, err := store.LoadQueryFeedback(ctx, "query_feedback_1")
		assert.NoError(t, err)
		assert.Equal(t, queryFeedback.Query, loaded.Query)
		assert.Equal(t, queryFeedback.ClickCount, loaded.ClickCount)
		assert.Equal(t, queryFeedback.Satisfied, loaded.Satisfied)
	})

	t.Run("SaveAndLoadDocumentFeedback", func(t *testing.T) {
		docFeedback := &DocumentFeedback{
			DocumentID:       "doc_feedback_1",
			CollectionID:     "collection1",
			TotalViews:       1000,
			TotalClicks:      250,
			TotalRatings:     50,
			AvgRating:        4.3,
			AvgDwellTime:     42.5,
			ClickThroughRate: 0.25,
			LastUpdated:      time.Now(),
		}

		// Save
		err := store.SaveDocumentFeedback(ctx, docFeedback)
		assert.NoError(t, err)

		// Load
		loaded, err := store.LoadDocumentFeedback(ctx, "doc_feedback_1")
		assert.NoError(t, err)
		assert.Equal(t, docFeedback.TotalViews, loaded.TotalViews)
		assert.Equal(t, docFeedback.AvgRating, loaded.AvgRating)
		assert.Equal(t, docFeedback.ClickThroughRate, loaded.ClickThroughRate)
	})

	t.Run("LoadNonExistent", func(t *testing.T) {
		// Try to load non-existent query feedback
		_, err := store.LoadQueryFeedback(ctx, "non_existent_query")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")

		// Try to load non-existent document feedback
		_, err = store.LoadDocumentFeedback(ctx, "non_existent_doc")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("IndexConsistency", func(t *testing.T) {
		// Save interaction
		interaction := &Interaction{
			ID:           "index_test_1",
			UserID:       "index_user",
			SessionID:    "index_session",
			QueryID:      "index_query",
			DocumentID:   "index_doc",
			CollectionID: "collection1",
			Type:         InteractionTypeClick,
			Position:     1,
			Timestamp:    time.Now(),
		}
		err := store.SaveInteraction(ctx, interaction)
		require.NoError(t, err)

		// Verify all indexes work
		filters := []InteractionFilter{
			{UserID: "index_user"},
			{SessionID: "index_session"},
			{QueryID: "index_query"},
			{DocumentID: "index_doc"},
		}

		for _, filter := range filters {
			interactions, err := store.LoadInteractions(ctx, filter)
			assert.NoError(t, err)
			assert.Len(t, interactions, 1)
			assert.Equal(t, "index_test_1", interactions[0].ID)
		}
	})
}

func TestBoltSessionStore(t *testing.T) {
	// Create temporary database
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_sessions.db")

	feedbackStore, err := NewBoltFeedbackStore(dbPath)
	require.NoError(t, err)
	defer feedbackStore.Close()

	store := NewBoltSessionStore(feedbackStore.db)
	ctx := context.Background()

	t.Run("SaveAndLoadSession", func(t *testing.T) {
		session := &Session{
			ID:         "test_session_1",
			UserID:     "user1",
			StartTime:  time.Now(),
			QueryCount: 5,
			Metadata: map[string]interface{}{
				"device":  "desktop",
				"browser": "chrome",
			},
		}

		// Save
		err := store.SaveSession(ctx, session)
		assert.NoError(t, err)

		// Load
		loaded, err := store.LoadSession(ctx, "test_session_1")
		assert.NoError(t, err)
		assert.Equal(t, session.UserID, loaded.UserID)
		assert.Equal(t, session.QueryCount, loaded.QueryCount)
		assert.Equal(t, "desktop", loaded.Metadata["device"])
	})

	t.Run("LoadUserSessions", func(t *testing.T) {
		// Save multiple sessions for a user
		for i := 0; i < 5; i++ {
			session := &Session{
				ID:         "user_session_" + string(rune('1'+i)),
				UserID:     "multi_user",
				StartTime:  time.Now().Add(time.Duration(-i) * time.Hour),
				QueryCount: i + 1,
			}
			err := store.SaveSession(ctx, session)
			require.NoError(t, err)
		}

		// Load user sessions
		sessions, err := store.LoadUserSessions(ctx, "multi_user", 3)
		assert.NoError(t, err)
		assert.LessOrEqual(t, len(sessions), 3)

		// Verify all belong to the user
		for _, session := range sessions {
			assert.Equal(t, "multi_user", session.UserID)
		}
	})

	t.Run("UpdateSession", func(t *testing.T) {
		session := &Session{
			ID:         "update_session_1",
			UserID:     "user1",
			StartTime:  time.Now(),
			QueryCount: 1,
		}

		// Save initial
		err := store.SaveSession(ctx, session)
		require.NoError(t, err)

		// Update
		endTime := time.Now()
		session.EndTime = &endTime
		session.QueryCount = 10
		err = store.SaveSession(ctx, session)
		assert.NoError(t, err)

		// Verify update
		loaded, err := store.LoadSession(ctx, "update_session_1")
		assert.NoError(t, err)
		assert.NotNil(t, loaded.EndTime)
		assert.Equal(t, 10, loaded.QueryCount)
	})

	t.Run("DeleteSession", func(t *testing.T) {
		session := &Session{
			ID:        "delete_session_1",
			UserID:    "user1",
			StartTime: time.Now(),
		}

		// Save
		err := store.SaveSession(ctx, session)
		require.NoError(t, err)

		// Delete
		err = store.DeleteSession(ctx, "delete_session_1")
		assert.NoError(t, err)

		// Verify deleted
		_, err = store.LoadSession(ctx, "delete_session_1")
		assert.Error(t, err)
	})
}

func TestBoltProfileStore(t *testing.T) {
	// Create temporary database
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_profiles.db")

	feedbackStore, err := NewBoltFeedbackStore(dbPath)
	require.NoError(t, err)
	defer feedbackStore.Close()

	store := NewBoltProfileStore(feedbackStore.db)
	ctx := context.Background()

	t.Run("SaveAndLoadProfile", func(t *testing.T) {
		profile := &UserProfile{
			ID:               "user1",
			CreatedAt:        time.Now(),
			UpdatedAt:        time.Now(),
			SearchCount:      100,
			InteractionCount: 50,
			Preferences: UserPreferences{
				PreferredLanguages: []string{"en", "es"},
				SearchRecency:      0.7,
				SearchDiversity:    0.3,
				TopicWeights: map[string]float64{
					"ai": 0.8,
					"ml": 0.7,
				},
				VectorWeight: 0.6,
				TextWeight:   0.4,
			},
			TopicInterests: map[string]float64{
				"machine_learning": 0.9,
				"deep_learning":    0.8,
			},
			EntityInterests: map[string]float64{
				"OpenAI": 0.7,
				"Google": 0.6,
			},
			TermFrequency: map[string]int{
				"learning": 10,
				"neural":   5,
			},
		}

		// Save
		err := store.SaveProfile(ctx, profile)
		assert.NoError(t, err)

		// Load
		loaded, err := store.LoadProfile(ctx, "user1")
		assert.NoError(t, err)
		assert.Equal(t, profile.SearchCount, loaded.SearchCount)
		assert.Equal(t, profile.Preferences.VectorWeight, loaded.Preferences.VectorWeight)
		assert.Equal(t, profile.TopicInterests["machine_learning"], loaded.TopicInterests["machine_learning"])
		assert.Equal(t, profile.TermFrequency["learning"], loaded.TermFrequency["learning"])
	})

	t.Run("UpdateProfile", func(t *testing.T) {
		profile := &UserProfile{
			ID:               "user2",
			CreatedAt:        time.Now(),
			UpdatedAt:        time.Now(),
			SearchCount:      10,
			InteractionCount: 5,
			Preferences:      UserPreferences{},
			TopicInterests:   map[string]float64{},
			EntityInterests:  map[string]float64{},
			TermFrequency:    map[string]int{},
		}

		// Save initial
		err := store.SaveProfile(ctx, profile)
		require.NoError(t, err)

		// Update
		profile.SearchCount = 50
		profile.TopicInterests["ai"] = 0.9
		err = store.SaveProfile(ctx, profile)
		assert.NoError(t, err)

		// Verify update
		loaded, err := store.LoadProfile(ctx, "user2")
		assert.NoError(t, err)
		assert.Equal(t, int64(50), loaded.SearchCount)
		assert.Equal(t, 0.9, loaded.TopicInterests["ai"])
	})

	t.Run("DeleteProfile", func(t *testing.T) {
		profile := &UserProfile{
			ID:              "user3",
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			TopicInterests:  map[string]float64{},
			EntityInterests: map[string]float64{},
			TermFrequency:   map[string]int{},
		}

		// Save
		err := store.SaveProfile(ctx, profile)
		require.NoError(t, err)

		// Delete
		err = store.DeleteProfile(ctx, "user3")
		assert.NoError(t, err)

		// Verify deleted
		_, err = store.LoadProfile(ctx, "user3")
		assert.Error(t, err)
	})
}

func TestPersistenceIntegration(t *testing.T) {
	// Test that reopening the database preserves data
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_persistence.db")

	ctx := context.Background()

	// Create and populate database
	{
		store, err := NewBoltFeedbackStore(dbPath)
		require.NoError(t, err)

		// Save interaction
		interaction := &Interaction{
			ID:         "persist_int_1",
			UserID:     "persist_user",
			SessionID:  "persist_session",
			QueryID:    "persist_query",
			DocumentID: "persist_doc",
			Type:       InteractionTypeClick,
			Timestamp:  time.Now(),
		}
		err = store.SaveInteraction(ctx, interaction)
		require.NoError(t, err)

		// Save query feedback
		queryFeedback := &QueryFeedback{
			QueryID:     "persist_query",
			Query:       "persistence test",
			UserID:      "persist_user",
			ResultCount: 5,
			ClickCount:  2,
		}
		err = store.SaveQueryFeedback(ctx, queryFeedback)
		require.NoError(t, err)

		store.Close()
	}

	// Reopen and verify data persisted
	{
		store, err := NewBoltFeedbackStore(dbPath)
		require.NoError(t, err)
		defer store.Close()

		// Load interaction
		filter := InteractionFilter{
			UserID: "persist_user",
		}
		interactions, err := store.LoadInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 1)
		assert.Equal(t, "persist_int_1", interactions[0].ID)

		// Load query feedback
		queryFeedback, err := store.LoadQueryFeedback(ctx, "persist_query")
		assert.NoError(t, err)
		assert.Equal(t, "persistence test", queryFeedback.Query)
		assert.Equal(t, 2, queryFeedback.ClickCount)
	}

	// Clean up
	os.Remove(dbPath)
}

func TestConcurrentAccess(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_concurrent.db")

	store, err := NewBoltFeedbackStore(dbPath)
	require.NoError(t, err)
	defer store.Close()

	ctx := context.Background()

	// Test concurrent writes
	var wg sync.WaitGroup
	numGoroutines := 10
	interactionsPerGoroutine := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for j := 0; j < interactionsPerGoroutine; j++ {
				interaction := &Interaction{
					ID:         fmt.Sprintf("concurrent_%d_%d", goroutineID, j),
					UserID:     fmt.Sprintf("user_%d", goroutineID),
					SessionID:  fmt.Sprintf("session_%d", goroutineID),
					QueryID:    fmt.Sprintf("query_%d_%d", goroutineID, j),
					DocumentID: fmt.Sprintf("doc_%d_%d", goroutineID, j),
					Type:       InteractionTypeClick,
					Timestamp:  time.Now(),
				}

				err := store.SaveInteraction(ctx, interaction)
				assert.NoError(t, err)
			}
		}(i)
	}

	wg.Wait()

	// Verify all interactions were saved
	for i := 0; i < numGoroutines; i++ {
		filter := InteractionFilter{
			UserID: fmt.Sprintf("user_%d", i),
		}
		interactions, err := store.LoadInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, interactionsPerGoroutine)
	}
}
