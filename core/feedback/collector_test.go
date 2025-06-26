package feedback

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryCollector(t *testing.T) {
	ctx := context.Background()
	collector := NewMemoryCollector()

	t.Run("RecordInteraction", func(t *testing.T) {
		interaction := &Interaction{
			UserID:       "user1",
			SessionID:    "session1",
			QueryID:      "query1",
			Query:        "test query",
			DocumentID:   "doc1",
			CollectionID: "collection1",
			Type:         InteractionTypeClick,
			Position:     1,
			Timestamp:    time.Now(),
		}

		err := collector.RecordInteraction(ctx, interaction)
		assert.NoError(t, err)
		assert.NotEmpty(t, interaction.ID)
	})

	t.Run("GetInteractions_ByUser", func(t *testing.T) {
		// Record multiple interactions
		for i := 0; i < 5; i++ {
			interaction := &Interaction{
				UserID:       "user2",
				SessionID:    "session2",
				QueryID:      "query2",
				DocumentID:   "doc" + string(rune('1'+i)),
				CollectionID: "collection1",
				Type:         InteractionTypeClick,
				Position:     i + 1,
				Timestamp:    time.Now(),
			}
			err := collector.RecordInteraction(ctx, interaction)
			require.NoError(t, err)
		}

		// Get interactions by user
		filter := InteractionFilter{
			UserID: "user2",
		}
		interactions, err := collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 5)
		
		// Verify all interactions belong to user2
		for _, interaction := range interactions {
			assert.Equal(t, "user2", interaction.UserID)
		}
	})

	t.Run("GetInteractions_WithLimit", func(t *testing.T) {
		filter := InteractionFilter{
			UserID: "user2",
			Limit:  3,
		}
		interactions, err := collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 3)
	})

	t.Run("GetInteractions_ByType", func(t *testing.T) {
		// Record different types of interactions
		dwellInteraction := &Interaction{
			UserID:       "user3",
			SessionID:    "session3",
			QueryID:      "query3",
			DocumentID:   "doc1",
			CollectionID: "collection1",
			Type:         InteractionTypeDwell,
			Value:        45.5, // dwell time in seconds
			Position:     1,
			Timestamp:    time.Now(),
		}
		err := collector.RecordInteraction(ctx, dwellInteraction)
		require.NoError(t, err)

		ratingInteraction := &Interaction{
			UserID:       "user3",
			SessionID:    "session3",
			QueryID:      "query3",
			DocumentID:   "doc1",
			CollectionID: "collection1",
			Type:         InteractionTypeRating,
			Value:        4.5, // rating
			Position:     1,
			Timestamp:    time.Now(),
		}
		err = collector.RecordInteraction(ctx, ratingInteraction)
		require.NoError(t, err)

		// Filter by type
		filter := InteractionFilter{
			UserID: "user3",
			Type:   InteractionTypeDwell,
		}
		interactions, err := collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Len(t, interactions, 1)
		assert.Equal(t, InteractionTypeDwell, interactions[0].Type)
		assert.Equal(t, 45.5, interactions[0].Value)
	})

	t.Run("GetInteractions_ByTimeRange", func(t *testing.T) {
		now := time.Now()
		yesterday := now.Add(-24 * time.Hour)
		tomorrow := now.Add(24 * time.Hour)

		filter := InteractionFilter{
			StartTime: &yesterday,
			EndTime:   &tomorrow,
		}
		interactions, err := collector.GetInteractions(ctx, filter)
		assert.NoError(t, err)
		assert.Greater(t, len(interactions), 0)
	})

	t.Run("RecordQuery", func(t *testing.T) {
		queryFeedback := &QueryFeedback{
			QueryID:          "query1",
			Query:            "test query",
			UserID:           "user1",
			SessionID:        "session1",
			Timestamp:        time.Now(),
			ResultCount:      10,
			ClickCount:       2,
			AvgDwellTime:     30.5,
			AvgClickPosition: 2.5,
			Satisfied:        true,
		}

		err := collector.RecordQuery(ctx, queryFeedback)
		assert.NoError(t, err)

		// Retrieve query feedback
		retrieved, err := collector.GetQueryFeedback(ctx, "query1")
		assert.NoError(t, err)
		assert.Equal(t, queryFeedback.Query, retrieved.Query)
		assert.Equal(t, queryFeedback.ClickCount, retrieved.ClickCount)
		assert.Equal(t, queryFeedback.Satisfied, retrieved.Satisfied)
	})

	t.Run("DocumentFeedback", func(t *testing.T) {
		// Record interactions for a document
		for i := 0; i < 10; i++ {
			interaction := &Interaction{
				UserID:       "user" + string(rune('1'+i)),
				SessionID:    "session" + string(rune('1'+i)),
				QueryID:      "query" + string(rune('1'+i)),
				DocumentID:   "doc_popular",
				CollectionID: "collection1",
				Type:         InteractionTypeClick,
				Position:     1,
				Timestamp:    time.Now(),
			}
			err := collector.RecordInteraction(ctx, interaction)
			require.NoError(t, err)
		}

		// Check document feedback was updated
		docFeedback, err := collector.GetDocumentFeedback(ctx, "doc_popular")
		assert.NoError(t, err)
		assert.Equal(t, int64(10), docFeedback.TotalViews)
		assert.Equal(t, int64(10), docFeedback.TotalClicks)
		assert.Equal(t, 1.0, docFeedback.ClickThroughRate)
	})
}

func TestPersistentCollector(t *testing.T) {
	// Create a mock store for testing
	store := &mockFeedbackStore{
		interactions:     make(map[string]*Interaction),
		queryFeedback:    make(map[string]*QueryFeedback),
		documentFeedback: make(map[string]*DocumentFeedback),
	}
	
	ctx := context.Background()
	collector := NewPersistentCollector(store)

	t.Run("RecordInteraction_Persists", func(t *testing.T) {
		interaction := &Interaction{
			UserID:       "user1",
			SessionID:    "session1",
			QueryID:      "query1",
			DocumentID:   "doc1",
			CollectionID: "collection1",
			Type:         InteractionTypeClick,
			Position:     1,
			Timestamp:    time.Now(),
		}

		err := collector.RecordInteraction(ctx, interaction)
		assert.NoError(t, err)
		
		// Verify it was persisted to store
		assert.Contains(t, store.interactions, interaction.ID)
	})

	t.Run("RecordQuery_Persists", func(t *testing.T) {
		queryFeedback := &QueryFeedback{
			QueryID:     "query1",
			Query:       "test query",
			UserID:      "user1",
			SessionID:   "session1",
			Timestamp:   time.Now(),
			ResultCount: 10,
		}

		err := collector.RecordQuery(ctx, queryFeedback)
		assert.NoError(t, err)
		
		// Verify it was persisted to store
		assert.Contains(t, store.queryFeedback, queryFeedback.QueryID)
	})
}

// Mock implementation of FeedbackStore for testing
type mockFeedbackStore struct {
	interactions     map[string]*Interaction
	queryFeedback    map[string]*QueryFeedback
	documentFeedback map[string]*DocumentFeedback
}

func (s *mockFeedbackStore) SaveInteraction(ctx context.Context, interaction *Interaction) error {
	s.interactions[interaction.ID] = interaction
	return nil
}

func (s *mockFeedbackStore) LoadInteractions(ctx context.Context, filter InteractionFilter) ([]*Interaction, error) {
	var results []*Interaction
	for _, interaction := range s.interactions {
		if matchesFilter(interaction, filter) {
			results = append(results, interaction)
		}
	}
	return results, nil
}

func (s *mockFeedbackStore) SaveQueryFeedback(ctx context.Context, feedback *QueryFeedback) error {
	s.queryFeedback[feedback.QueryID] = feedback
	return nil
}

func (s *mockFeedbackStore) LoadQueryFeedback(ctx context.Context, queryID string) (*QueryFeedback, error) {
	feedback, exists := s.queryFeedback[queryID]
	if !exists {
		return nil, nil
	}
	return feedback, nil
}

func (s *mockFeedbackStore) SaveDocumentFeedback(ctx context.Context, feedback *DocumentFeedback) error {
	s.documentFeedback[feedback.DocumentID] = feedback
	return nil
}

func (s *mockFeedbackStore) LoadDocumentFeedback(ctx context.Context, documentID string) (*DocumentFeedback, error) {
	feedback, exists := s.documentFeedback[documentID]
	if !exists {
		return nil, nil
	}
	return feedback, nil
}

func matchesFilter(interaction *Interaction, filter InteractionFilter) bool {
	if filter.UserID != "" && interaction.UserID != filter.UserID {
		return false
	}
	if filter.SessionID != "" && interaction.SessionID != filter.SessionID {
		return false
	}
	if filter.QueryID != "" && interaction.QueryID != filter.QueryID {
		return false
	}
	if filter.DocumentID != "" && interaction.DocumentID != filter.DocumentID {
		return false
	}
	if filter.Type != "" && interaction.Type != filter.Type {
		return false
	}
	return true
}