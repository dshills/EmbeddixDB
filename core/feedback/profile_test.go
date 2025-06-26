package feedback

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryProfileManager(t *testing.T) {
	ctx := context.Background()
	manager := NewMemoryProfileManager()

	t.Run("CreateProfile", func(t *testing.T) {
		profile, err := manager.CreateProfile(ctx, "user1")
		assert.NoError(t, err)
		assert.Equal(t, "user1", profile.ID)
		assert.Equal(t, int64(0), profile.SearchCount)
		assert.Equal(t, int64(0), profile.InteractionCount)
		assert.NotNil(t, profile.Preferences)
		assert.Equal(t, 0.5, profile.Preferences.VectorWeight)
		assert.Equal(t, 0.5, profile.Preferences.TextWeight)
		assert.NotNil(t, profile.TopicInterests)
		assert.NotNil(t, profile.EntityInterests)
		assert.NotNil(t, profile.TermFrequency)
	})

	t.Run("CreateProfile_AlreadyExists", func(t *testing.T) {
		// Create a profile
		_, err := manager.CreateProfile(ctx, "user2")
		require.NoError(t, err)

		// Try to create again
		_, err = manager.CreateProfile(ctx, "user2")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already exists")
	})

	t.Run("GetProfile", func(t *testing.T) {
		// Create a profile
		created, err := manager.CreateProfile(ctx, "user3")
		require.NoError(t, err)

		// Get the profile
		retrieved, err := manager.GetProfile(ctx, "user3")
		assert.NoError(t, err)
		assert.Equal(t, created.ID, retrieved.ID)
		assert.Equal(t, created.CreatedAt.Unix(), retrieved.CreatedAt.Unix())
	})

	t.Run("GetProfile_NotFound", func(t *testing.T) {
		_, err := manager.GetProfile(ctx, "non-existent-user")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("UpdateProfile", func(t *testing.T) {
		// Create a profile
		profile, err := manager.CreateProfile(ctx, "user4")
		require.NoError(t, err)

		// Update profile
		profile.SearchCount = 100
		profile.InteractionCount = 50
		err = manager.UpdateProfile(ctx, profile)
		assert.NoError(t, err)

		// Verify update
		retrieved, err := manager.GetProfile(ctx, "user4")
		assert.NoError(t, err)
		assert.Equal(t, int64(100), retrieved.SearchCount)
		assert.Equal(t, int64(50), retrieved.InteractionCount)
		assert.True(t, retrieved.UpdatedAt.After(retrieved.CreatedAt))
	})

	t.Run("UpdatePreferences", func(t *testing.T) {
		// Create a profile
		_, err := manager.CreateProfile(ctx, "user5")
		require.NoError(t, err)

		// Update preferences
		newPrefs := UserPreferences{
			PreferredLanguages: []string{"en", "es"},
			SearchRecency:      0.7,
			SearchDiversity:    0.4,
			TopicWeights: map[string]float64{
				"ai":   0.8,
				"tech": 0.6,
			},
			SourceWeights: map[string]float64{
				"arxiv":     0.9,
				"wikipedia": 0.7,
			},
			VectorWeight: 0.6,
			TextWeight:   0.4,
		}

		err = manager.UpdatePreferences(ctx, "user5", newPrefs)
		assert.NoError(t, err)

		// Verify update
		profile, err := manager.GetProfile(ctx, "user5")
		assert.NoError(t, err)
		assert.Equal(t, newPrefs.PreferredLanguages, profile.Preferences.PreferredLanguages)
		assert.Equal(t, newPrefs.SearchRecency, profile.Preferences.SearchRecency)
		assert.Equal(t, newPrefs.VectorWeight, profile.Preferences.VectorWeight)
		assert.Equal(t, 0.8, profile.Preferences.TopicWeights["ai"])
	})

	t.Run("IncrementInterests", func(t *testing.T) {
		// Create a profile
		_, err := manager.CreateProfile(ctx, "user6")
		require.NoError(t, err)

		// Increment interests
		topics := map[string]float64{
			"machine_learning": 0.5,
			"deep_learning":    0.3,
			"nlp":              0.2,
		}
		entities := map[string]float64{
			"OpenAI":   0.4,
			"DeepMind": 0.3,
		}

		err = manager.IncrementInterests(ctx, "user6", topics, entities)
		assert.NoError(t, err)

		// Verify interests were updated
		profile, err := manager.GetProfile(ctx, "user6")
		assert.NoError(t, err)
		assert.Equal(t, 0.5, profile.TopicInterests["machine_learning"])
		assert.Equal(t, 0.3, profile.TopicInterests["deep_learning"])
		assert.Equal(t, 0.4, profile.EntityInterests["OpenAI"])

		// Increment again to test accumulation
		topics2 := map[string]float64{
			"machine_learning": 0.3,
			"computer_vision":  0.4,
		}
		err = manager.IncrementInterests(ctx, "user6", topics2, nil)
		assert.NoError(t, err)

		// Verify accumulation with decay
		profile, err = manager.GetProfile(ctx, "user6")
		assert.NoError(t, err)
		// machine_learning should be ~0.5*0.95 + 0.3 = 0.775
		assert.InDelta(t, 0.775, profile.TopicInterests["machine_learning"], 0.001)
		assert.Equal(t, 0.4, profile.TopicInterests["computer_vision"])
	})

	t.Run("IncrementInterests_AutoCreateProfile", func(t *testing.T) {
		// Try to increment interests for non-existent user
		topics := map[string]float64{"ai": 0.5}
		err := manager.IncrementInterests(ctx, "user7", topics, nil)
		assert.NoError(t, err)

		// Verify profile was auto-created
		profile, err := manager.GetProfile(ctx, "user7")
		assert.NoError(t, err)
		assert.Equal(t, 0.5, profile.TopicInterests["ai"])
	})

	t.Run("GetTopInterests", func(t *testing.T) {
		// Create a profile with various interests
		_, err := manager.CreateProfile(ctx, "user8")
		require.NoError(t, err)

		// Add many interests
		topics := map[string]float64{
			"ai":               0.9,
			"machine_learning": 0.8,
			"deep_learning":    0.7,
			"nlp":              0.6,
			"computer_vision":  0.5,
			"robotics":         0.4,
		}
		entities := map[string]float64{
			"OpenAI":    0.9,
			"Google":    0.8,
			"Microsoft": 0.7,
			"Meta":      0.6,
			"Amazon":    0.5,
		}

		err = manager.IncrementInterests(ctx, "user8", topics, entities)
		assert.NoError(t, err)

		// Get top 3 interests
		topTopics, topEntities, err := manager.GetTopInterests(ctx, "user8", 3)
		assert.NoError(t, err)
		assert.Len(t, topTopics, 3)
		assert.Len(t, topEntities, 3)

		// Verify ordering (highest scores first)
		assert.Contains(t, topTopics, "ai")
		assert.Contains(t, topTopics, "machine_learning")
		assert.Contains(t, topTopics, "deep_learning")
		assert.Contains(t, topEntities, "OpenAI")
		assert.Contains(t, topEntities, "Google")
		assert.Contains(t, topEntities, "Microsoft")
	})
}

func TestPersistentProfileManager(t *testing.T) {
	// Create a mock store
	store := &mockProfileStore{
		profiles: make(map[string]*UserProfile),
	}
	
	ctx := context.Background()
	manager := NewPersistentProfileManager(store)

	t.Run("CreateProfile_Persists", func(t *testing.T) {
		profile, err := manager.CreateProfile(ctx, "user1")
		assert.NoError(t, err)
		
		// Verify it was persisted to store
		assert.Contains(t, store.profiles, profile.ID)
	})

	t.Run("GetProfile_LoadsFromStore", func(t *testing.T) {
		// Add a profile directly to store
		testProfile := &UserProfile{
			ID:               "user2",
			CreatedAt:        time.Now(),
			UpdatedAt:        time.Now(),
			SearchCount:      42,
			InteractionCount: 24,
			Preferences:      UserPreferences{VectorWeight: 0.7, TextWeight: 0.3},
			TopicInterests:   map[string]float64{"test": 0.5},
			EntityInterests:  map[string]float64{},
			TermFrequency:    map[string]int{},
		}
		store.profiles["user2"] = testProfile

		// Get profile through manager
		profile, err := manager.GetProfile(ctx, "user2")
		assert.NoError(t, err)
		assert.Equal(t, int64(42), profile.SearchCount)
		assert.Equal(t, 0.7, profile.Preferences.VectorWeight)
	})

	t.Run("UpdateProfile_Persists", func(t *testing.T) {
		// Create a profile
		profile, err := manager.CreateProfile(ctx, "user3")
		require.NoError(t, err)

		// Update it
		profile.SearchCount = 100
		err = manager.UpdateProfile(ctx, profile)
		assert.NoError(t, err)

		// Verify it was persisted
		stored := store.profiles["user3"]
		assert.Equal(t, int64(100), stored.SearchCount)
	})

	t.Run("UpdatePreferences_Persists", func(t *testing.T) {
		// Create a profile
		_, err := manager.CreateProfile(ctx, "user4")
		require.NoError(t, err)

		// Update preferences
		newPrefs := UserPreferences{
			VectorWeight: 0.8,
			TextWeight:   0.2,
		}
		err = manager.UpdatePreferences(ctx, "user4", newPrefs)
		assert.NoError(t, err)

		// Verify it was persisted
		stored := store.profiles["user4"]
		assert.Equal(t, 0.8, stored.Preferences.VectorWeight)
	})

	t.Run("IncrementInterests_Persists", func(t *testing.T) {
		// Create a profile
		_, err := manager.CreateProfile(ctx, "user5")
		require.NoError(t, err)

		// Increment interests
		topics := map[string]float64{"ai": 0.5}
		err = manager.IncrementInterests(ctx, "user5", topics, nil)
		assert.NoError(t, err)

		// Verify it was persisted
		stored := store.profiles["user5"]
		assert.Equal(t, 0.5, stored.TopicInterests["ai"])
	})
}

// Mock implementation of ProfileStore for testing
type mockProfileStore struct {
	profiles map[string]*UserProfile
}

func (s *mockProfileStore) SaveProfile(ctx context.Context, profile *UserProfile) error {
	s.profiles[profile.ID] = profile
	return nil
}

func (s *mockProfileStore) LoadProfile(ctx context.Context, userID string) (*UserProfile, error) {
	profile, exists := s.profiles[userID]
	if !exists {
		return nil, nil
	}
	return profile, nil
}

func (s *mockProfileStore) DeleteProfile(ctx context.Context, userID string) error {
	delete(s.profiles, userID)
	return nil
}