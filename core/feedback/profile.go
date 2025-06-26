package feedback

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// memoryProfileManager implements ProfileManager with in-memory storage
type memoryProfileManager struct {
	mu       sync.RWMutex
	profiles map[string]*UserProfile
}

// NewMemoryProfileManager creates a new in-memory profile manager
func NewMemoryProfileManager() ProfileManager {
	return &memoryProfileManager{
		profiles: make(map[string]*UserProfile),
	}
}

func (m *memoryProfileManager) GetProfile(ctx context.Context, userID string) (*UserProfile, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	profile, exists := m.profiles[userID]
	if !exists {
		return nil, fmt.Errorf("profile not found for user: %s", userID)
	}

	return profile, nil
}

func (m *memoryProfileManager) CreateProfile(ctx context.Context, userID string) (*UserProfile, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.profiles[userID]; exists {
		return nil, fmt.Errorf("profile already exists for user: %s", userID)
	}

	profile := &UserProfile{
		ID:               userID,
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
		SearchCount:      0,
		InteractionCount: 0,
		Preferences: UserPreferences{
			PreferredLanguages: []string{"en"},
			SearchRecency:      0.5,
			SearchDiversity:    0.3,
			TopicWeights:       make(map[string]float64),
			SourceWeights:      make(map[string]float64),
			VectorWeight:       0.5,
			TextWeight:         0.5,
		},
		TopicInterests:  make(map[string]float64),
		EntityInterests: make(map[string]float64),
		TermFrequency:   make(map[string]int),
		Metadata:        make(map[string]interface{}),
	}

	m.profiles[userID] = profile
	return profile, nil
}

func (m *memoryProfileManager) UpdateProfile(ctx context.Context, profile *UserProfile) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.profiles[profile.ID]; !exists {
		return fmt.Errorf("profile not found for user: %s", profile.ID)
	}

	profile.UpdatedAt = time.Now()
	m.profiles[profile.ID] = profile
	return nil
}

func (m *memoryProfileManager) UpdatePreferences(ctx context.Context, userID string, preferences UserPreferences) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	profile, exists := m.profiles[userID]
	if !exists {
		return fmt.Errorf("profile not found for user: %s", userID)
	}

	profile.Preferences = preferences
	profile.UpdatedAt = time.Now()
	m.profiles[userID] = profile

	return nil
}

func (m *memoryProfileManager) IncrementInterests(ctx context.Context, userID string, topics map[string]float64, entities map[string]float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	profile, exists := m.profiles[userID]
	if !exists {
		// Auto-create profile if it doesn't exist
		profile = &UserProfile{
			ID:              userID,
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			TopicInterests:  make(map[string]float64),
			EntityInterests: make(map[string]float64),
			TermFrequency:   make(map[string]int),
			Preferences: UserPreferences{
				PreferredLanguages: []string{"en"},
				SearchRecency:      0.5,
				SearchDiversity:    0.3,
				TopicWeights:       make(map[string]float64),
				SourceWeights:      make(map[string]float64),
				VectorWeight:       0.5,
				TextWeight:         0.5,
			},
		}
		m.profiles[userID] = profile
	}

	// Update topic interests with decay
	decayFactor := 0.95
	for topic, currentScore := range profile.TopicInterests {
		profile.TopicInterests[topic] = currentScore * decayFactor
	}
	for topic, increment := range topics {
		profile.TopicInterests[topic] = profile.TopicInterests[topic] + increment
	}

	// Update entity interests with decay
	for entity, currentScore := range profile.EntityInterests {
		profile.EntityInterests[entity] = currentScore * decayFactor
	}
	for entity, increment := range entities {
		profile.EntityInterests[entity] = profile.EntityInterests[entity] + increment
	}

	// Clean up interests below threshold
	threshold := 0.01
	for topic, score := range profile.TopicInterests {
		if score < threshold {
			delete(profile.TopicInterests, topic)
		}
	}
	for entity, score := range profile.EntityInterests {
		if score < threshold {
			delete(profile.EntityInterests, entity)
		}
	}

	profile.UpdatedAt = time.Now()
	m.profiles[userID] = profile

	return nil
}

func (m *memoryProfileManager) GetTopInterests(ctx context.Context, userID string, limit int) (topics map[string]float64, entities map[string]float64, err error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	profile, exists := m.profiles[userID]
	if !exists {
		return make(map[string]float64), make(map[string]float64), nil
	}

	// Sort topics by interest score
	type interest struct {
		name  string
		score float64
	}

	var topicList []interest
	for topic, score := range profile.TopicInterests {
		topicList = append(topicList, interest{topic, score})
	}
	sort.Slice(topicList, func(i, j int) bool {
		return topicList[i].score > topicList[j].score
	})

	topics = make(map[string]float64)
	for i := 0; i < limit && i < len(topicList); i++ {
		topics[topicList[i].name] = topicList[i].score
	}

	// Sort entities by interest score
	var entityList []interest
	for entity, score := range profile.EntityInterests {
		entityList = append(entityList, interest{entity, score})
	}
	sort.Slice(entityList, func(i, j int) bool {
		return entityList[i].score > entityList[j].score
	})

	entities = make(map[string]float64)
	for i := 0; i < limit && i < len(entityList); i++ {
		entities[entityList[i].name] = entityList[i].score
	}

	return topics, entities, nil
}

// persistentProfileManager implements ProfileManager with persistent storage
type persistentProfileManager struct {
	memoryProfileManager
	store ProfileStore
}

// ProfileStore defines the interface for persistent profile storage
type ProfileStore interface {
	SaveProfile(ctx context.Context, profile *UserProfile) error
	LoadProfile(ctx context.Context, userID string) (*UserProfile, error)
	DeleteProfile(ctx context.Context, userID string) error
}

// NewPersistentProfileManager creates a profile manager with persistent storage
func NewPersistentProfileManager(store ProfileStore) ProfileManager {
	return &persistentProfileManager{
		memoryProfileManager: memoryProfileManager{
			profiles: make(map[string]*UserProfile),
		},
		store: store,
	}
}

func (m *persistentProfileManager) GetProfile(ctx context.Context, userID string) (*UserProfile, error) {
	// Try memory first
	profile, err := m.memoryProfileManager.GetProfile(ctx, userID)
	if err == nil {
		return profile, nil
	}

	// Load from store
	profile, err = m.store.LoadProfile(ctx, userID)
	if err != nil {
		return nil, err
	}

	// Cache in memory
	m.mu.Lock()
	m.profiles[userID] = profile
	m.mu.Unlock()

	return profile, nil
}

func (m *persistentProfileManager) CreateProfile(ctx context.Context, userID string) (*UserProfile, error) {
	profile, err := m.memoryProfileManager.CreateProfile(ctx, userID)
	if err != nil {
		return nil, err
	}

	// Persist
	if err := m.store.SaveProfile(ctx, profile); err != nil {
		// Remove from memory if persistence fails
		m.mu.Lock()
		delete(m.profiles, userID)
		m.mu.Unlock()
		return nil, err
	}

	return profile, nil
}

func (m *persistentProfileManager) UpdateProfile(ctx context.Context, profile *UserProfile) error {
	if err := m.memoryProfileManager.UpdateProfile(ctx, profile); err != nil {
		return err
	}

	return m.store.SaveProfile(ctx, profile)
}

func (m *persistentProfileManager) UpdatePreferences(ctx context.Context, userID string, preferences UserPreferences) error {
	if err := m.memoryProfileManager.UpdatePreferences(ctx, userID, preferences); err != nil {
		return err
	}

	// Get the updated profile and persist
	m.mu.RLock()
	profile := m.profiles[userID]
	m.mu.RUnlock()

	if profile != nil {
		return m.store.SaveProfile(ctx, profile)
	}

	return nil
}

func (m *persistentProfileManager) IncrementInterests(ctx context.Context, userID string, topics map[string]float64, entities map[string]float64) error {
	if err := m.memoryProfileManager.IncrementInterests(ctx, userID, topics, entities); err != nil {
		return err
	}

	// Get the updated profile and persist
	m.mu.RLock()
	profile := m.profiles[userID]
	m.mu.RUnlock()

	if profile != nil {
		return m.store.SaveProfile(ctx, profile)
	}

	return nil
}
