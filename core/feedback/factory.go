package feedback

import (
	"fmt"
	"time"
)

// ManagerConfig contains configuration for the feedback manager
type ManagerConfig struct {
	// Session configuration
	SessionTimeout time.Duration

	// Cache configuration
	MaxCacheSize int

	// CTR tracking configuration
	EnableCTRTracking bool
	CTRMaxDataSize    int
	CTRDecayFactor    float64

	// Learning configuration
	EnableLearning      bool
	MaxTrainingDataSize int

	// Persistence configuration (for future use)
	EnablePersistence bool
	StoragePath       string
}

// DefaultManagerConfig returns default configuration
func DefaultManagerConfig() ManagerConfig {
	return ManagerConfig{
		SessionTimeout:      30 * time.Minute,
		MaxCacheSize:        10000,
		EnableCTRTracking:   true,
		CTRMaxDataSize:      100000,
		CTRDecayFactor:      0.95,
		EnableLearning:      true,
		MaxTrainingDataSize: 10000,
		EnablePersistence:   false,
		StoragePath:         "",
	}
}

// Manager represents the complete feedback management system
type Manager struct {
	Collector      Collector
	SessionManager SessionManager
	ProfileManager ProfileManager
	LearningEngine LearningEngine
	Analyzer       FeedbackAnalyzer
	CTRTracker     CTRTracker
	store          *BoltFeedbackStore // For cleanup
}

// NewManager creates a new feedback manager with the given configuration
func NewManager(config ManagerConfig) (*Manager, error) {
	var err error
	var feedbackStore *BoltFeedbackStore
	var sessionStore *BoltSessionStore
	var profileStore *BoltProfileStore

	// Open persistent storage if enabled
	if config.EnablePersistence && config.StoragePath != "" {
		feedbackStore, err = NewBoltFeedbackStore(config.StoragePath)
		if err != nil {
			return nil, fmt.Errorf("failed to create feedback store: %w", err)
		}

		// Share the same DB for all stores
		sessionStore = NewBoltSessionStore(feedbackStore.db)
		profileStore = NewBoltProfileStore(feedbackStore.db)
	}

	// Create collector
	var collector Collector
	if config.EnablePersistence && feedbackStore != nil {
		collector = NewPersistentCollector(feedbackStore)
	} else {
		collector = NewMemoryCollector()
	}

	// Create session manager
	var sessionManager SessionManager
	if config.EnablePersistence && sessionStore != nil {
		sessionManager = NewPersistentSessionManager(sessionStore, config.SessionTimeout)
	} else {
		sessionManager = NewMemorySessionManager(config.SessionTimeout)
	}

	// Create profile manager
	var profileManager ProfileManager
	if config.EnablePersistence && profileStore != nil {
		profileManager = NewPersistentProfileManager(profileStore)
	} else {
		profileManager = NewMemoryProfileManager()
	}

	// Create learning engine
	var learningEngine LearningEngine
	if config.EnableLearning {
		learningEngine = NewSimpleLearningEngine()
	}

	// Create analyzer
	analyzer := NewFeedbackAnalyzer(collector)

	// Create CTR tracker
	var ctrTracker CTRTracker
	if config.EnableCTRTracking {
		ctrTracker = NewMemoryCTRTracker()
	}

	manager := &Manager{
		Collector:      collector,
		SessionManager: sessionManager,
		ProfileManager: profileManager,
		LearningEngine: learningEngine,
		Analyzer:       analyzer,
		CTRTracker:     ctrTracker,
	}

	// Store reference to feedbackStore for cleanup
	if feedbackStore != nil {
		manager.store = feedbackStore
	}

	return manager, nil
}

// Close closes all resources
func (m *Manager) Close() error {
	if m.store != nil {
		return m.store.Close()
	}
	return nil
}
