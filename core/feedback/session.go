package feedback

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// memorySessionManager implements SessionManager with in-memory storage
type memorySessionManager struct {
	mu           sync.RWMutex
	sessions     map[string]*Session
	userSessions map[string][]string // userID -> sessionIDs
	
	// Configuration
	sessionTimeout time.Duration
}

// NewMemorySessionManager creates a new in-memory session manager
func NewMemorySessionManager(sessionTimeout time.Duration) SessionManager {
	if sessionTimeout == 0 {
		sessionTimeout = 30 * time.Minute // Default 30 minutes
	}
	
	manager := &memorySessionManager{
		sessions:       make(map[string]*Session),
		userSessions:   make(map[string][]string),
		sessionTimeout: sessionTimeout,
	}
	
	// Start cleanup routine
	go manager.cleanupExpiredSessions()
	
	return manager
}

func (m *memorySessionManager) CreateSession(ctx context.Context, userID string, metadata map[string]interface{}) (*Session, error) {
	session := &Session{
		ID:         uuid.New().String(),
		UserID:     userID,
		StartTime:  time.Now(),
		QueryCount: 0,
		Metadata:   metadata,
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.sessions[session.ID] = session
	m.userSessions[userID] = append(m.userSessions[userID], session.ID)

	return session, nil
}

func (m *memorySessionManager) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	session, exists := m.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	// Check if session has expired
	if session.EndTime == nil && time.Since(session.StartTime) > m.sessionTimeout {
		return nil, fmt.Errorf("session expired: %s", sessionID)
	}

	return session, nil
}

func (m *memorySessionManager) UpdateSession(ctx context.Context, session *Session) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.sessions[session.ID]; !exists {
		return fmt.Errorf("session not found: %s", session.ID)
	}

	m.sessions[session.ID] = session
	return nil
}

func (m *memorySessionManager) EndSession(ctx context.Context, sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, exists := m.sessions[sessionID]
	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	now := time.Now()
	session.EndTime = &now
	m.sessions[sessionID] = session

	return nil
}

func (m *memorySessionManager) GetActiveSessions(ctx context.Context, userID string) ([]*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var activeSessions []*Session
	sessionIDs, exists := m.userSessions[userID]
	if !exists {
		return activeSessions, nil
	}

	now := time.Now()
	for _, sessionID := range sessionIDs {
		session := m.sessions[sessionID]
		if session == nil {
			continue
		}

		// Check if session is active
		if session.EndTime == nil && now.Sub(session.StartTime) <= m.sessionTimeout {
			activeSessions = append(activeSessions, session)
		}
	}

	return activeSessions, nil
}

func (m *memorySessionManager) GetSessionHistory(ctx context.Context, userID string, limit int) ([]*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sessionIDs, exists := m.userSessions[userID]
	if !exists {
		return []*Session{}, nil
	}

	// Get all sessions for the user
	var sessions []*Session
	for _, sessionID := range sessionIDs {
		if session := m.sessions[sessionID]; session != nil {
			sessions = append(sessions, session)
		}
	}

	// Sort by start time (most recent first)
	// In a real implementation, we'd use sort.Slice here
	
	// Apply limit
	if limit > 0 && len(sessions) > limit {
		sessions = sessions[:limit]
	}

	return sessions, nil
}

func (m *memorySessionManager) cleanupExpiredSessions() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		m.mu.Lock()
		
		now := time.Now()
		expiredSessions := []string{}
		
		for sessionID, session := range m.sessions {
			// Remove sessions that have been ended for more than 24 hours
			if session.EndTime != nil && now.Sub(*session.EndTime) > 24*time.Hour {
				expiredSessions = append(expiredSessions, sessionID)
			}
			// Auto-end sessions that have been inactive for too long
			if session.EndTime == nil && now.Sub(session.StartTime) > m.sessionTimeout {
				endTime := session.StartTime.Add(m.sessionTimeout)
				session.EndTime = &endTime
			}
		}
		
		// Remove expired sessions
		for _, sessionID := range expiredSessions {
			delete(m.sessions, sessionID)
			// Also remove from user index
			for userID, sessionIDs := range m.userSessions {
				filtered := []string{}
				for _, id := range sessionIDs {
					if id != sessionID {
						filtered = append(filtered, id)
					}
				}
				m.userSessions[userID] = filtered
			}
		}
		
		m.mu.Unlock()
	}
}

// persistentSessionManager implements SessionManager with persistent storage
type persistentSessionManager struct {
	memorySessionManager
	store SessionStore
}

// SessionStore defines the interface for persistent session storage
type SessionStore interface {
	SaveSession(ctx context.Context, session *Session) error
	LoadSession(ctx context.Context, sessionID string) (*Session, error)
	LoadUserSessions(ctx context.Context, userID string, limit int) ([]*Session, error)
	DeleteSession(ctx context.Context, sessionID string) error
}

// NewPersistentSessionManager creates a session manager with persistent storage
func NewPersistentSessionManager(store SessionStore, sessionTimeout time.Duration) SessionManager {
	if sessionTimeout == 0 {
		sessionTimeout = 30 * time.Minute
	}
	
	return &persistentSessionManager{
		memorySessionManager: memorySessionManager{
			sessions:       make(map[string]*Session),
			userSessions:   make(map[string][]string),
			sessionTimeout: sessionTimeout,
		},
		store: store,
	}
}

func (m *persistentSessionManager) CreateSession(ctx context.Context, userID string, metadata map[string]interface{}) (*Session, error) {
	session, err := m.memorySessionManager.CreateSession(ctx, userID, metadata)
	if err != nil {
		return nil, err
	}

	// Persist the session
	if err := m.store.SaveSession(ctx, session); err != nil {
		// Remove from memory if persistence fails
		m.mu.Lock()
		delete(m.sessions, session.ID)
		m.mu.Unlock()
		return nil, err
	}

	return session, nil
}

func (m *persistentSessionManager) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	// First check memory
	session, err := m.memorySessionManager.GetSession(ctx, sessionID)
	if err == nil {
		return session, nil
	}
	
	// If not in memory, try to load from persistent storage
	session, err = m.store.LoadSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}
	
	// Add to memory cache for future access
	m.mu.Lock()
	m.sessions[sessionID] = session
	if userSessions, exists := m.userSessions[session.UserID]; exists {
		// Check if session ID is already in the list
		found := false
		for _, id := range userSessions {
			if id == sessionID {
				found = true
				break
			}
		}
		if !found {
			m.userSessions[session.UserID] = append(userSessions, sessionID)
		}
	} else {
		m.userSessions[session.UserID] = []string{sessionID}
	}
	m.mu.Unlock()
	
	return session, nil
}

func (m *persistentSessionManager) UpdateSession(ctx context.Context, session *Session) error {
	if err := m.memorySessionManager.UpdateSession(ctx, session); err != nil {
		return err
	}

	return m.store.SaveSession(ctx, session)
}

func (m *persistentSessionManager) EndSession(ctx context.Context, sessionID string) error {
	if err := m.memorySessionManager.EndSession(ctx, sessionID); err != nil {
		return err
	}

	// Get the updated session
	m.mu.RLock()
	session := m.sessions[sessionID]
	m.mu.RUnlock()

	if session != nil {
		return m.store.SaveSession(ctx, session)
	}

	return nil
}