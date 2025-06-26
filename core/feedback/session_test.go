package feedback

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemorySessionManager(t *testing.T) {
	ctx := context.Background()
	sessionTimeout := 30 * time.Minute
	manager := NewMemorySessionManager(sessionTimeout)

	t.Run("CreateSession", func(t *testing.T) {
		metadata := map[string]interface{}{
			"device": "mobile",
			"app":    "web",
		}

		session, err := manager.CreateSession(ctx, "user1", metadata)
		assert.NoError(t, err)
		assert.NotEmpty(t, session.ID)
		assert.Equal(t, "user1", session.UserID)
		assert.Equal(t, 0, session.QueryCount)
		assert.Equal(t, metadata, session.Metadata)
		assert.False(t, session.StartTime.IsZero())
		assert.Nil(t, session.EndTime)
	})

	t.Run("GetSession", func(t *testing.T) {
		// Create a session
		session, err := manager.CreateSession(ctx, "user2", nil)
		require.NoError(t, err)

		// Retrieve the session
		retrieved, err := manager.GetSession(ctx, session.ID)
		assert.NoError(t, err)
		assert.Equal(t, session.ID, retrieved.ID)
		assert.Equal(t, session.UserID, retrieved.UserID)
	})

	t.Run("GetSession_NotFound", func(t *testing.T) {
		_, err := manager.GetSession(ctx, "non-existent-session")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("UpdateSession", func(t *testing.T) {
		// Create a session
		session, err := manager.CreateSession(ctx, "user3", nil)
		require.NoError(t, err)

		// Update query count
		session.QueryCount = 5
		err = manager.UpdateSession(ctx, session)
		assert.NoError(t, err)

		// Verify update
		retrieved, err := manager.GetSession(ctx, session.ID)
		assert.NoError(t, err)
		assert.Equal(t, 5, retrieved.QueryCount)
	})

	t.Run("EndSession", func(t *testing.T) {
		// Create a session
		session, err := manager.CreateSession(ctx, "user4", nil)
		require.NoError(t, err)

		// End the session
		err = manager.EndSession(ctx, session.ID)
		assert.NoError(t, err)

		// Verify session was ended
		retrieved, err := manager.GetSession(ctx, session.ID)
		assert.NoError(t, err)
		assert.NotNil(t, retrieved.EndTime)
	})

	t.Run("GetActiveSessions", func(t *testing.T) {
		// Create multiple sessions for a user
		for i := 0; i < 3; i++ {
			_, err := manager.CreateSession(ctx, "user5", nil)
			require.NoError(t, err)
		}

		// Get active sessions
		activeSessions, err := manager.GetActiveSessions(ctx, "user5")
		assert.NoError(t, err)
		assert.Len(t, activeSessions, 3)

		// All should be active (no end time)
		for _, session := range activeSessions {
			assert.Nil(t, session.EndTime)
			assert.Equal(t, "user5", session.UserID)
		}
	})

	t.Run("GetSessionHistory", func(t *testing.T) {
		// Create sessions with some ended
		var sessions []*Session
		for i := 0; i < 5; i++ {
			session, err := manager.CreateSession(ctx, "user6", nil)
			require.NoError(t, err)
			sessions = append(sessions, session)
		}

		// End some sessions
		for i := 0; i < 3; i++ {
			err := manager.EndSession(ctx, sessions[i].ID)
			require.NoError(t, err)
		}

		// Get session history with limit
		history, err := manager.GetSessionHistory(ctx, "user6", 3)
		assert.NoError(t, err)
		assert.LessOrEqual(t, len(history), 3)
	})

	t.Run("SessionTimeout", func(t *testing.T) {
		// Create a manager with short timeout for testing
		shortTimeout := 100 * time.Millisecond
		shortManager := NewMemorySessionManager(shortTimeout)

		// Create a session
		session, err := shortManager.CreateSession(ctx, "user7", nil)
		require.NoError(t, err)

		// Wait for timeout
		time.Sleep(150 * time.Millisecond)

		// Session should be expired
		_, err = shortManager.GetSession(ctx, session.ID)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "expired")
	})
}

func TestPersistentSessionManager(t *testing.T) {
	// Create a mock store
	store := &mockSessionStore{
		sessions: make(map[string]*Session),
	}

	ctx := context.Background()
	sessionTimeout := 30 * time.Minute
	manager := NewPersistentSessionManager(store, sessionTimeout)

	t.Run("CreateSession_Persists", func(t *testing.T) {
		session, err := manager.CreateSession(ctx, "user1", nil)
		assert.NoError(t, err)

		// Verify it was persisted to store
		assert.Contains(t, store.sessions, session.ID)
	})

	t.Run("UpdateSession_Persists", func(t *testing.T) {
		// Create a session
		session, err := manager.CreateSession(ctx, "user2", nil)
		require.NoError(t, err)

		// Update it
		session.QueryCount = 10
		err = manager.UpdateSession(ctx, session)
		assert.NoError(t, err)

		// Verify it was persisted
		stored := store.sessions[session.ID]
		assert.Equal(t, 10, stored.QueryCount)
	})

	t.Run("EndSession_Persists", func(t *testing.T) {
		// Create a session
		session, err := manager.CreateSession(ctx, "user3", nil)
		require.NoError(t, err)

		// End it
		err = manager.EndSession(ctx, session.ID)
		assert.NoError(t, err)

		// Verify end time was persisted
		stored := store.sessions[session.ID]
		assert.NotNil(t, stored.EndTime)
	})
}

// Mock implementation of SessionStore for testing
type mockSessionStore struct {
	sessions map[string]*Session
}

func (s *mockSessionStore) SaveSession(ctx context.Context, session *Session) error {
	s.sessions[session.ID] = session
	return nil
}

func (s *mockSessionStore) LoadSession(ctx context.Context, sessionID string) (*Session, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, nil
	}
	return session, nil
}

func (s *mockSessionStore) LoadUserSessions(ctx context.Context, userID string, limit int) ([]*Session, error) {
	var results []*Session
	for _, session := range s.sessions {
		if session.UserID == userID {
			results = append(results, session)
			if limit > 0 && len(results) >= limit {
				break
			}
		}
	}
	return results, nil
}

func (s *mockSessionStore) DeleteSession(ctx context.Context, sessionID string) error {
	delete(s.sessions, sessionID)
	return nil
}
