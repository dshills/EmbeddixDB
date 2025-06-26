package feedback

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	bolt "go.etcd.io/bbolt"
)

// Bucket names for BoltDB
var (
	bucketInteractions     = []byte("interactions")
	bucketQueryFeedback    = []byte("query_feedback")
	bucketDocumentFeedback = []byte("document_feedback")
	bucketSessions         = []byte("sessions")
	bucketProfiles         = []byte("profiles")

	// Index buckets
	bucketUserInteractions     = []byte("idx_user_interactions")
	bucketSessionInteractions  = []byte("idx_session_interactions")
	bucketQueryInteractions    = []byte("idx_query_interactions")
	bucketDocumentInteractions = []byte("idx_document_interactions")
)

// BoltFeedbackStore implements FeedbackStore using BoltDB
type BoltFeedbackStore struct {
	db *bolt.DB
}

// NewBoltFeedbackStore creates a new BoltDB-backed feedback store
func NewBoltFeedbackStore(dbPath string) (*BoltFeedbackStore, error) {
	db, err := bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Create buckets
	err = db.Update(func(tx *bolt.Tx) error {
		buckets := [][]byte{
			bucketInteractions,
			bucketQueryFeedback,
			bucketDocumentFeedback,
			bucketSessions,
			bucketProfiles,
			bucketUserInteractions,
			bucketSessionInteractions,
			bucketQueryInteractions,
			bucketDocumentInteractions,
		}

		for _, bucket := range buckets {
			if _, err := tx.CreateBucketIfNotExists(bucket); err != nil {
				return err
			}
		}

		return nil
	})

	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create buckets: %w", err)
	}

	return &BoltFeedbackStore{db: db}, nil
}

// Close closes the database
func (s *BoltFeedbackStore) Close() error {
	return s.db.Close()
}

// SaveInteraction saves an interaction to the store
func (s *BoltFeedbackStore) SaveInteraction(ctx context.Context, interaction *Interaction) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		// Marshal interaction
		data, err := json.Marshal(interaction)
		if err != nil {
			return err
		}

		// Save to main bucket
		b := tx.Bucket(bucketInteractions)
		if err := b.Put([]byte(interaction.ID), data); err != nil {
			return err
		}

		// Update indexes
		if err := s.addToIndex(tx, bucketUserInteractions, interaction.UserID, interaction.ID); err != nil {
			return err
		}
		if err := s.addToIndex(tx, bucketSessionInteractions, interaction.SessionID, interaction.ID); err != nil {
			return err
		}
		if err := s.addToIndex(tx, bucketQueryInteractions, interaction.QueryID, interaction.ID); err != nil {
			return err
		}
		if err := s.addToIndex(tx, bucketDocumentInteractions, interaction.DocumentID, interaction.ID); err != nil {
			return err
		}

		return nil
	})
}

// LoadInteractions loads interactions based on filter
func (s *BoltFeedbackStore) LoadInteractions(ctx context.Context, filter InteractionFilter) ([]*Interaction, error) {
	var interactions []*Interaction

	err := s.db.View(func(tx *bolt.Tx) error {
		// Determine which index to use
		var indexBucket []byte
		var indexKey string

		if filter.UserID != "" {
			indexBucket = bucketUserInteractions
			indexKey = filter.UserID
		} else if filter.SessionID != "" {
			indexBucket = bucketSessionInteractions
			indexKey = filter.SessionID
		} else if filter.QueryID != "" {
			indexBucket = bucketQueryInteractions
			indexKey = filter.QueryID
		} else if filter.DocumentID != "" {
			indexBucket = bucketDocumentInteractions
			indexKey = filter.DocumentID
		}

		// Get interaction IDs from index
		var interactionIDs []string
		if indexBucket != nil {
			ids, err := s.getFromIndex(tx, indexBucket, indexKey)
			if err != nil {
				return err
			}
			interactionIDs = ids
		} else {
			// No specific filter, get all interactions
			b := tx.Bucket(bucketInteractions)
			c := b.Cursor()
			for k, _ := c.First(); k != nil; k, _ = c.Next() {
				interactionIDs = append(interactionIDs, string(k))
			}
		}

		// Load interactions
		b := tx.Bucket(bucketInteractions)
		for _, id := range interactionIDs {
			data := b.Get([]byte(id))
			if data == nil {
				continue
			}

			var interaction Interaction
			if err := json.Unmarshal(data, &interaction); err != nil {
				continue
			}

			// Apply filters
			if !s.matchesFilter(&interaction, filter) {
				continue
			}

			interactions = append(interactions, &interaction)

			// Apply limit
			if filter.Limit > 0 && len(interactions) >= filter.Limit {
				break
			}
		}

		return nil
	})

	return interactions, err
}

// SaveQueryFeedback saves query feedback
func (s *BoltFeedbackStore) SaveQueryFeedback(ctx context.Context, feedback *QueryFeedback) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		data, err := json.Marshal(feedback)
		if err != nil {
			return err
		}

		b := tx.Bucket(bucketQueryFeedback)
		return b.Put([]byte(feedback.QueryID), data)
	})
}

// LoadQueryFeedback loads query feedback
func (s *BoltFeedbackStore) LoadQueryFeedback(ctx context.Context, queryID string) (*QueryFeedback, error) {
	var feedback *QueryFeedback

	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketQueryFeedback)
		data := b.Get([]byte(queryID))
		if data == nil {
			return fmt.Errorf("query feedback not found: %s", queryID)
		}

		var f QueryFeedback
		if err := json.Unmarshal(data, &f); err != nil {
			return err
		}

		feedback = &f
		return nil
	})

	return feedback, err
}

// SaveDocumentFeedback saves document feedback
func (s *BoltFeedbackStore) SaveDocumentFeedback(ctx context.Context, feedback *DocumentFeedback) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		data, err := json.Marshal(feedback)
		if err != nil {
			return err
		}

		b := tx.Bucket(bucketDocumentFeedback)
		return b.Put([]byte(feedback.DocumentID), data)
	})
}

// LoadDocumentFeedback loads document feedback
func (s *BoltFeedbackStore) LoadDocumentFeedback(ctx context.Context, documentID string) (*DocumentFeedback, error) {
	var feedback *DocumentFeedback

	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketDocumentFeedback)
		data := b.Get([]byte(documentID))
		if data == nil {
			return fmt.Errorf("document feedback not found: %s", documentID)
		}

		var f DocumentFeedback
		if err := json.Unmarshal(data, &f); err != nil {
			return err
		}

		feedback = &f
		return nil
	})

	return feedback, err
}

// BoltSessionStore implements SessionStore using BoltDB
type BoltSessionStore struct {
	db *bolt.DB
}

// NewBoltSessionStore creates a new BoltDB-backed session store
func NewBoltSessionStore(db *bolt.DB) *BoltSessionStore {
	return &BoltSessionStore{db: db}
}

// SaveSession saves a session
func (s *BoltSessionStore) SaveSession(ctx context.Context, session *Session) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		data, err := json.Marshal(session)
		if err != nil {
			return err
		}

		b := tx.Bucket(bucketSessions)
		return b.Put([]byte(session.ID), data)
	})
}

// LoadSession loads a session
func (s *BoltSessionStore) LoadSession(ctx context.Context, sessionID string) (*Session, error) {
	var session *Session

	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketSessions)
		data := b.Get([]byte(sessionID))
		if data == nil {
			return fmt.Errorf("session not found: %s", sessionID)
		}

		var sess Session
		if err := json.Unmarshal(data, &sess); err != nil {
			return err
		}

		session = &sess
		return nil
	})

	return session, err
}

// LoadUserSessions loads sessions for a user
func (s *BoltSessionStore) LoadUserSessions(ctx context.Context, userID string, limit int) ([]*Session, error) {
	var sessions []*Session

	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketSessions)
		c := b.Cursor()

		count := 0
		for k, v := c.Last(); k != nil && (limit <= 0 || count < limit); k, v = c.Prev() {
			var session Session
			if err := json.Unmarshal(v, &session); err != nil {
				continue
			}

			if session.UserID == userID {
				sessions = append(sessions, &session)
				count++
			}
		}

		return nil
	})

	return sessions, err
}

// DeleteSession deletes a session
func (s *BoltSessionStore) DeleteSession(ctx context.Context, sessionID string) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketSessions)
		return b.Delete([]byte(sessionID))
	})
}

// BoltProfileStore implements ProfileStore using BoltDB
type BoltProfileStore struct {
	db *bolt.DB
}

// NewBoltProfileStore creates a new BoltDB-backed profile store
func NewBoltProfileStore(db *bolt.DB) *BoltProfileStore {
	return &BoltProfileStore{db: db}
}

// SaveProfile saves a user profile
func (s *BoltProfileStore) SaveProfile(ctx context.Context, profile *UserProfile) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		data, err := json.Marshal(profile)
		if err != nil {
			return err
		}

		b := tx.Bucket(bucketProfiles)
		return b.Put([]byte(profile.ID), data)
	})
}

// LoadProfile loads a user profile
func (s *BoltProfileStore) LoadProfile(ctx context.Context, userID string) (*UserProfile, error) {
	var profile *UserProfile

	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketProfiles)
		data := b.Get([]byte(userID))
		if data == nil {
			return fmt.Errorf("profile not found: %s", userID)
		}

		var prof UserProfile
		if err := json.Unmarshal(data, &prof); err != nil {
			return err
		}

		profile = &prof
		return nil
	})

	return profile, err
}

// DeleteProfile deletes a user profile
func (s *BoltProfileStore) DeleteProfile(ctx context.Context, userID string) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketProfiles)
		return b.Delete([]byte(userID))
	})
}

// Helper methods

func (s *BoltFeedbackStore) addToIndex(tx *bolt.Tx, bucketName []byte, key, value string) error {
	if key == "" {
		return nil
	}

	b := tx.Bucket(bucketName)

	// Get existing values
	data := b.Get([]byte(key))
	var values []string
	if data != nil {
		if err := json.Unmarshal(data, &values); err != nil {
			return err
		}
	}

	// Add new value
	values = append(values, value)

	// Save updated values
	data, err := json.Marshal(values)
	if err != nil {
		return err
	}

	return b.Put([]byte(key), data)
}

func (s *BoltFeedbackStore) getFromIndex(tx *bolt.Tx, bucketName []byte, key string) ([]string, error) {
	b := tx.Bucket(bucketName)
	data := b.Get([]byte(key))
	if data == nil {
		return nil, nil
	}

	var values []string
	if err := json.Unmarshal(data, &values); err != nil {
		return nil, err
	}

	return values, nil
}

func (s *BoltFeedbackStore) matchesFilter(interaction *Interaction, filter InteractionFilter) bool {
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
	if filter.CollectionID != "" && interaction.CollectionID != filter.CollectionID {
		return false
	}
	if filter.Type != "" && interaction.Type != filter.Type {
		return false
	}
	if filter.StartTime != nil && interaction.Timestamp.Before(*filter.StartTime) {
		return false
	}
	if filter.EndTime != nil && interaction.Timestamp.After(*filter.EndTime) {
		return false
	}

	return true
}
