package feedback

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// memoryCollector implements Collector interface with in-memory storage
type memoryCollector struct {
	mu               sync.RWMutex
	interactions     map[string]*Interaction
	queryFeedback    map[string]*QueryFeedback
	documentFeedback map[string]*DocumentFeedback

	// Indexes for efficient lookups
	userInteractions     map[string][]string // userID -> interactionIDs
	sessionInteractions  map[string][]string // sessionID -> interactionIDs
	queryInteractions    map[string][]string // queryID -> interactionIDs
	documentInteractions map[string][]string // documentID -> interactionIDs
}

// NewMemoryCollector creates a new in-memory feedback collector
func NewMemoryCollector() Collector {
	return &memoryCollector{
		interactions:         make(map[string]*Interaction),
		queryFeedback:        make(map[string]*QueryFeedback),
		documentFeedback:     make(map[string]*DocumentFeedback),
		userInteractions:     make(map[string][]string),
		sessionInteractions:  make(map[string][]string),
		queryInteractions:    make(map[string][]string),
		documentInteractions: make(map[string][]string),
	}
}

func (c *memoryCollector) RecordInteraction(ctx context.Context, interaction *Interaction) error {
	if interaction.ID == "" {
		interaction.ID = uuid.New().String()
	}
	if interaction.Timestamp.IsZero() {
		interaction.Timestamp = time.Now()
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Store interaction
	c.interactions[interaction.ID] = interaction

	// Update indexes
	c.userInteractions[interaction.UserID] = append(c.userInteractions[interaction.UserID], interaction.ID)
	c.sessionInteractions[interaction.SessionID] = append(c.sessionInteractions[interaction.SessionID], interaction.ID)
	c.queryInteractions[interaction.QueryID] = append(c.queryInteractions[interaction.QueryID], interaction.ID)
	c.documentInteractions[interaction.DocumentID] = append(c.documentInteractions[interaction.DocumentID], interaction.ID)

	// Update document feedback synchronously to avoid race conditions in tests
	c.updateDocumentFeedbackSync(interaction)

	return nil
}

func (c *memoryCollector) RecordQuery(ctx context.Context, feedback *QueryFeedback) error {
	if feedback.QueryID == "" {
		feedback.QueryID = uuid.New().String()
	}
	if feedback.Timestamp.IsZero() {
		feedback.Timestamp = time.Now()
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.queryFeedback[feedback.QueryID] = feedback
	return nil
}

func (c *memoryCollector) GetInteractions(ctx context.Context, filter InteractionFilter) ([]*Interaction, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var results []*Interaction
	var candidateIDs []string

	// Determine candidate interactions based on filter
	if filter.UserID != "" {
		candidateIDs = c.userInteractions[filter.UserID]
	} else if filter.SessionID != "" {
		candidateIDs = c.sessionInteractions[filter.SessionID]
	} else if filter.QueryID != "" {
		candidateIDs = c.queryInteractions[filter.QueryID]
	} else if filter.DocumentID != "" {
		candidateIDs = c.documentInteractions[filter.DocumentID]
	} else {
		// No specific filter, check all interactions
		for id := range c.interactions {
			candidateIDs = append(candidateIDs, id)
		}
	}

	// Apply filters
	for _, id := range candidateIDs {
		interaction := c.interactions[id]
		if interaction == nil {
			continue
		}

		// Apply all filters
		if filter.UserID != "" && interaction.UserID != filter.UserID {
			continue
		}
		if filter.SessionID != "" && interaction.SessionID != filter.SessionID {
			continue
		}
		if filter.QueryID != "" && interaction.QueryID != filter.QueryID {
			continue
		}
		if filter.DocumentID != "" && interaction.DocumentID != filter.DocumentID {
			continue
		}
		if filter.CollectionID != "" && interaction.CollectionID != filter.CollectionID {
			continue
		}
		if filter.Type != "" && interaction.Type != filter.Type {
			continue
		}
		if filter.StartTime != nil && interaction.Timestamp.Before(*filter.StartTime) {
			continue
		}
		if filter.EndTime != nil && interaction.Timestamp.After(*filter.EndTime) {
			continue
		}

		results = append(results, interaction)
	}

	// Apply limit
	if filter.Limit > 0 && len(results) > filter.Limit {
		results = results[:filter.Limit]
	}

	return results, nil
}

func (c *memoryCollector) GetQueryFeedback(ctx context.Context, queryID string) (*QueryFeedback, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	feedback, exists := c.queryFeedback[queryID]
	if !exists {
		return nil, fmt.Errorf("query feedback not found: %s", queryID)
	}

	return feedback, nil
}

func (c *memoryCollector) GetDocumentFeedback(ctx context.Context, documentID string) (*DocumentFeedback, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	feedback, exists := c.documentFeedback[documentID]
	if !exists {
		return nil, fmt.Errorf("document feedback not found: %s", documentID)
	}

	return feedback, nil
}

func (c *memoryCollector) UpdateDocumentFeedback(ctx context.Context, feedback *DocumentFeedback) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	feedback.LastUpdated = time.Now()
	c.documentFeedback[feedback.DocumentID] = feedback
	return nil
}

func (c *memoryCollector) updateDocumentFeedbackSync(interaction *Interaction) {
	// Note: This is called from RecordInteraction which already holds the lock
	feedback, exists := c.documentFeedback[interaction.DocumentID]
	if !exists {
		feedback = &DocumentFeedback{
			DocumentID:   interaction.DocumentID,
			CollectionID: interaction.CollectionID,
			LastUpdated:  time.Now(),
		}
	}

	// Update feedback based on interaction type
	switch interaction.Type {
	case InteractionTypeClick:
		feedback.TotalClicks++
	case InteractionTypeDwell:
		feedback.AvgDwellTime = (feedback.AvgDwellTime*float64(feedback.TotalViews) + interaction.Value) / float64(feedback.TotalViews+1)
	case InteractionTypeRating:
		feedback.AvgRating = (feedback.AvgRating*float64(feedback.TotalRatings) + interaction.Value) / float64(feedback.TotalRatings+1)
		feedback.TotalRatings++
	}

	feedback.TotalViews++
	if feedback.TotalViews > 0 {
		feedback.ClickThroughRate = float64(feedback.TotalClicks) / float64(feedback.TotalViews)
	}

	feedback.LastUpdated = time.Now()
	c.documentFeedback[interaction.DocumentID] = feedback
}

// persistentCollector implements Collector with persistent storage
type persistentCollector struct {
	memoryCollector
	store FeedbackStore
}

// FeedbackStore defines the interface for persistent feedback storage
type FeedbackStore interface {
	SaveInteraction(ctx context.Context, interaction *Interaction) error
	LoadInteractions(ctx context.Context, filter InteractionFilter) ([]*Interaction, error)
	SaveQueryFeedback(ctx context.Context, feedback *QueryFeedback) error
	LoadQueryFeedback(ctx context.Context, queryID string) (*QueryFeedback, error)
	SaveDocumentFeedback(ctx context.Context, feedback *DocumentFeedback) error
	LoadDocumentFeedback(ctx context.Context, documentID string) (*DocumentFeedback, error)
}

// NewPersistentCollector creates a feedback collector with persistent storage
func NewPersistentCollector(store FeedbackStore) Collector {
	return &persistentCollector{
		memoryCollector: memoryCollector{
			interactions:         make(map[string]*Interaction),
			queryFeedback:        make(map[string]*QueryFeedback),
			documentFeedback:     make(map[string]*DocumentFeedback),
			userInteractions:     make(map[string][]string),
			sessionInteractions:  make(map[string][]string),
			queryInteractions:    make(map[string][]string),
			documentInteractions: make(map[string][]string),
		},
		store: store,
	}
}

func (c *persistentCollector) GetInteractions(ctx context.Context, filter InteractionFilter) ([]*Interaction, error) {
	// First try to get from memory
	memoryResults, err := c.memoryCollector.GetInteractions(ctx, filter)
	if err != nil {
		// If memory fails, try to load from persistent storage
		return c.store.LoadInteractions(ctx, filter)
	}

	// If memory returns results, use those
	if len(memoryResults) > 0 {
		return memoryResults, nil
	}

	// If memory is empty, try to load from persistent storage and cache in memory
	persistentResults, err := c.store.LoadInteractions(ctx, filter)
	if err != nil {
		return []*Interaction{}, nil // Return empty slice if no data found
	}

	// Cache the results in memory for future access
	for _, interaction := range persistentResults {
		// Add to memory (ignore errors since we already have the data)
		c.memoryCollector.RecordInteraction(ctx, interaction)
	}

	return persistentResults, nil
}

func (c *persistentCollector) RecordInteraction(ctx context.Context, interaction *Interaction) error {
	// Save to memory first
	if err := c.memoryCollector.RecordInteraction(ctx, interaction); err != nil {
		return err
	}

	// Then persist
	return c.store.SaveInteraction(ctx, interaction)
}

func (c *persistentCollector) RecordQuery(ctx context.Context, feedback *QueryFeedback) error {
	// Save to memory first
	if err := c.memoryCollector.RecordQuery(ctx, feedback); err != nil {
		return err
	}

	// Then persist
	return c.store.SaveQueryFeedback(ctx, feedback)
}

func (c *persistentCollector) UpdateDocumentFeedback(ctx context.Context, feedback *DocumentFeedback) error {
	// Update in memory first
	if err := c.memoryCollector.UpdateDocumentFeedback(ctx, feedback); err != nil {
		return err
	}

	// Then persist
	return c.store.SaveDocumentFeedback(ctx, feedback)
}
