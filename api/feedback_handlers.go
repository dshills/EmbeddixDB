package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/dshills/EmbeddixDB/core/feedback"
	"github.com/dshills/EmbeddixDB/core/search"
	"github.com/gorilla/mux"
)

// Feedback API request/response types

// RecordInteractionRequest represents a user interaction to record
type RecordInteractionRequest struct {
	UserID       string                 `json:"user_id"`
	SessionID    string                 `json:"session_id"`
	QueryID      string                 `json:"query_id"`
	Query        string                 `json:"query"`
	DocumentID   string                 `json:"document_id"`
	CollectionID string                 `json:"collection_id"`
	Type         string                 `json:"type"`  // click, dwell, rating, etc.
	Value        float64                `json:"value"` // Rating value, dwell time, etc.
	Position     int                    `json:"position"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// CreateSessionRequest represents a request to create a new session
type CreateSessionRequest struct {
	UserID   string                 `json:"user_id"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// UpdatePreferencesRequest represents user preference updates
type UpdatePreferencesRequest struct {
	UserID             string             `json:"user_id"`
	PreferredLanguages []string           `json:"preferred_languages,omitempty"`
	SearchRecency      *float64           `json:"search_recency,omitempty"`
	SearchDiversity    *float64           `json:"search_diversity,omitempty"`
	TopicWeights       map[string]float64 `json:"topic_weights,omitempty"`
	SourceWeights      map[string]float64 `json:"source_weights,omitempty"`
	VectorWeight       *float64           `json:"vector_weight,omitempty"`
	TextWeight         *float64           `json:"text_weight,omitempty"`
}

// PersonalizedSearchRequest represents a personalized search request
type PersonalizedSearchRequest struct {
	CollectionID          string                 `json:"collection_id"`
	Query                 string                 `json:"query"`
	Vector                []float32              `json:"vector,omitempty"`
	K                     int                    `json:"k"`
	Filter                map[string]interface{} `json:"filter,omitempty"`
	UserID                string                 `json:"user_id,omitempty"`
	SessionID             string                 `json:"session_id,omitempty"`
	SearchMode            string                 `json:"search_mode,omitempty"`
	VectorWeight          *float64               `json:"vector_weight,omitempty"`
	TextWeight            *float64               `json:"text_weight,omitempty"`
	EnableReranking       bool                   `json:"enable_reranking"`
	EnablePersonalization bool                   `json:"enable_personalization"`
	CollectFeedback       bool                   `json:"collect_feedback"`
	BoostFields           map[string]float64     `json:"boost_fields,omitempty"`
}

// Feedback API handlers

// handleRecordInteraction records a user interaction
func (s *Server) handleRecordInteraction(w http.ResponseWriter, r *http.Request) {
	var req RecordInteractionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Convert to feedback interaction
	interaction := &feedback.Interaction{
		UserID:       req.UserID,
		SessionID:    req.SessionID,
		QueryID:      req.QueryID,
		Query:        req.Query,
		DocumentID:   req.DocumentID,
		CollectionID: req.CollectionID,
		Type:         feedback.InteractionType(req.Type),
		Value:        req.Value,
		Position:     req.Position,
		Timestamp:    time.Now(),
		Metadata:     req.Metadata,
	}

	// Record the interaction
	if s.personalizedSearchManager != nil {
		ctx := context.Background()
		if err := s.personalizedSearchManager.RecordInteraction(ctx, interaction); err != nil {
			s.respondWithError(w, http.StatusInternalServerError, err.Error())
			return
		}
	} else {
		s.respondWithError(w, http.StatusServiceUnavailable, "Personalized search not available")
		return
	}

	s.respondWithJSON(w, http.StatusOK, map[string]string{
		"status":  "success",
		"message": "Interaction recorded",
	})
}

// handleCreateSession creates a new search session
func (s *Server) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	var req CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if s.feedbackManager == nil || s.feedbackManager.SessionManager == nil {
		s.respondWithError(w, http.StatusServiceUnavailable, "Session management not available")
		return
	}

	ctx := context.Background()
	session, err := s.feedbackManager.SessionManager.CreateSession(ctx, req.UserID, req.Metadata)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, session)
}

// handleGetByID is a generic handler for retrieving resources by ID
func (s *Server) handleGetByID(w http.ResponseWriter, r *http.Request,
	paramName string,
	serviceName string,
	serviceAvailable func() bool,
	getFunc func(context.Context, string) (interface{}, error)) {
	vars := mux.Vars(r)
	id := vars[paramName]

	if !serviceAvailable() {
		s.respondWithError(w, http.StatusServiceUnavailable, serviceName+" not available")
		return
	}

	ctx := context.Background()
	result, err := getFunc(ctx, id)
	if err != nil {
		s.respondWithError(w, http.StatusNotFound, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, result)
}

// handleGetSession retrieves session information
func (s *Server) handleGetSession(w http.ResponseWriter, r *http.Request) {
	s.handleGetByID(w, r, "session_id", "Session management",
		func() bool { return s.feedbackManager != nil && s.feedbackManager.SessionManager != nil },
		func(ctx context.Context, id string) (interface{}, error) {
			return s.feedbackManager.SessionManager.GetSession(ctx, id)
		})
}

// handleGetUserProfile retrieves user profile
func (s *Server) handleGetUserProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["user_id"]

	if s.feedbackManager == nil || s.feedbackManager.ProfileManager == nil {
		s.respondWithError(w, http.StatusServiceUnavailable, "Profile management not available")
		return
	}

	ctx := context.Background()
	profile, err := s.feedbackManager.ProfileManager.GetProfile(ctx, userID)
	if err != nil {
		// Try to create profile if it doesn't exist
		profile, err = s.feedbackManager.ProfileManager.CreateProfile(ctx, userID)
		if err != nil {
			s.respondWithError(w, http.StatusNotFound, err.Error())
			return
		}
	}

	s.respondWithJSON(w, http.StatusOK, profile)
}

// handleUpdatePreferences updates user preferences
func (s *Server) handleUpdatePreferences(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["user_id"]

	var req UpdatePreferencesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if s.feedbackManager == nil || s.feedbackManager.ProfileManager == nil {
		s.respondWithError(w, http.StatusServiceUnavailable, "Profile management not available")
		return
	}

	ctx := context.Background()

	// Get current profile
	profile, err := s.feedbackManager.ProfileManager.GetProfile(ctx, userID)
	if err != nil {
		// Create profile if it doesn't exist
		profile, err = s.feedbackManager.ProfileManager.CreateProfile(ctx, userID)
		if err != nil {
			s.respondWithError(w, http.StatusInternalServerError, err.Error())
			return
		}
	}

	// Update preferences
	if len(req.PreferredLanguages) > 0 {
		profile.Preferences.PreferredLanguages = req.PreferredLanguages
	}
	if req.SearchRecency != nil {
		profile.Preferences.SearchRecency = *req.SearchRecency
	}
	if req.SearchDiversity != nil {
		profile.Preferences.SearchDiversity = *req.SearchDiversity
	}
	if req.TopicWeights != nil {
		profile.Preferences.TopicWeights = req.TopicWeights
	}
	if req.SourceWeights != nil {
		profile.Preferences.SourceWeights = req.SourceWeights
	}
	if req.VectorWeight != nil {
		profile.Preferences.VectorWeight = *req.VectorWeight
	}
	if req.TextWeight != nil {
		profile.Preferences.TextWeight = *req.TextWeight
	}

	// Save updated preferences
	if err := s.feedbackManager.ProfileManager.UpdatePreferences(ctx, userID, profile.Preferences); err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, profile.Preferences)
}

// handlePersonalizedSearch performs personalized search
func (s *Server) handlePersonalizedSearch(w http.ResponseWriter, r *http.Request) {
	var req PersonalizedSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondWithError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if s.personalizedSearchManager == nil {
		s.respondWithError(w, http.StatusServiceUnavailable, "Personalized search not available")
		return
	}

	// Convert to internal request type
	searchReq := search.PersonalizedSearchRequest{
		CollectionID:          req.CollectionID,
		Query:                 req.Query,
		Vector:                req.Vector,
		K:                     req.K,
		Filter:                req.Filter,
		UserID:                req.UserID,
		SessionID:             req.SessionID,
		SearchMode:            req.SearchMode,
		VectorWeight:          req.VectorWeight,
		TextWeight:            req.TextWeight,
		EnableReranking:       req.EnableReranking,
		EnablePersonalization: req.EnablePersonalization,
		CollectFeedback:       req.CollectFeedback,
		BoostFields:           req.BoostFields,
	}

	// Perform search
	ctx := context.Background()
	response, err := s.personalizedSearchManager.Search(ctx, searchReq)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, response)
}

// handleGetCTRReport retrieves CTR analytics report
func (s *Server) handleGetCTRReport(w http.ResponseWriter, r *http.Request) {
	if s.personalizedSearchManager == nil {
		s.respondWithError(w, http.StatusServiceUnavailable, "CTR tracking not available")
		return
	}

	ctx := context.Background()
	report, err := s.personalizedSearchManager.GetCTRReport(ctx)
	if err != nil {
		s.respondWithError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, report)
}

// handleGetDocumentCTR retrieves CTR metrics for a specific document
func (s *Server) handleGetDocumentCTR(w http.ResponseWriter, r *http.Request) {
	s.handleGetByID(w, r, "document_id", "CTR tracking",
		func() bool { return s.feedbackManager != nil && s.feedbackManager.CTRTracker != nil },
		func(ctx context.Context, id string) (interface{}, error) {
			return s.feedbackManager.CTRTracker.GetDocumentCTR(ctx, id)
		})
}

// handleGetQueryCTR retrieves CTR metrics for a specific query
func (s *Server) handleGetQueryCTR(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		s.respondWithError(w, http.StatusBadRequest, "Query parameter 'q' is required")
		return
	}

	if s.feedbackManager == nil || s.feedbackManager.CTRTracker == nil {
		s.respondWithError(w, http.StatusServiceUnavailable, "CTR tracking not available")
		return
	}

	ctx := context.Background()
	metrics, err := s.feedbackManager.CTRTracker.GetQueryCTR(ctx, query)
	if err != nil {
		s.respondWithError(w, http.StatusNotFound, err.Error())
		return
	}

	s.respondWithJSON(w, http.StatusOK, metrics)
}

// RegisterFeedbackRoutes registers all feedback-related routes
func (s *Server) RegisterFeedbackRoutes(router *mux.Router) {
	// Interaction recording
	router.HandleFunc("/api/v1/feedback/interaction", s.handleRecordInteraction).Methods("POST")

	// Session management
	router.HandleFunc("/api/v1/sessions", s.handleCreateSession).Methods("POST")
	router.HandleFunc("/api/v1/sessions/{session_id}", s.handleGetSession).Methods("GET")

	// User profiles
	router.HandleFunc("/api/v1/users/{user_id}/profile", s.handleGetUserProfile).Methods("GET")
	router.HandleFunc("/api/v1/users/{user_id}/preferences", s.handleUpdatePreferences).Methods("PUT")

	// Personalized search
	router.HandleFunc("/api/v1/search/personalized", s.handlePersonalizedSearch).Methods("POST")

	// CTR analytics
	router.HandleFunc("/api/v1/analytics/ctr/report", s.handleGetCTRReport).Methods("GET")
	router.HandleFunc("/api/v1/analytics/ctr/document/{document_id}", s.handleGetDocumentCTR).Methods("GET")
	router.HandleFunc("/api/v1/analytics/ctr/query", s.handleGetQueryCTR).Methods("GET")
}
