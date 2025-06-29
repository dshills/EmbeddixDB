package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/search"
	"github.com/gorilla/mux"
)

// Server represents the REST API server
type Server struct {
	vectorStore               core.VectorStore
	router                    *mux.Router
	httpServer                *http.Server
	config                    ServerConfig
	personalizedSearchManager *search.PersonalizedSearchManager
	feedbackManager           *search.FeedbackManager
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Host            string        `json:"host"`
	Port            int           `json:"port"`
	ReadTimeout     time.Duration `json:"read_timeout"`
	WriteTimeout    time.Duration `json:"write_timeout"`
	IdleTimeout     time.Duration `json:"idle_timeout"`
	ShutdownTimeout time.Duration `json:"shutdown_timeout"`
}

// DefaultServerConfig returns default server configuration
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Host:            "0.0.0.0",
		Port:            8080,
		ReadTimeout:     15 * time.Second,
		WriteTimeout:    15 * time.Second,
		IdleTimeout:     60 * time.Second,
		ShutdownTimeout: 10 * time.Second,
	}
}

// NewServer creates a new API server
func NewServer(vectorStore core.VectorStore, config ServerConfig) *Server {
	s := &Server{
		vectorStore: vectorStore,
		config:      config,
	}

	s.setupRoutes()
	return s
}

// NewServerWithFeedback creates a new API server with feedback capabilities
func NewServerWithFeedback(
	vectorStore core.VectorStore,
	config ServerConfig,
	personalizedSearchManager *search.PersonalizedSearchManager,
	feedbackManager *search.FeedbackManager,
) *Server {
	s := &Server{
		vectorStore:               vectorStore,
		config:                    config,
		personalizedSearchManager: personalizedSearchManager,
		feedbackManager:           feedbackManager,
	}

	s.setupRoutes()
	return s
}

// setupRoutes configures all API routes
func (s *Server) setupRoutes() {
	s.router = mux.NewRouter()

	// Middleware
	s.router.Use(loggingMiddleware)
	s.router.Use(jsonContentTypeMiddleware)

	// Health check
	s.router.HandleFunc("/health", s.handleHealth).Methods("GET")

	// Collection endpoints
	s.router.HandleFunc("/collections", s.handleListCollections).Methods("GET")
	s.router.HandleFunc("/collections", s.handleCreateCollection).Methods("POST")
	s.router.HandleFunc("/collections/{collection}", s.handleGetCollection).Methods("GET")
	s.router.HandleFunc("/collections/{collection}", s.handleDeleteCollection).Methods("DELETE")

	// Vector endpoints
	s.router.HandleFunc("/collections/{collection}/vectors", s.handleAddVector).Methods("POST")
	s.router.HandleFunc("/collections/{collection}/vectors/batch", s.handleAddVectorsBatch).Methods("POST")
	s.router.HandleFunc("/collections/{collection}/vectors/{id}", s.handleGetVector).Methods("GET")
	s.router.HandleFunc("/collections/{collection}/vectors/{id}", s.handleUpdateVector).Methods("PUT")
	s.router.HandleFunc("/collections/{collection}/vectors/{id}", s.handleDeleteVector).Methods("DELETE")

	// Search endpoints
	s.router.HandleFunc("/collections/{collection}/search", s.handleSearch).Methods("POST")
	s.router.HandleFunc("/collections/{collection}/search/batch", s.handleBatchSearch).Methods("POST")
	s.router.HandleFunc("/collections/{collection}/search/range", s.handleRangeSearch).Methods("POST")

	// Stats endpoints
	s.router.HandleFunc("/stats", s.handleStats).Methods("GET")
	s.router.HandleFunc("/collections/{collection}/stats", s.handleCollectionStats).Methods("GET")

	// Feedback and personalization endpoints (if available)
	if s.personalizedSearchManager != nil || s.feedbackManager != nil {
		s.RegisterFeedbackRoutes(s.router)
	}

	// Documentation endpoints
	s.setupSwaggerUI()
	s.setupReDoc()
}

// Start starts the HTTP server
func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)

	s.httpServer = &http.Server{
		Addr:         addr,
		Handler:      s.router,
		ReadTimeout:  s.config.ReadTimeout,
		WriteTimeout: s.config.WriteTimeout,
		IdleTimeout:  s.config.IdleTimeout,
	}

	fmt.Printf("Starting EmbeddixDB API server on %s\n", addr)
	fmt.Printf("API Documentation available at:\n")
	fmt.Printf("  - Swagger UI: http://%s/docs\n", addr)
	fmt.Printf("  - ReDoc: http://%s/redoc\n", addr)
	fmt.Printf("  - OpenAPI Spec: http://%s/swagger.yaml\n", addr)
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

// ServeHTTP implements the http.Handler interface
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

// Middleware functions
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		fmt.Printf("[%s] %s %s %v\n", time.Now().Format("2006-01-02 15:04:05"), r.Method, r.URL.Path, time.Since(start))
	})
}

func jsonContentTypeMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		next.ServeHTTP(w, r)
	})
}

// Error response helper
func (s *Server) respondWithError(w http.ResponseWriter, code int, message string) {
	s.respondWithJSON(w, code, map[string]string{"error": message})
}

// JSON response helper
func (s *Server) respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	response, err := json.Marshal(payload)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "Error marshaling JSON"}`))
		return
	}

	w.WriteHeader(code)
	w.Write(response)
}
