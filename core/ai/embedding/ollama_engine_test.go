package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockOllamaServer creates a mock Ollama API server for testing
func mockOllamaServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/embeddings", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var req OllamaEmbedRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		// Generate mock embeddings based on the model
		var embedding []float32
		switch req.Model {
		case "nomic-embed-text":
			embedding = make([]float32, 768)
		case "all-minilm":
			embedding = make([]float32, 384)
		default:
			embedding = make([]float32, 512)
		}

		// Fill with mock values
		for i := range embedding {
			embedding[i] = float32(i) / float32(len(embedding))
		}

		resp := OllamaEmbedResponse{
			Embedding: embedding,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestNewOllamaEmbeddingEngine(t *testing.T) {
	tests := []struct {
		name    string
		config  *ai.ModelConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: &ai.ModelConfig{
				Path:           "nomic-embed-text",
				Type:           "ollama",
				OllamaEndpoint: "http://localhost:11434",
			},
			wantErr: false,
		},
		{
			name: "missing model name",
			config: &ai.ModelConfig{
				Type:           "ollama",
				OllamaEndpoint: "http://localhost:11434",
			},
			wantErr: true,
		},
		{
			name:    "nil config",
			config:  nil,
			wantErr: true,
		},
		{
			name: "default endpoint",
			config: &ai.ModelConfig{
				Path: "nomic-embed-text",
				Type: "ollama",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine, err := NewOllamaEmbeddingEngine(tt.config)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, engine)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, engine)
				if tt.config.OllamaEndpoint == "" {
					assert.Equal(t, "http://localhost:11434", engine.config.OllamaEndpoint)
				}
			}
		})
	}
}

func TestOllamaEmbeddingEngine_Embed(t *testing.T) {
	server := mockOllamaServer(t)
	defer server.Close()

	config := &ai.ModelConfig{
		Path:                "nomic-embed-text",
		Type:                "ollama",
		OllamaEndpoint:      server.URL,
		NormalizeEmbeddings: true,
		Dimensions:          768,
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = engine.Warm(ctx)
	require.NoError(t, err)

	tests := []struct {
		name    string
		content []string
		wantLen int
	}{
		{
			name:    "single text",
			content: []string{"Hello, world!"},
			wantLen: 1,
		},
		{
			name:    "multiple texts",
			content: []string{"First text", "Second text", "Third text"},
			wantLen: 3,
		},
		{
			name:    "empty input",
			content: []string{},
			wantLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embeddings, err := engine.Embed(ctx, tt.content)
			require.NoError(t, err)
			assert.Len(t, embeddings, tt.wantLen)

			for i, embedding := range embeddings {
				assert.Len(t, embedding, 768, "embedding %d should have 768 dimensions", i)
				
				// Check normalization
				if config.NormalizeEmbeddings && len(embedding) > 0 {
					var sum float32
					for _, val := range embedding {
						sum += val * val
					}
					assert.InDelta(t, 1.0, sum, 0.01, "normalized embedding should have unit length")
				}
			}
		})
	}
}

func TestOllamaEmbeddingEngine_EmbedBatch(t *testing.T) {
	server := mockOllamaServer(t)
	defer server.Close()

	config := &ai.ModelConfig{
		Path:           "all-minilm",
		Type:           "ollama",
		OllamaEndpoint: server.URL,
		BatchSize:      2,
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = engine.Warm(ctx)
	require.NoError(t, err)

	content := []string{"Text 1", "Text 2", "Text 3", "Text 4", "Text 5"}
	embeddings, err := engine.EmbedBatch(ctx, content, 2)
	require.NoError(t, err)
	assert.Len(t, embeddings, 5)

	for _, embedding := range embeddings {
		assert.Len(t, embedding, 384)
	}
}

func TestOllamaEmbeddingEngine_GetModelInfo(t *testing.T) {
	config := &ai.ModelConfig{
		Path:       "nomic-embed-text",
		Type:       "ollama",
		Dimensions: 768,
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	info := engine.GetModelInfo()
	assert.Equal(t, "nomic-embed-text", info.Name)
	assert.Equal(t, 768, info.Dimension)
}

func TestOllamaEmbeddingEngine_ErrorHandling(t *testing.T) {
	// Test server that returns errors
	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal server error"))
	}))
	defer errorServer.Close()

	config := &ai.ModelConfig{
		Path:           "nomic-embed-text",
		Type:           "ollama",
		OllamaEndpoint: errorServer.URL,
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = engine.Warm(ctx)
	assert.Error(t, err)
}

func TestOllamaEmbeddingEngine_Timeout(t *testing.T) {
	// Test server that delays response
	slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(OllamaEmbedResponse{
			Embedding: make([]float32, 768),
		})
	}))
	defer slowServer.Close()

	config := &ai.ModelConfig{
		Path:           "nomic-embed-text",
		Type:           "ollama",
		OllamaEndpoint: slowServer.URL,
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	// Set a short timeout
	engine.httpClient.Timeout = 50 * time.Millisecond

	ctx := context.Background()
	err = engine.Warm(ctx)
	assert.Error(t, err)
}

func TestOllamaEmbeddingEngine_DimensionAutoDetection(t *testing.T) {
	server := mockOllamaServer(t)
	defer server.Close()

	config := &ai.ModelConfig{
		Path:           "nomic-embed-text",
		Type:           "ollama",
		OllamaEndpoint: server.URL,
		// Don't set Dimensions - let it auto-detect
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = engine.Warm(ctx)
	require.NoError(t, err)

	// After warm-up, dimensions should be set
	assert.Equal(t, 768, engine.config.Dimensions)
	assert.Equal(t, 768, engine.modelInfo.Dimension)
}

func TestOllamaEmbeddingEngine_Stats(t *testing.T) {
	server := mockOllamaServer(t)
	defer server.Close()

	config := &ai.ModelConfig{
		Path:           "nomic-embed-text",
		Type:           "ollama",
		OllamaEndpoint: server.URL,
	}

	engine, err := NewOllamaEmbeddingEngine(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = engine.Warm(ctx)
	require.NoError(t, err)

	// Generate some embeddings
	for i := 0; i < 5; i++ {
		_, err = engine.Embed(ctx, []string{"Test text"})
		require.NoError(t, err)
	}

	// Check stats
	assert.Equal(t, int64(5), engine.stats.TotalInferences)
	assert.Equal(t, int64(5), engine.stats.TotalTokens)
	assert.Greater(t, engine.stats.AverageLatency, time.Duration(0))
	assert.Greater(t, engine.stats.ThroughputTPS, float64(0))
}