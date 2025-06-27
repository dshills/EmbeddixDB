package embedding

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/dshills/EmbeddixDB/core/ai"
)

func TestEmbeddingErrorScenarios(t *testing.T) {
	ctx := context.Background()

	t.Run("model not found", func(t *testing.T) {
		config := ai.ModelConfig{
			Name:      "non-existent-model",
			Type:      "onnx",
			Path:      "/path/to/non/existent/model.onnx",
			BatchSize: 32,
		}

		_, err := NewONNXEmbeddingEngine(config.Path, config)
		if err == nil {
			t.Fatal("expected error for non-existent model")
		}

		// Check if it's the correct error type
		if !IsModelNotFound(err) {
			t.Errorf("expected model not found error, got: %v", err)
		}

		// Check if it's an embedding error
		var embErr *EmbeddingError
		if !IsEmbeddingError(err) {
			t.Error("expected EmbeddingError type")
		} else if errors.As(err, &embErr) {
			if embErr.Retryable {
				t.Error("model not found should not be retryable")
			}
		}
	})

	t.Run("empty input", func(t *testing.T) {
		// Use mock engine for this test
		os.Setenv("EMBEDDIX_USE_MOCK_ONNX", "true")
		defer os.Unsetenv("EMBEDDIX_USE_MOCK_ONNX")

		config := ai.ModelConfig{
			Name:      "test-model",
			Type:      "onnx",
			Path:      "", // Empty path triggers mock
			BatchSize: 32,
		}

		engine, err := NewONNXEmbeddingEngine(config.Path, config)
		if err != nil {
			t.Fatalf("failed to create mock engine: %v", err)
		}
		defer engine.Close()

		// Test with empty input
		_, err = engine.Embed(ctx, []string{})
		if err == nil {
			t.Fatal("expected error for empty input")
		}

		// Check if it's the correct error type
		if !IsInvalidInput(err) {
			t.Errorf("expected invalid input error, got: %v", err)
		}
	})

	t.Run("nil session", func(t *testing.T) {
		// Create engine with mock
		os.Setenv("EMBEDDIX_USE_MOCK_ONNX", "true")
		defer os.Unsetenv("EMBEDDIX_USE_MOCK_ONNX")

		config := ai.ModelConfig{
			Name:      "test-model",
			Type:      "onnx",
			Path:      "",
			BatchSize: 32,
		}

		engine, err := NewONNXEmbeddingEngine(config.Path, config)
		if err != nil {
			t.Fatalf("failed to create engine: %v", err)
		}

		// Manually set session to nil to simulate uninitialized state
		engine.session = nil

		// Try to embed
		_, err = engine.Embed(ctx, []string{"test content"})
		if err == nil {
			t.Fatal("expected error for nil session")
		}

		// Check error message
		if !IsEmbeddingError(err) {
			t.Errorf("expected EmbeddingError, got: %v", err)
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		// Use mock engine
		os.Setenv("EMBEDDIX_USE_MOCK_ONNX", "true")
		defer os.Unsetenv("EMBEDDIX_USE_MOCK_ONNX")

		config := ai.ModelConfig{
			Name:            "test-model",
			Type:            "onnx",
			Path:            "",
			BatchSize:       32,
			TimeoutDuration: 100 * time.Millisecond,
		}

		engine, err := NewONNXEmbeddingEngine(config.Path, config)
		if err != nil {
			t.Fatalf("failed to create engine: %v", err)
		}
		defer engine.Close()

		// Create context that will be cancelled
		cancelCtx, cancel := context.WithCancel(ctx)
		cancel() // Cancel immediately

		// Try to embed with cancelled context
		_, err = engine.Embed(cancelCtx, []string{"test content"})
		if err == nil {
			t.Fatal("expected error for cancelled context")
		}
	})
}

// TestAPIErrorHandling tests error handling at the API level
func TestAPIErrorHandling(t *testing.T) {
	// This test would require setting up a full API handler
	// Here we just demonstrate the error type checking that would happen

	scenarios := []struct {
		name           string
		err            error
		expectedStatus int
		expectedMsg    string
	}{
		{
			name:           "model not found",
			err:            NewEmbeddingError("load", "bert", ErrModelNotFound, "", false),
			expectedStatus: 404,
			expectedMsg:    "Model not found: bert",
		},
		{
			name:           "invalid input",
			err:            NewEmbeddingError("embed", "model", ErrInvalidInput, "", false),
			expectedStatus: 400,
			expectedMsg:    "Invalid or empty input for embedding",
		},
		{
			name:           "resource exhausted",
			err:            NewEmbeddingError("embed", "model", ErrResourceExhausted, "", false),
			expectedStatus: 503,
			expectedMsg:    "Embedding service temporarily unavailable due to high load",
		},
		{
			name:           "timeout",
			err:            NewEmbeddingError("embed", "model", ErrInferenceTimeout, "", true),
			expectedStatus: 408,
			expectedMsg:    "Embedding generation timed out",
		},
	}

	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			// This simulates what the API handler would do
			status := 500 // default
			msg := sc.err.Error()

			var embErr *EmbeddingError
			if errors.As(sc.err, &embErr) {
				if errors.Is(sc.err, ErrModelNotFound) {
					status = 404
					msg = "Model not found: " + embErr.Model
				} else if errors.Is(sc.err, ErrInvalidInput) || errors.Is(sc.err, ErrEmptyInput) {
					status = 400
					msg = "Invalid or empty input for embedding"
				} else if errors.Is(sc.err, ErrResourceExhausted) {
					status = 503
					msg = "Embedding service temporarily unavailable due to high load"
				} else if errors.Is(sc.err, ErrInferenceTimeout) {
					status = 408
					msg = "Embedding generation timed out"
				}
			}

			if status != sc.expectedStatus {
				t.Errorf("expected status %d, got %d", sc.expectedStatus, status)
			}
			if msg != sc.expectedMsg {
				t.Errorf("expected message %q, got %q", sc.expectedMsg, msg)
			}
		})
	}
}

