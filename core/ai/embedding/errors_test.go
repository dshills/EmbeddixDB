package embedding

import (
	"errors"
	"testing"
)

func TestEmbeddingError(t *testing.T) {
	tests := []struct {
		name      string
		err       *EmbeddingError
		wantMsg   string
		retryable bool
	}{
		{
			name:      "model not found error",
			err:       NewEmbeddingError("load", "test-model", ErrModelNotFound, "/path/to/model", false),
			wantMsg:   "embedding error in load with model test-model: embedding model not found (/path/to/model)",
			retryable: false,
		},
		{
			name:      "inference timeout error",
			err:       NewEmbeddingError("inference", "bert-base", ErrInferenceTimeout, "processing took too long", true),
			wantMsg:   "embedding error in inference with model bert-base: embedding inference timeout (processing took too long)",
			retryable: true,
		},
		{
			name:      "empty input error",
			err:       NewEmbeddingError("embed", "all-MiniLM-L6-v2", ErrEmptyInput, "", false),
			wantMsg:   "embedding error in embed with model all-MiniLM-L6-v2: empty input for embedding",
			retryable: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.wantMsg {
				t.Errorf("Error() = %v, want %v", got, tt.wantMsg)
			}
			if got := tt.err.IsRetryable(); got != tt.retryable {
				t.Errorf("IsRetryable() = %v, want %v", got, tt.retryable)
			}
		})
	}
}

func TestErrorChecking(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		checkFn  func(error) bool
		expected bool
	}{
		{
			name:     "IsModelNotFound with direct error",
			err:      ErrModelNotFound,
			checkFn:  IsModelNotFound,
			expected: true,
		},
		{
			name:     "IsModelNotFound with wrapped error",
			err:      NewEmbeddingError("load", "model", ErrModelNotFound, "", false),
			checkFn:  IsModelNotFound,
			expected: true,
		},
		{
			name:     "IsTimeout with direct error",
			err:      ErrInferenceTimeout,
			checkFn:  IsTimeout,
			expected: true,
		},
		{
			name:     "IsInvalidInput with empty input error",
			err:      ErrEmptyInput,
			checkFn:  IsInvalidInput,
			expected: true,
		},
		{
			name:     "IsInvalidInput with invalid input error",
			err:      ErrInvalidInput,
			checkFn:  IsInvalidInput,
			expected: true,
		},
		{
			name:     "IsResourceExhausted with direct error",
			err:      ErrResourceExhausted,
			checkFn:  IsResourceExhausted,
			expected: true,
		},
		{
			name:     "IsEmbeddingError with EmbeddingError",
			err:      NewEmbeddingError("test", "model", errors.New("test"), "", false),
			checkFn:  IsEmbeddingError,
			expected: true,
		},
		{
			name:     "IsEmbeddingError with regular error",
			err:      errors.New("regular error"),
			checkFn:  IsEmbeddingError,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.checkFn(tt.err); got != tt.expected {
				t.Errorf("%s = %v, want %v", tt.name, got, tt.expected)
			}
		})
	}
}

func TestErrorUnwrap(t *testing.T) {
	baseErr := errors.New("base error")
	embErr := NewEmbeddingError("test", "model", baseErr, "", false)

	if unwrapped := embErr.Unwrap(); unwrapped != baseErr {
		t.Errorf("Unwrap() = %v, want %v", unwrapped, baseErr)
	}

	// Test with errors.Is
	if !errors.Is(embErr, baseErr) {
		t.Error("errors.Is should find the base error")
	}
}
