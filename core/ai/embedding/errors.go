package embedding

import (
	"errors"
	"fmt"
)

// Common embedding errors
var (
	// ErrModelNotFound indicates the requested model could not be found
	ErrModelNotFound = errors.New("embedding model not found")

	// ErrModelNotLoaded indicates the model is not loaded or initialized
	ErrModelNotLoaded = errors.New("embedding model not loaded")

	// ErrModelInitFailed indicates model initialization failed
	ErrModelInitFailed = errors.New("embedding model initialization failed")

	// ErrInferenceTimeout indicates the inference operation timed out
	ErrInferenceTimeout = errors.New("embedding inference timeout")

	// ErrInvalidInput indicates the input data is invalid
	ErrInvalidInput = errors.New("invalid input for embedding")

	// ErrEmptyInput indicates no input was provided
	ErrEmptyInput = errors.New("empty input for embedding")

	// ErrDimensionMismatch indicates dimension mismatch in embeddings
	ErrDimensionMismatch = errors.New("embedding dimension mismatch")

	// ErrResourceExhausted indicates system resources are exhausted
	ErrResourceExhausted = errors.New("embedding resources exhausted")

	// ErrUnsupportedModel indicates the model type is not supported
	ErrUnsupportedModel = errors.New("unsupported embedding model")

	// ErrCorruptedModel indicates the model file is corrupted
	ErrCorruptedModel = errors.New("corrupted embedding model")
)

// EmbeddingError represents a structured embedding error with context
type EmbeddingError struct {
	Op        string // Operation that failed
	Model     string // Model name
	Err       error  // Underlying error
	Details   string // Additional details
	Retryable bool   // Whether the operation can be retried
}

// Error implements the error interface
func (e *EmbeddingError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("embedding error in %s with model %s: %v (%s)", e.Op, e.Model, e.Err, e.Details)
	}
	return fmt.Sprintf("embedding error in %s with model %s: %v", e.Op, e.Model, e.Err)
}

// Unwrap returns the underlying error
func (e *EmbeddingError) Unwrap() error {
	return e.Err
}

// IsRetryable returns whether the error is retryable
func (e *EmbeddingError) IsRetryable() bool {
	return e.Retryable
}

// NewEmbeddingError creates a new embedding error
func NewEmbeddingError(op, model string, err error, details string, retryable bool) *EmbeddingError {
	return &EmbeddingError{
		Op:        op,
		Model:     model,
		Err:       err,
		Details:   details,
		Retryable: retryable,
	}
}

// IsEmbeddingError checks if an error is an EmbeddingError
func IsEmbeddingError(err error) bool {
	var embErr *EmbeddingError
	return errors.As(err, &embErr)
}

// IsModelNotFound checks if the error indicates a model was not found
func IsModelNotFound(err error) bool {
	return errors.Is(err, ErrModelNotFound)
}

// IsTimeout checks if the error indicates a timeout
func IsTimeout(err error) bool {
	return errors.Is(err, ErrInferenceTimeout)
}

// IsInvalidInput checks if the error indicates invalid input
func IsInvalidInput(err error) bool {
	return errors.Is(err, ErrInvalidInput) || errors.Is(err, ErrEmptyInput)
}

// IsResourceExhausted checks if the error indicates exhausted resources
func IsResourceExhausted(err error) bool {
	return errors.Is(err, ErrResourceExhausted)
}
