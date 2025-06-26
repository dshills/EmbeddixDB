package core

import "errors"

// Common errors
var (
	ErrVectorNotFound    = errors.New("vector not found")
	ErrCollectionExists  = errors.New("collection already exists")
	ErrCollectionNotFound = errors.New("collection not found")
	ErrInvalidDimension  = errors.New("invalid vector dimension")
	ErrInvalidDistance   = errors.New("invalid distance metric")
	ErrInvalidIndexType  = errors.New("invalid index type")
	ErrEmptyVector       = errors.New("empty vector")
	ErrDuplicateID       = errors.New("duplicate vector ID")
)