package core

import (
	"fmt"
	"strings"
)

// ValidateVector checks if a vector is valid
func ValidateVector(vec Vector) error {
	if vec.ID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	if len(vec.Values) == 0 {
		return fmt.Errorf("vector values cannot be empty")
	}

	// Check for NaN or infinite values
	for i, val := range vec.Values {
		if isNaN(val) {
			return fmt.Errorf("vector contains NaN at index %d", i)
		}
		if isInf(val) {
			return fmt.Errorf("vector contains infinite value at index %d", i)
		}
	}

	return nil
}

// ValidateCollection checks if a collection specification is valid
func ValidateCollection(collection Collection) error {
	if collection.Name == "" {
		return fmt.Errorf("collection name cannot be empty")
	}

	if strings.Contains(collection.Name, "/") || strings.Contains(collection.Name, "\\") {
		return fmt.Errorf("collection name cannot contain path separators")
	}

	if collection.Dimension <= 0 {
		return fmt.Errorf("collection dimension must be positive, got %d", collection.Dimension)
	}

	if !isValidIndexType(collection.IndexType) {
		return fmt.Errorf("invalid index type: %s", collection.IndexType)
	}

	if !isValidDistanceMetric(collection.Distance) {
		return fmt.Errorf("invalid distance metric: %s", collection.Distance)
	}

	return nil
} // ValidateSearchRequest checks if a search request is valid
func ValidateSearchRequest(req SearchRequest, dimension int) error {
	if len(req.Query) == 0 {
		return fmt.Errorf("query vector cannot be empty")
	}

	if len(req.Query) != dimension {
		return fmt.Errorf("query dimension %d does not match collection dimension %d",
			len(req.Query), dimension)
	}

	if req.TopK <= 0 {
		return fmt.Errorf("topK must be positive, got %d", req.TopK)
	}

	// Check for NaN or infinite values in query
	for i, val := range req.Query {
		if isNaN(val) {
			return fmt.Errorf("query contains NaN at index %d", i)
		}
		if isInf(val) {
			return fmt.Errorf("query contains infinite value at index %d", i)
		}
	}

	return nil
}

// ValidateVectorDimension checks if vector matches expected dimension
func ValidateVectorDimension(vec Vector, expectedDim int) error {
	if len(vec.Values) != expectedDim {
		return fmt.Errorf("vector dimension %d does not match expected dimension %d",
			len(vec.Values), expectedDim)
	}
	return nil
}

// Helper functions for NaN and Inf detection
func isNaN(f float32) bool {
	return f != f
}

func isInf(f float32) bool {
	return f > 3.4e38 || f < -3.4e38
}

func isValidIndexType(indexType string) bool {
	validTypes := []string{"flat", "hnsw"}
	for _, valid := range validTypes {
		if indexType == valid {
			return true
		}
	}
	return false
}

func isValidDistanceMetric(metric string) bool {
	validMetrics := []string{"cosine", "l2", "dot"}
	for _, valid := range validMetrics {
		if metric == valid {
			return true
		}
	}
	return false
}
