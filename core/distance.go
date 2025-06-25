package core

import (
	"fmt"
	"math"
)

// DistanceMetric represents supported distance calculation methods
type DistanceMetric string

const (
	DistanceCosine    DistanceMetric = "cosine"
	DistanceL2        DistanceMetric = "l2"
	DistanceEuclidean DistanceMetric = "l2" // Alias for L2
	DistanceDot       DistanceMetric = "dot"
)

// CosineSimilarity calculates cosine similarity between two vectors
// Returns similarity score (higher = more similar)
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector dimensions must match: %d != %d", len(a), len(b))
	}
	
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0, nil
	}
	
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))), nil
}

// CosineDistance calculates cosine distance (1 - cosine similarity)
// Returns distance score (lower = more similar)
func CosineDistance(a, b []float32) (float32, error) {
	similarity, err := CosineSimilarity(a, b)
	if err != nil {
		return 0, err
	}
	return 1 - similarity, nil
}// DotProduct calculates dot product between two vectors
func DotProduct(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector dimensions must match: %d != %d", len(a), len(b))
	}
	
	var product float32
	for i := range a {
		product += a[i] * b[i]
	}
	
	return product, nil
}

// EuclideanDistance calculates L2 (Euclidean) distance between two vectors
// Returns distance score (lower = more similar)
func EuclideanDistance(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector dimensions must match: %d != %d", len(a), len(b))
	}
	
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	
	return float32(math.Sqrt(float64(sum))), nil
}

// CalculateDistance calculates distance using the specified metric
func CalculateDistance(a, b []float32, metric DistanceMetric) (float32, error) {
	switch metric {
	case DistanceCosine:
		return CosineDistance(a, b)
	case DistanceL2:
		return EuclideanDistance(a, b)
	case DistanceDot:
		// For dot product, we return negative value so that higher values = closer
		dot, err := DotProduct(a, b)
		return -dot, err
	default:
		return 0, fmt.Errorf("unsupported distance metric: %s", metric)
	}
}

// CalculateDistanceOptimized uses SIMD optimizations when available.
// This function provides vectorized implementations for supported architectures
// and falls back to scalar implementations otherwise.
// For dot product, returns negative value so higher values indicate closer similarity.
func CalculateDistanceOptimized(a, b []float32, metric DistanceMetric) (float32, error) {
	switch metric {
	case DistanceCosine:
		return CosineDistanceSIMD(a, b)
	case DistanceL2:
		return EuclideanDistanceSIMD(a, b)
	case DistanceDot:
		dot, err := DotProductSIMD(a, b)
		return -dot, err
	default:
		return 0, fmt.Errorf("unsupported distance metric: %s", metric)
	}
}