//go:build !amd64

package core

// Stub implementations for non-amd64 architectures
// These fall back to the regular scalar implementations

// CosineSimilaritySIMD falls back to scalar implementation
func CosineSimilaritySIMD(a, b []float32) (float32, error) {
	return CosineSimilarity(a, b)
}

// DotProductSIMD falls back to scalar implementation
func DotProductSIMD(a, b []float32) (float32, error) {
	return DotProduct(a, b)
}

// EuclideanDistanceSIMD falls back to scalar implementation
func EuclideanDistanceSIMD(a, b []float32) (float32, error) {
	return EuclideanDistance(a, b)
}

// CosineDistanceSIMD falls back to scalar implementation
func CosineDistanceSIMD(a, b []float32) (float32, error) {
	return CosineDistance(a, b)
}

// GetSIMDInfo returns empty info for non-amd64
func GetSIMDInfo() map[string]bool {
	return map[string]bool{
		"AVX2":   false,
		"AVX512": false,
		"FMA":    false,
	}
}