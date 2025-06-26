package core

import (
	"testing"
)

func BenchmarkCosineSimilarity(b *testing.B) {
	a := make([]float32, 128)
	vec := make([]float32, 128)

	// Initialize with some values
	for i := range a {
		a[i] = float32(i) * 0.1
		vec[i] = float32(i+1) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = CosineSimilarity(a, vec)
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	a := make([]float32, 128)
	vec := make([]float32, 128)

	// Initialize with some values
	for i := range a {
		a[i] = float32(i) * 0.1
		vec[i] = float32(i+1) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EuclideanDistance(a, vec)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	a := make([]float32, 128)
	vec := make([]float32, 128)

	// Initialize with some values
	for i := range a {
		a[i] = float32(i) * 0.1
		vec[i] = float32(i+1) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DotProduct(a, vec)
	}
}
