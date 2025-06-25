package core

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
		wantErr  bool
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			wantErr:  false,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
			wantErr:  false,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
			wantErr:  false,
		},
		{
			name:    "different dimensions",
			a:       []float32{1, 0},
			b:       []float32{1, 0, 0},
			wantErr: true,
		},
		{
			name:     "zero vector",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 0.0,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := CosineSimilarity(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("CosineSimilarity() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(float64(result-tt.expected)) > 1e-6 {
				t.Errorf("CosineSimilarity() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
		wantErr  bool
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0.0,
			wantErr:  false,
		},
		{
			name:     "unit distance",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			wantErr:  false,
		},
		{
			name:     "pythagorean distance",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 5.0,
			wantErr:  false,
		},
		{
			name:    "different dimensions",
			a:       []float32{1, 0},
			b:       []float32{1, 0, 0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := EuclideanDistance(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("EuclideanDistance() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(float64(result-tt.expected)) > 1e-6 {
				t.Errorf("EuclideanDistance() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
		wantErr  bool
	}{
		{
			name:     "simple dot product",
			a:        []float32{1, 2, 3},
			b:        []float32{4, 5, 6},
			expected: 32.0, // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
			wantErr:  false,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 0.0,
			wantErr:  false,
		},
		{
			name:    "different dimensions",
			a:       []float32{1, 0},
			b:       []float32{1, 0, 0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := DotProduct(tt.a, tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("DotProduct() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(float64(result-tt.expected)) > 1e-6 {
				t.Errorf("DotProduct() = %v, want %v", result, tt.expected)
			}
		})
	}
}