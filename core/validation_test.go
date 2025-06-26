package core

import (
	"math"
	"testing"
	"time"
)

func TestValidateVector(t *testing.T) {
	tests := []struct {
		name    string
		vector  Vector
		wantErr bool
	}{
		{
			name: "valid vector",
			vector: Vector{
				ID:     "test1",
				Values: []float32{1.0, 2.0, 3.0},
			},
			wantErr: false,
		},
		{
			name: "empty ID",
			vector: Vector{
				ID:     "",
				Values: []float32{1.0, 2.0, 3.0},
			},
			wantErr: true,
		},
		{
			name: "empty values",
			vector: Vector{
				ID:     "test1",
				Values: []float32{},
			},
			wantErr: true,
		},
		{
			name: "NaN value",
			vector: Vector{
				ID:     "test1",
				Values: []float32{1.0, float32(math.NaN()), 3.0},
			},
			wantErr: true,
		},
		{
			name: "infinite value",
			vector: Vector{
				ID:     "test1",
				Values: []float32{1.0, float32(math.Inf(1)), 3.0},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateVector(tt.vector)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateVector() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateCollection(t *testing.T) {
	tests := []struct {
		name       string
		collection Collection
		wantErr    bool
	}{
		{
			name: "valid collection",
			collection: Collection{
				Name:      "test",
				Dimension: 128,
				IndexType: "flat",
				Distance:  "cosine",
				CreatedAt: time.Now(),
			},
			wantErr: false,
		},
		{
			name: "empty name",
			collection: Collection{
				Name:      "",
				Dimension: 128,
				IndexType: "flat",
				Distance:  "cosine",
			},
			wantErr: true,
		},
		{
			name: "invalid dimension",
			collection: Collection{
				Name:      "test",
				Dimension: 0,
				IndexType: "flat",
				Distance:  "cosine",
			},
			wantErr: true,
		},
		{
			name: "invalid index type",
			collection: Collection{
				Name:      "test",
				Dimension: 128,
				IndexType: "invalid",
				Distance:  "cosine",
			},
			wantErr: true,
		},
		{
			name: "invalid distance metric",
			collection: Collection{
				Name:      "test",
				Dimension: 128,
				IndexType: "flat",
				Distance:  "invalid",
			},
			wantErr: true,
		},
		{
			name: "name with path separator",
			collection: Collection{
				Name:      "test/collection",
				Dimension: 128,
				IndexType: "flat",
				Distance:  "cosine",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateCollection(tt.collection)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateCollection() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
