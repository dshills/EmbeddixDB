package index

import "math"

// HNSWConfig contains configuration parameters for HNSW index
type HNSWConfig struct {
	// M is the number of bi-directional links for every new element during construction
	// Higher M leads to better recall but slower construction and more memory usage
	// Typical values: 16-64
	M int

	// MMax is the maximum number of connections for level > 0
	// Usually set to M
	MMax int

	// ML (mL) is the level normalization factor
	// Used in level assignment: level = floor(-ln(unif(0,1)) * mL)
	// Typical value: 1/ln(2.0) ≈ 1.442
	ML float64

	// EfConstruction is the size of the dynamic candidate list
	// Higher values improve quality but slow down construction
	// Typical values: 100-800
	EfConstruction int

	// EfSearch is the size of the dynamic candidate list for search
	// Can be set lower than EfConstruction for faster search
	// Should be >= k (number of nearest neighbors requested)
	EfSearch int

	// MaxLevels is the maximum number of levels in the graph
	// Usually calculated as: ceil(log2(maxElements)) + 1
	MaxLevels int

	// Seed for random number generation (for reproducible builds)
	Seed int64
}

// DefaultHNSWConfig returns a configuration with sensible defaults
func DefaultHNSWConfig() HNSWConfig {
	return HNSWConfig{
		M:              16,
		MMax:           16,
		ML:             1.0 / math.Log(2.0), // 1/ln(2)
		EfConstruction: 200,
		EfSearch:       50,
		MaxLevels:      16,
		Seed:           42,
	}
}

// ValidateConfig validates HNSW configuration parameters
func (c HNSWConfig) Validate() error {
	// Implementation will be added with validation logic
	return nil
}
