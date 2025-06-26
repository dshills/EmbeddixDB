package quantization

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKMeans(t *testing.T) {
	// Generate test data
	data := generateClusteredData(100, 2, 3, 42)

	config := DefaultKMeansConfig(3)
	config.Seed = 42 // For reproducibility

	kmeans := NewKMeans(config)
	ctx := context.Background()

	result, err := kmeans.Fit(ctx, data)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Check basic properties
	assert.Len(t, result.Centroids, 3)
	assert.Len(t, result.Assignments, len(data))
	assert.True(t, result.Converged)
	assert.Greater(t, result.Iterations, 0)
	assert.Greater(t, result.Inertia, 0.0)

	// Check all points are assigned
	for _, assignment := range result.Assignments {
		assert.GreaterOrEqual(t, assignment, 0)
		assert.Less(t, assignment, 3)
	}

	// Check cluster sizes
	totalPoints := 0
	for _, size := range result.ClusterSizes {
		totalPoints += size
		assert.Greater(t, size, 0) // No empty clusters
	}
	assert.Equal(t, len(data), totalPoints)
}

func TestKMeansInitialization(t *testing.T) {
	data := generateRandomData(50, 3, 42)
	ctx := context.Background()

	initMethods := []InitMethod{InitRandom, InitKMeansPP, InitUniform}

	for _, method := range initMethods {
		t.Run(string(method), func(t *testing.T) {
			config := DefaultKMeansConfig(5)
			config.InitMethod = method
			config.Seed = 42

			kmeans := NewKMeans(config)
			result, err := kmeans.Fit(ctx, data)

			require.NoError(t, err)
			assert.Len(t, result.Centroids, 5)
			assert.Len(t, result.ClusterSizes, 5)
		})
	}
}

func TestProductQuantizer(t *testing.T) {
	dimension := 128
	numVectors := 1000

	// Generate test data
	vectors := generateRandomData(numVectors, dimension, 42)

	// Create and configure quantizer
	config := DefaultProductQuantizerConfig(dimension)
	config.NumSubvectors = 8
	config.BitsPerSubvector = 4 // Reduce to 16 clusters per subvector

	pq, err := NewProductQuantizer(config)
	require.NoError(t, err)
	require.NotNil(t, pq)

	// Check initial state
	assert.False(t, pq.IsTrained())
	assert.Equal(t, dimension, pq.Config().Dimension)
	assert.Equal(t, config.NumSubvectors, len(pq.codebooks))

	// Train quantizer
	ctx := context.Background()
	err = pq.Train(ctx, vectors[:500]) // Use subset for training
	require.NoError(t, err)

	// Check trained state
	assert.True(t, pq.IsTrained())
	assert.Greater(t, pq.MemoryReduction(), 1.0)

	stats := pq.GetTrainingStats()
	assert.Greater(t, stats.TrainingTime, time.Duration(0))
	assert.Equal(t, 500, stats.TrainingVectors)

	// Test encoding/decoding
	for i := 0; i < 10; i++ {
		vector := vectors[i]

		// Encode
		code, err := pq.Encode(vector)
		require.NoError(t, err)
		assert.Equal(t, pq.CodeSize(), len(code))

		// Decode
		decoded, err := pq.Decode(code)
		require.NoError(t, err)
		assert.Len(t, decoded, dimension)

		// Check reconstruction quality (should be reasonable)
		mse := calculateMSE(vector, decoded)
		assert.Less(t, mse, 10.0) // Reasonable bound for test data
	}
}

func TestProductQuantizerDistances(t *testing.T) {
	dimension := 64
	vectors := generateRandomData(100, dimension, 42)

	config := DefaultProductQuantizerConfig(dimension)
	config.NumSubvectors = 4
	config.BitsPerSubvector = 4 // Reduce to 16 clusters per subvector

	pq, err := NewProductQuantizer(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = pq.Train(ctx, vectors[:50])
	require.NoError(t, err)

	// Test symmetric distance
	vec1, vec2 := vectors[0], vectors[1]

	code1, err := pq.Encode(vec1)
	require.NoError(t, err)

	code2, err := pq.Encode(vec2)
	require.NoError(t, err)

	dist, err := pq.Distance(code1, code2)
	require.NoError(t, err)
	assert.Greater(t, dist, float32(0))

	// Test asymmetric distance
	asymDist, err := pq.AsymmetricDistance(code1, vec2)
	require.NoError(t, err)
	assert.Greater(t, asymDist, float32(0))

	// Distance to self should be zero (or very small)
	selfDist, err := pq.Distance(code1, code1)
	require.NoError(t, err)
	assert.Less(t, selfDist, float32(0.1))
}

func TestScalarQuantizer(t *testing.T) {
	dimension := 100
	numVectors := 500

	vectors := generateRandomData(numVectors, dimension, 42)

	config := DefaultScalarQuantizerConfig(dimension)
	config.BitsPerComponent = 8

	sq, err := NewScalarQuantizer(config)
	require.NoError(t, err)
	require.NotNil(t, sq)

	// Check initial state
	assert.False(t, sq.IsTrained())
	assert.Equal(t, dimension, sq.Config().Dimension)

	// Train quantizer
	ctx := context.Background()
	err = sq.Train(ctx, vectors[:200])
	require.NoError(t, err)

	// Check trained state
	assert.True(t, sq.IsTrained())
	assert.Greater(t, sq.MemoryReduction(), 1.0)

	// Test encoding/decoding
	for i := 0; i < 10; i++ {
		vector := vectors[i]

		// Encode
		code, err := sq.Encode(vector)
		require.NoError(t, err)
		assert.Equal(t, sq.CodeSize(), len(code))

		// Decode
		decoded, err := sq.Decode(code)
		require.NoError(t, err)
		assert.Len(t, decoded, dimension)

		// Test distance computation
		code2, err := sq.Encode(vectors[i+1])
		require.NoError(t, err)

		dist, err := sq.Distance(code, code2)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, dist, float32(0))

		// Test asymmetric distance
		asymDist, err := sq.AsymmetricDistance(code, vectors[i+1])
		require.NoError(t, err)
		assert.GreaterOrEqual(t, asymDist, float32(0))
	}
}

func TestScalarQuantizerDifferentBits(t *testing.T) {
	dimension := 32
	vectors := generateRandomData(100, dimension, 42)

	bitCounts := []int{4, 6, 8, 16}

	for _, bits := range bitCounts {
		t.Run(fmt.Sprintf("bits_%d", bits), func(t *testing.T) {
			config := DefaultScalarQuantizerConfig(dimension)
			config.BitsPerComponent = bits

			sq, err := NewScalarQuantizer(config)
			require.NoError(t, err)

			ctx := context.Background()
			err = sq.Train(ctx, vectors[:50])
			require.NoError(t, err)

			// Check compression ratio increases with fewer bits
			expectedSize := (dimension*bits + 7) / 8
			assert.Equal(t, expectedSize, sq.CodeSize())

			// Check encoding works
			code, err := sq.Encode(vectors[0])
			require.NoError(t, err)
			assert.Equal(t, expectedSize, len(code))

			// Check decoding works
			decoded, err := sq.Decode(code)
			require.NoError(t, err)
			assert.Len(t, decoded, dimension)
		})
	}
}

func TestQuantizerFactory(t *testing.T) {
	factory := NewQuantizerFactory()

	// Test supported types
	supportedTypes := factory.SupportedTypes()
	assert.Contains(t, supportedTypes, ProductQuantization)
	assert.Contains(t, supportedTypes, ScalarQuantization)

	// Test Product Quantizer creation
	config := QuantizerConfig{
		Type:           ProductQuantization,
		Dimension:      128,
		DistanceMetric: "l2",
		MemoryBudgetMB: 10,
	}

	pq, err := factory.CreateQuantizer(config)
	require.NoError(t, err)
	assert.Equal(t, ProductQuantization, pq.Config().Type)

	// Test Scalar Quantizer creation
	config.Type = ScalarQuantization
	config.Dimension = 64

	sq, err := factory.CreateQuantizer(config)
	require.NoError(t, err)
	assert.Equal(t, ScalarQuantization, sq.Config().Type)

	// Test unsupported type
	config.Type = BinaryQuantization
	_, err = factory.CreateQuantizer(config)
	assert.Error(t, err)
}

func TestQuantizerPool(t *testing.T) {
	factory := NewQuantizerFactory()
	pool := NewQuantizerPool(factory)

	// Test auto-creation
	quantizer, err := pool.GetQuantizer(128)
	require.NoError(t, err)
	assert.NotNil(t, quantizer)
	assert.Equal(t, 128, quantizer.Config().Dimension)

	// Test retrieval of existing quantizer
	quantizer2, err := pool.GetQuantizer(128)
	require.NoError(t, err)
	assert.Same(t, quantizer, quantizer2)

	// Test manual registration
	config := DefaultScalarQuantizerConfig(64)
	sq, err := NewScalarQuantizer(config)
	require.NoError(t, err)

	err = pool.RegisterQuantizer(64, sq)
	require.NoError(t, err)

	retrieved, err := pool.GetQuantizer(64)
	require.NoError(t, err)
	assert.Same(t, sq, retrieved)

	// Test listing
	quantizers := pool.ListQuantizers()
	assert.Len(t, quantizers, 2)
	assert.Contains(t, quantizers, 64)
	assert.Contains(t, quantizers, 128)

	// Test removal
	err = pool.RemoveQuantizer(64)
	require.NoError(t, err)

	quantizers = pool.ListQuantizers()
	assert.Len(t, quantizers, 1)
	assert.NotContains(t, quantizers, 64)
}

func TestValidator(t *testing.T) {
	validator := NewValidator()

	// Create test data
	dimension := 32
	numVectors := 100
	vectors := generateRandomData(numVectors, dimension, 42)

	// Train quantizer
	config := DefaultProductQuantizerConfig(dimension)
	config.NumSubvectors = 4
	config.BitsPerSubvector = 4 // Reduce to 16 clusters per subvector

	pq, err := NewProductQuantizer(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = pq.Train(ctx, vectors[:50])
	require.NoError(t, err)

	// Test reconstruction validation
	result, err := validator.ValidateReconstruction(pq, vectors[50:60])
	require.NoError(t, err)

	assert.GreaterOrEqual(t, result.MSE, 0.0)
	assert.GreaterOrEqual(t, result.MAE, 0.0)
	assert.GreaterOrEqual(t, result.MaxError, 0.0)

	// Test distance validation
	result, err = validator.ValidateDistances(pq, vectors[50:60])
	require.NoError(t, err)

	assert.GreaterOrEqual(t, result.MAE, 0.0)
	assert.GreaterOrEqual(t, result.MaxError, 0.0)

	// Test recall validation (simplified)
	queries := vectors[60:65]
	database := vectors[65:75]

	result, err = validator.ValidateRecall(pq, queries, database, 5)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, result.RecallAt1, 0.0)
	assert.LessOrEqual(t, result.RecallAt1, 1.0)
	assert.GreaterOrEqual(t, result.RecallAt10, 0.0)
	assert.LessOrEqual(t, result.RecallAt10, 1.0)
}

func TestDistanceTable(t *testing.T) {
	dimension := 64
	vectors := generateRandomData(100, dimension, 42)

	config := DefaultProductQuantizerConfig(dimension)
	config.NumSubvectors = 8
	config.BitsPerSubvector = 4 // Reduce to 16 clusters per subvector

	pq, err := NewProductQuantizer(config)
	require.NoError(t, err)

	ctx := context.Background()
	err = pq.Train(ctx, vectors[:50])
	require.NoError(t, err)

	// Create distance table
	queryVector := vectors[0]
	distTable, err := pq.CreateDistanceTable(queryVector)
	require.NoError(t, err)
	assert.NotNil(t, distTable)

	// Test distance computation
	for i := 1; i < 10; i++ {
		code, err := pq.Encode(vectors[i])
		require.NoError(t, err)

		// Distance via table
		tableDist := distTable.Distance(code)

		// Distance via asymmetric method
		asymDist, err := pq.AsymmetricDistance(code, queryVector)
		require.NoError(t, err)

		// Should be approximately equal
		diff := math.Abs(float64(tableDist - asymDist))
		assert.Less(t, diff, 0.001, "Table and asymmetric distances should match")
	}

	// Test batch distances
	codes := make([][]byte, 5)
	for i := 0; i < 5; i++ {
		codes[i], err = pq.Encode(vectors[i+1])
		require.NoError(t, err)
	}

	batchDistances := distTable.BatchDistances(codes)
	assert.Len(t, batchDistances, 5)

	for i, dist := range batchDistances {
		singleDist := distTable.Distance(codes[i])
		assert.Equal(t, singleDist, dist)
	}
}

func TestElbowMethod(t *testing.T) {
	// Generate data with 3 clear clusters
	data := generateClusteredData(150, 2, 3, 42)

	ctx := context.Background()
	optimalK, inertias, err := ElbowMethod(ctx, data, 8)
	require.NoError(t, err)

	assert.Greater(t, optimalK, 1)
	assert.LessOrEqual(t, optimalK, 8)
	assert.Len(t, inertias, 7) // k=2 to k=8

	// Inertia should generally decrease
	for i := 1; i < len(inertias); i++ {
		assert.LessOrEqual(t, inertias[i], inertias[i-1], "Inertia should decrease with more clusters")
	}
}

func TestQuantizationBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping benchmark test in short mode")
	}

	dimension := 128
	numVectors := 1000
	vectors := generateRandomData(numVectors, dimension, 42)

	// Test Product Quantizer performance
	config := DefaultProductQuantizerConfig(dimension)
	pq, err := NewProductQuantizer(config)
	require.NoError(t, err)

	ctx := context.Background()

	// Measure training time
	start := time.Now()
	err = pq.Train(ctx, vectors[:500])
	require.NoError(t, err)
	trainingTime := time.Since(start)

	t.Logf("PQ training time: %v", trainingTime)
	t.Logf("Memory reduction: %.2fx", pq.MemoryReduction())

	// Measure encoding performance
	start = time.Now()
	for i := 0; i < 100; i++ {
		_, err := pq.Encode(vectors[i])
		require.NoError(t, err)
	}
	encodingTime := time.Since(start)
	t.Logf("PQ encoding time (100 vectors): %v", encodingTime)

	// Compare with Scalar Quantizer
	sqConfig := DefaultScalarQuantizerConfig(dimension)
	sq, err := NewScalarQuantizer(sqConfig)
	require.NoError(t, err)

	start = time.Now()
	err = sq.Train(ctx, vectors[:500])
	require.NoError(t, err)
	sqTrainingTime := time.Since(start)

	t.Logf("SQ training time: %v", sqTrainingTime)
	t.Logf("SQ memory reduction: %.2fx", sq.MemoryReduction())
}

// Helper functions

func generateRandomData(numVectors, dimension int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, numVectors)

	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.Float32()*2 - 1 // Random values between -1 and 1
		}
	}

	return vectors
}

func generateClusteredData(numVectors, dimension, numClusters int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, numVectors)

	// Generate cluster centers
	centers := make([][]float32, numClusters)
	for i := 0; i < numClusters; i++ {
		centers[i] = make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			centers[i][j] = rng.Float32()*10 - 5 // Centers between -5 and 5
		}
	}

	// Generate points around centers
	for i := 0; i < numVectors; i++ {
		cluster := i % numClusters
		vectors[i] = make([]float32, dimension)

		for j := 0; j < dimension; j++ {
			// Add noise around cluster center
			noise := rng.Float32()*2 - 1 // Noise between -1 and 1
			vectors[i][j] = centers[cluster][j] + noise
		}
	}

	return vectors
}

func calculateMSE(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum / float64(len(a))
}

func BenchmarkProductQuantizer(b *testing.B) {
	dimension := 128
	vectors := generateRandomData(1000, dimension, 42)

	config := DefaultProductQuantizerConfig(dimension)
	pq, _ := NewProductQuantizer(config)

	ctx := context.Background()
	_ = pq.Train(ctx, vectors[:500])

	b.Run("Encode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vector := vectors[i%len(vectors)]
			_, _ = pq.Encode(vector)
		}
	})

	// Pre-encode for decode benchmark
	codes := make([][]byte, 100)
	for i := 0; i < 100; i++ {
		codes[i], _ = pq.Encode(vectors[i])
	}

	b.Run("Decode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code := codes[i%len(codes)]
			_, _ = pq.Decode(code)
		}
	})

	b.Run("Distance", func(b *testing.B) {
		code1, _ := pq.Encode(vectors[0])

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code2, _ := pq.Encode(vectors[(i+1)%len(vectors)])
			_, _ = pq.Distance(code1, code2)
		}
	})
}

func BenchmarkScalarQuantizer(b *testing.B) {
	dimension := 128
	vectors := generateRandomData(1000, dimension, 42)

	config := DefaultScalarQuantizerConfig(dimension)
	sq, _ := NewScalarQuantizer(config)

	ctx := context.Background()
	_ = sq.Train(ctx, vectors[:500])

	b.Run("Encode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vector := vectors[i%len(vectors)]
			_, _ = sq.Encode(vector)
		}
	})

	codes := make([][]byte, 100)
	for i := 0; i < 100; i++ {
		codes[i], _ = sq.Encode(vectors[i])
	}

	b.Run("Decode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code := codes[i%len(codes)]
			_, _ = sq.Decode(code)
		}
	})
}
