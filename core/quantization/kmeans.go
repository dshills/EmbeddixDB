package quantization

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// KMeansConfig configures K-means clustering
type KMeansConfig struct {
	K                   int        `json:"k"`              // Number of clusters
	MaxIterations       int        `json:"max_iterations"` // Maximum iterations
	Tolerance           float64    `json:"tolerance"`      // Convergence tolerance
	InitMethod          InitMethod `json:"init_method"`    // Initialization method
	NumRestarts         int        `json:"num_restarts"`   // Number of random restarts
	Seed                int64      `json:"seed"`           // Random seed
	MinPointsPerCluster int        `json:"min_points_per_cluster"`
	ParallelWorkers     int        `json:"parallel_workers"` // Parallel assignment workers
}

// InitMethod represents different cluster initialization strategies
type InitMethod string

const (
	InitRandom   InitMethod = "random"   // Random initialization
	InitKMeansPP InitMethod = "kmeans++" // K-means++ initialization
	InitUniform  InitMethod = "uniform"  // Uniform spacing
	InitPCA      InitMethod = "pca"      // PCA-based initialization
)

// DefaultKMeansConfig returns sensible defaults
func DefaultKMeansConfig(k int) KMeansConfig {
	return KMeansConfig{
		K:                   k,
		MaxIterations:       100,
		Tolerance:           1e-6,
		InitMethod:          InitKMeansPP,
		NumRestarts:         3,
		Seed:                time.Now().UnixNano(),
		MinPointsPerCluster: 1,
		ParallelWorkers:     4,
	}
}

// KMeansResult holds the results of K-means clustering
type KMeansResult struct {
	Centroids       [][]float32 `json:"centroids"`         // Final cluster centroids
	Assignments     []int       `json:"assignments"`       // Point to cluster assignments
	Inertia         float64     `json:"inertia"`           // Sum of squared distances to centroids
	Iterations      int         `json:"iterations"`        // Number of iterations until convergence
	Converged       bool        `json:"converged"`         // Whether algorithm converged
	ClusterSizes    []int       `json:"cluster_sizes"`     // Number of points per cluster
	WithinClusterSS []float64   `json:"within_cluster_ss"` // Within-cluster sum of squares
}

// KMeans implements K-means clustering algorithm
type KMeans struct {
	config KMeansConfig
	rng    *rand.Rand
	mu     sync.RWMutex
}

// NewKMeans creates a new K-means clusterer
func NewKMeans(config KMeansConfig) *KMeans {
	return &KMeans{
		config: config,
		rng:    rand.New(rand.NewSource(config.Seed)),
	}
}

// Fit performs K-means clustering on the given data
func (km *KMeans) Fit(ctx context.Context, data [][]float32) (*KMeansResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data provided")
	}

	if len(data) < km.config.K {
		return nil, fmt.Errorf("not enough data points (%d) for %d clusters", len(data), km.config.K)
	}

	dimension := len(data[0])
	for i, point := range data {
		if len(point) != dimension {
			return nil, fmt.Errorf("inconsistent dimension at point %d: expected %d, got %d", i, dimension, len(point))
		}
	}

	var bestResult *KMeansResult
	bestInertia := math.Inf(1)

	// Multiple restarts to find best clustering
	for restart := 0; restart < km.config.NumRestarts; restart++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := km.singleRun(ctx, data, dimension)
		if err != nil {
			continue // Try next restart
		}

		if result.Inertia < bestInertia {
			bestInertia = result.Inertia
			bestResult = result
		}
	}

	if bestResult == nil {
		return nil, fmt.Errorf("all K-means runs failed")
	}

	return bestResult, nil
}

// singleRun performs a single K-means run
func (km *KMeans) singleRun(ctx context.Context, data [][]float32, dimension int) (*KMeansResult, error) {
	// Initialize centroids
	centroids, err := km.initializeCentroids(data, dimension)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize centroids: %w", err)
	}

	assignments := make([]int, len(data))
	clusterSizes := make([]int, km.config.K)

	var prevInertia float64 = math.Inf(1)

	for iter := 0; iter < km.config.MaxIterations; iter++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Assignment step - assign points to nearest centroids
		inertia := km.assignPoints(data, centroids, assignments, clusterSizes)

		// Check for convergence
		improvement := prevInertia - inertia
		if improvement < km.config.Tolerance {
			return km.buildResult(centroids, assignments, inertia, iter+1, true, clusterSizes, data), nil
		}
		prevInertia = inertia

		// Update step - recalculate centroids
		newCentroids := km.updateCentroids(data, assignments, dimension, clusterSizes)

		// Check for empty clusters and reinitialize if needed
		if km.handleEmptyClusters(newCentroids, clusterSizes, data) {
			// Reset if we had to reinitialize
			prevInertia = math.Inf(1)
		}

		centroids = newCentroids

		// Reset cluster sizes for next iteration
		for i := range clusterSizes {
			clusterSizes[i] = 0
		}
	}

	// Did not converge within max iterations
	finalInertia := km.assignPoints(data, centroids, assignments, clusterSizes)
	return km.buildResult(centroids, assignments, finalInertia, km.config.MaxIterations, false, clusterSizes, data), nil
}

// initializeCentroids initializes cluster centroids using the specified method
func (km *KMeans) initializeCentroids(data [][]float32, dimension int) ([][]float32, error) {
	switch km.config.InitMethod {
	case InitRandom:
		return km.initRandom(data, dimension)
	case InitKMeansPP:
		return km.initKMeansPlusPlus(data, dimension)
	case InitUniform:
		return km.initUniform(data, dimension)
	case InitPCA:
		return km.initPCA(data, dimension)
	default:
		return km.initKMeansPlusPlus(data, dimension) // Default to k-means++
	}
}

// initRandom performs random initialization
func (km *KMeans) initRandom(data [][]float32, dimension int) ([][]float32, error) {
	centroids := make([][]float32, km.config.K)

	for i := 0; i < km.config.K; i++ {
		idx := km.rng.Intn(len(data))
		centroids[i] = make([]float32, dimension)
		copy(centroids[i], data[idx])
	}

	return centroids, nil
}

// initKMeansPlusPlus performs K-means++ initialization
func (km *KMeans) initKMeansPlusPlus(data [][]float32, dimension int) ([][]float32, error) {
	centroids := make([][]float32, km.config.K)

	// Choose first centroid randomly
	idx := km.rng.Intn(len(data))
	centroids[0] = make([]float32, dimension)
	copy(centroids[0], data[idx])

	// Choose remaining centroids with probability proportional to squared distance
	for k := 1; k < km.config.K; k++ {
		distances := make([]float64, len(data))
		totalDistance := 0.0

		// Calculate minimum squared distance to existing centroids
		for i, point := range data {
			minDist := math.Inf(1)
			for j := 0; j < k; j++ {
				dist := km.squaredDistance(point, centroids[j])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist
			totalDistance += minDist
		}

		// Choose next centroid with weighted probability
		target := km.rng.Float64() * totalDistance
		cumsum := 0.0

		for i, dist := range distances {
			cumsum += dist
			if cumsum >= target {
				centroids[k] = make([]float32, dimension)
				copy(centroids[k], data[i])
				break
			}
		}
	}

	return centroids, nil
}

// initUniform performs uniform initialization across data range
func (km *KMeans) initUniform(data [][]float32, dimension int) ([][]float32, error) {
	// Find min/max for each dimension
	mins := make([]float32, dimension)
	maxs := make([]float32, dimension)

	for d := 0; d < dimension; d++ {
		mins[d] = data[0][d]
		maxs[d] = data[0][d]
	}

	for _, point := range data {
		for d := 0; d < dimension; d++ {
			if point[d] < mins[d] {
				mins[d] = point[d]
			}
			if point[d] > maxs[d] {
				maxs[d] = point[d]
			}
		}
	}

	centroids := make([][]float32, km.config.K)
	for k := 0; k < km.config.K; k++ {
		centroids[k] = make([]float32, dimension)
		for d := 0; d < dimension; d++ {
			// Uniform distribution between min and max
			centroids[k][d] = mins[d] + km.rng.Float32()*(maxs[d]-mins[d])
		}
	}

	return centroids, nil
}

// initPCA performs PCA-based initialization (simplified version)
func (km *KMeans) initPCA(data [][]float32, dimension int) ([][]float32, error) {
	// For now, fall back to k-means++ (full PCA implementation would be complex)
	return km.initKMeansPlusPlus(data, dimension)
}

// assignPoints assigns each point to the nearest centroid
func (km *KMeans) assignPoints(data [][]float32, centroids [][]float32, assignments []int, clusterSizes []int) float64 {
	var totalInertia float64
	var mu sync.Mutex

	// Use worker pool for parallel assignment
	numWorkers := km.config.ParallelWorkers
	if numWorkers > len(data) {
		numWorkers = len(data)
	}

	chunkSize := len(data) / numWorkers
	if chunkSize == 0 {
		chunkSize = 1
	}

	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if w == numWorkers-1 {
			end = len(data) // Last worker takes remaining points
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			localInertia := 0.0
			localClusterSizes := make([]int, km.config.K)

			for i := start; i < end; i++ {
				minDist := math.Inf(1)
				bestCluster := 0

				for k, centroid := range centroids {
					dist := km.squaredDistance(data[i], centroid)
					if dist < minDist {
						minDist = dist
						bestCluster = k
					}
				}

				assignments[i] = bestCluster
				localInertia += minDist
				localClusterSizes[bestCluster]++
			}

			// Update global state
			mu.Lock()
			totalInertia += localInertia
			for k := 0; k < km.config.K; k++ {
				clusterSizes[k] += localClusterSizes[k]
			}
			mu.Unlock()
		}(start, end)
	}

	wg.Wait()
	return totalInertia
}

// updateCentroids recalculates centroids based on current assignments
func (km *KMeans) updateCentroids(data [][]float32, assignments []int, dimension int, clusterSizes []int) [][]float32 {
	centroids := make([][]float32, km.config.K)

	// Initialize centroids
	for k := 0; k < km.config.K; k++ {
		centroids[k] = make([]float32, dimension)
	}

	// Sum all points assigned to each cluster
	for i, point := range data {
		cluster := assignments[i]
		for d := 0; d < dimension; d++ {
			centroids[cluster][d] += point[d]
		}
	}

	// Divide by cluster size to get mean (centroid)
	for k := 0; k < km.config.K; k++ {
		if clusterSizes[k] > 0 {
			for d := 0; d < dimension; d++ {
				centroids[k][d] /= float32(clusterSizes[k])
			}
		}
	}

	return centroids
}

// handleEmptyClusters reinitializes empty clusters
func (km *KMeans) handleEmptyClusters(centroids [][]float32, clusterSizes []int, data [][]float32) bool {
	hasEmptyCluster := false

	for k := 0; k < km.config.K; k++ {
		if clusterSizes[k] < km.config.MinPointsPerCluster {
			// Reinitialize empty cluster to a random data point
			idx := km.rng.Intn(len(data))
			copy(centroids[k], data[idx])
			hasEmptyCluster = true
		}
	}

	return hasEmptyCluster
}

// squaredDistance computes squared Euclidean distance between two points
func (km *KMeans) squaredDistance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum
}

// buildResult constructs the final KMeansResult
func (km *KMeans) buildResult(centroids [][]float32, assignments []int, inertia float64, iterations int, converged bool, clusterSizes []int, data [][]float32) *KMeansResult {
	// Calculate within-cluster sum of squares
	withinClusterSS := make([]float64, km.config.K)

	for i, point := range data {
		cluster := assignments[i]
		dist := km.squaredDistance(point, centroids[cluster])
		withinClusterSS[cluster] += dist
	}

	return &KMeansResult{
		Centroids:       centroids,
		Assignments:     assignments,
		Inertia:         inertia,
		Iterations:      iterations,
		Converged:       converged,
		ClusterSizes:    clusterSizes,
		WithinClusterSS: withinClusterSS,
	}
}

// SilhouetteScore calculates the silhouette score for the clustering
func (km *KMeans) SilhouetteScore(data [][]float32, result *KMeansResult) float64 {
	if len(data) == 0 || len(result.Assignments) != len(data) {
		return 0
	}

	n := len(data)
	silhouettes := make([]float64, n)

	// Calculate silhouette score for each point
	for i := 0; i < n; i++ {
		a := km.averageIntraClusterDistance(i, data, result.Assignments)
		b := km.averageNearestClusterDistance(i, data, result.Assignments)

		if a == 0 && b == 0 {
			silhouettes[i] = 0
		} else {
			silhouettes[i] = (b - a) / math.Max(a, b)
		}
	}

	// Return average silhouette score
	var sum float64
	for _, s := range silhouettes {
		sum += s
	}

	return sum / float64(n)
}

// averageIntraClusterDistance calculates average distance within the same cluster
func (km *KMeans) averageIntraClusterDistance(pointIdx int, data [][]float32, assignments []int) float64 {
	cluster := assignments[pointIdx]
	var sum float64
	var count int

	for i, otherCluster := range assignments {
		if i != pointIdx && otherCluster == cluster {
			sum += math.Sqrt(km.squaredDistance(data[pointIdx], data[i]))
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return sum / float64(count)
}

// averageNearestClusterDistance calculates average distance to nearest cluster
func (km *KMeans) averageNearestClusterDistance(pointIdx int, data [][]float32, assignments []int) float64 {
	myCluster := assignments[pointIdx]

	// Find unique clusters
	clusterDistances := make(map[int][]float64)

	for i, cluster := range assignments {
		if i != pointIdx && cluster != myCluster {
			dist := math.Sqrt(km.squaredDistance(data[pointIdx], data[i]))
			clusterDistances[cluster] = append(clusterDistances[cluster], dist)
		}
	}

	if len(clusterDistances) == 0 {
		return 0
	}

	// Find minimum average distance to other clusters
	minAvgDist := math.Inf(1)

	for _, distances := range clusterDistances {
		if len(distances) > 0 {
			var sum float64
			for _, dist := range distances {
				sum += dist
			}
			avgDist := sum / float64(len(distances))

			if avgDist < minAvgDist {
				minAvgDist = avgDist
			}
		}
	}

	return minAvgDist
}

// ElbowMethod finds optimal K using the elbow method
func ElbowMethod(ctx context.Context, data [][]float32, maxK int) (int, []float64, error) {
	if maxK < 2 {
		return 0, nil, fmt.Errorf("maxK must be at least 2")
	}

	inertias := make([]float64, maxK-1)

	for k := 2; k <= maxK; k++ {
		select {
		case <-ctx.Done():
			return 0, nil, ctx.Err()
		default:
		}

		config := DefaultKMeansConfig(k)
		kmeans := NewKMeans(config)

		result, err := kmeans.Fit(ctx, data)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to fit k=%d: %w", k, err)
		}

		inertias[k-2] = result.Inertia
	}

	// Find elbow using rate of change
	optimalK := 2
	maxImprovement := 0.0

	for i := 1; i < len(inertias)-1; i++ {
		// Calculate improvement rate
		prev := inertias[i-1] - inertias[i]
		next := inertias[i] - inertias[i+1]
		improvement := prev - next

		if improvement > maxImprovement {
			maxImprovement = improvement
			optimalK = i + 2
		}
	}

	return optimalK, inertias, nil
}
