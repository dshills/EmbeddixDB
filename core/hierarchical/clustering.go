package hierarchical

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// KMeansClusterer implements balanced k-means clustering for hierarchical indexing
type KMeansClusterer struct {
	k              int
	maxIterations  int
	dimension      int
	distanceMetric core.DistanceMetric
	balanceFactor  float64 // Controls cluster size balance (0.0 = no balance, 1.0 = perfect balance)
	seed           int64
	mu             sync.RWMutex
}

// ClusterAssignment represents a vector's assignment to a cluster
type ClusterAssignment struct {
	VectorID  string
	ClusterID int
	Distance  float32
}

// Cluster represents a cluster with its centroid and members
type Cluster struct {
	ID       int
	Centroid []float32
	Members  []string
	Size     int
}

// ClusteringResult contains the results of k-means clustering
type ClusteringResult struct {
	Clusters     []Cluster
	Assignments  map[string]int
	Iterations   int
	Inertia      float64 // Total within-cluster sum of squares
	ElapsedTime  time.Duration
}

// NewKMeansClusterer creates a new balanced k-means clusterer
func NewKMeansClusterer(k int, dimension int, metric core.DistanceMetric, opts ...ClustererOption) *KMeansClusterer {
	kmc := &KMeansClusterer{
		k:              k,
		maxIterations:  100,
		dimension:      dimension,
		distanceMetric: metric,
		balanceFactor:  0.5,
		seed:           time.Now().UnixNano(),
	}

	// Apply options
	for _, opt := range opts {
		opt(kmc)
	}

	return kmc
}

// ClustererOption is a functional option for configuring KMeansClusterer
type ClustererOption func(*KMeansClusterer)

// WithMaxIterations sets the maximum number of iterations
func WithMaxIterations(maxIter int) ClustererOption {
	return func(kmc *KMeansClusterer) {
		kmc.maxIterations = maxIter
	}
}

// WithBalanceFactor sets the balance factor for cluster size control
func WithBalanceFactor(factor float64) ClustererOption {
	return func(kmc *KMeansClusterer) {
		kmc.balanceFactor = factor
	}
}

// WithSeed sets the random seed for reproducible clustering
func WithSeed(seed int64) ClustererOption {
	return func(kmc *KMeansClusterer) {
		kmc.seed = seed
	}
}

// Cluster performs balanced k-means clustering on the given vectors
func (kmc *KMeansClusterer) Cluster(vectors []core.Vector) (*ClusteringResult, error) {
	if len(vectors) < kmc.k {
		return nil, fmt.Errorf("number of vectors (%d) must be >= k (%d)", len(vectors), kmc.k)
	}

	startTime := time.Now()
	rng := rand.New(rand.NewSource(kmc.seed))

	// Initialize clusters with k-means++ method
	centroids, err := kmc.initializeCentroidsKMeansPlusPlus(vectors, rng)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize centroids: %w", err)
	}

	// Initialize cluster assignments
	assignments := make(map[string]int)
	clusters := make([]Cluster, kmc.k)
	for i := 0; i < kmc.k; i++ {
		clusters[i] = Cluster{
			ID:       i,
			Centroid: centroids[i],
			Members:  make([]string, 0),
		}
	}

	var inertia float64
	converged := false

	// Main k-means loop
	for iter := 0; iter < kmc.maxIterations && !converged; iter++ {
		// Assignment step with balancing
		newAssignments := kmc.assignVectorsBalanced(vectors, centroids, assignments)

		// Check for convergence
		if kmc.hasConverged(assignments, newAssignments) {
			converged = true
		}
		assignments = newAssignments

		// Update clusters
		for i := range clusters {
			clusters[i].Members = clusters[i].Members[:0] // Clear members
		}
		for vid, cid := range assignments {
			clusters[cid].Members = append(clusters[cid].Members, vid)
		}

		// Update step - compute new centroids
		for i := range centroids {
			if len(clusters[i].Members) > 0 {
				centroids[i] = kmc.computeCentroid(vectors, clusters[i].Members)
				clusters[i].Centroid = centroids[i]
			}
		}

		// Compute inertia (within-cluster sum of squares)
		inertia = kmc.computeInertia(vectors, assignments, centroids)
	}

	// Final cluster size update
	for i := range clusters {
		clusters[i].Size = len(clusters[i].Members)
	}

	result := &ClusteringResult{
		Clusters:    clusters,
		Assignments: assignments,
		Iterations:  kmc.maxIterations,
		Inertia:     inertia,
		ElapsedTime: time.Since(startTime),
	}

	return result, nil
}

// initializeCentroidsKMeansPlusPlus uses k-means++ initialization for better convergence
func (kmc *KMeansClusterer) initializeCentroidsKMeansPlusPlus(vectors []core.Vector, rng *rand.Rand) ([][]float32, error) {
	centroids := make([][]float32, kmc.k)
	
	// Create vector map for efficient lookup
	vectorMap := make(map[string][]float32)
	for _, v := range vectors {
		vectorMap[v.ID] = v.Values
	}

	// Choose first centroid randomly
	firstIdx := rng.Intn(len(vectors))
	centroids[0] = make([]float32, kmc.dimension)
	copy(centroids[0], vectors[firstIdx].Values)

	// Choose remaining centroids with probability proportional to squared distance
	for i := 1; i < kmc.k; i++ {
		distances := make([]float64, len(vectors))
		totalDist := 0.0

		// Compute distances to nearest centroid
		for j, v := range vectors {
			minDist := float64(math.MaxFloat32)
			for k := 0; k < i; k++ {
				dist := float64(kmc.computeDistance(v.Values, centroids[k]))
				if dist < minDist {
					minDist = dist
				}
			}
			distances[j] = minDist * minDist // Square the distance
			totalDist += distances[j]
		}

		// Choose next centroid with weighted probability
		r := rng.Float64() * totalDist
		cumSum := 0.0
		for j, d := range distances {
			cumSum += d
			if cumSum >= r {
				centroids[i] = make([]float32, kmc.dimension)
				copy(centroids[i], vectors[j].Values)
				break
			}
		}
	}

	return centroids, nil
}

// assignVectorsBalanced assigns vectors to clusters with size balancing
func (kmc *KMeansClusterer) assignVectorsBalanced(vectors []core.Vector, centroids [][]float32, currentAssignments map[string]int) map[string]int {
	assignments := make(map[string]int)
	
	// If no balancing, use standard assignment
	if kmc.balanceFactor == 0.0 {
		for _, v := range vectors {
			assignments[v.ID] = kmc.findNearestCentroid(v.Values, centroids)
		}
		return assignments
	}

	// Balanced assignment using the Hungarian algorithm approximation
	targetSize := len(vectors) / kmc.k
	clusterSizes := make([]int, kmc.k)
	
	// Sort vectors by distance to their nearest centroid
	type vectorDist struct {
		vector    core.Vector
		distances []float32
	}
	
	vdists := make([]vectorDist, len(vectors))
	for i, v := range vectors {
		vd := vectorDist{
			vector:    v,
			distances: make([]float32, kmc.k),
		}
		for j, centroid := range centroids {
			vd.distances[j] = kmc.computeDistance(v.Values, centroid)
		}
		vdists[i] = vd
	}

	// Assign vectors to clusters with capacity constraints
	assigned := make(map[string]bool)
	
	// First pass: assign vectors to their nearest available cluster
	for _, vd := range vdists {
		if assigned[vd.vector.ID] {
			continue
		}
		
		// Find nearest cluster with capacity
		indices := make([]int, kmc.k)
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool {
			return vd.distances[indices[i]] < vd.distances[indices[j]]
		})
		
		for _, clusterID := range indices {
			maxSize := targetSize + int(float64(targetSize)*kmc.balanceFactor)
			if clusterSizes[clusterID] < maxSize {
				assignments[vd.vector.ID] = clusterID
				clusterSizes[clusterID]++
				assigned[vd.vector.ID] = true
				break
			}
		}
	}

	// Second pass: assign any remaining vectors
	for _, vd := range vdists {
		if !assigned[vd.vector.ID] {
			// Find cluster with minimum size
			minSize := clusterSizes[0]
			minCluster := 0
			for i := 1; i < kmc.k; i++ {
				if clusterSizes[i] < minSize {
					minSize = clusterSizes[i]
					minCluster = i
				}
			}
			assignments[vd.vector.ID] = minCluster
			clusterSizes[minCluster]++
		}
	}

	return assignments
}

// findNearestCentroid finds the nearest centroid for a vector
func (kmc *KMeansClusterer) findNearestCentroid(vector []float32, centroids [][]float32) int {
	minDist := float32(math.MaxFloat32)
	nearestIdx := 0

	for i, centroid := range centroids {
		dist := kmc.computeDistance(vector, centroid)
		if dist < minDist {
			minDist = dist
			nearestIdx = i
		}
	}

	return nearestIdx
}

// computeDistance computes distance between two vectors
func (kmc *KMeansClusterer) computeDistance(v1, v2 []float32) float32 {
	dist, err := core.CalculateDistanceOptimized(v1, v2, kmc.distanceMetric)
	if err != nil {
		// Fall back to L2 distance on error
		dist, _ = core.EuclideanDistance(v1, v2)
	}
	return dist
}

// computeCentroid computes the centroid of a set of vectors
func (kmc *KMeansClusterer) computeCentroid(vectors []core.Vector, memberIDs []string) []float32 {
	centroid := make([]float32, kmc.dimension)
	count := 0

	// Create a map for efficient lookup
	memberMap := make(map[string]bool)
	for _, id := range memberIDs {
		memberMap[id] = true
	}

	// Sum all member vectors
	for _, v := range vectors {
		if memberMap[v.ID] {
			for j, val := range v.Values {
				centroid[j] += val
			}
			count++
		}
	}

	// Compute average
	if count > 0 {
		for j := range centroid {
			centroid[j] /= float32(count)
		}
	}

	return centroid
}

// hasConverged checks if the clustering has converged
func (kmc *KMeansClusterer) hasConverged(oldAssignments, newAssignments map[string]int) bool {
	if len(oldAssignments) != len(newAssignments) {
		return false
	}

	for id, oldCluster := range oldAssignments {
		if newCluster, exists := newAssignments[id]; !exists || oldCluster != newCluster {
			return false
		}
	}

	return true
}

// computeInertia computes the total within-cluster sum of squares
func (kmc *KMeansClusterer) computeInertia(vectors []core.Vector, assignments map[string]int, centroids [][]float32) float64 {
	inertia := 0.0

	for _, v := range vectors {
		if clusterID, exists := assignments[v.ID]; exists {
			dist := kmc.computeDistance(v.Values, centroids[clusterID])
			inertia += float64(dist * dist)
		}
	}

	return inertia
}

// GetClusterQuality computes cluster quality metrics
func (kmc *KMeansClusterer) GetClusterQuality(result *ClusteringResult, vectors []core.Vector) *ClusterQualityMetrics {
	metrics := &ClusterQualityMetrics{
		Clusters: make([]ClusterMetrics, len(result.Clusters)),
	}

	// Compute per-cluster metrics
	for i, cluster := range result.Clusters {
		cm := ClusterMetrics{
			ClusterID: cluster.ID,
			Size:      cluster.Size,
		}

		// Compute intra-cluster distance (cohesion)
		var intraSum float64
		count := 0
		for _, memberID := range cluster.Members {
			for _, v := range vectors {
				if v.ID == memberID {
					dist := kmc.computeDistance(v.Values, cluster.Centroid)
					intraSum += float64(dist)
					count++
					break
				}
			}
		}
		if count > 0 {
			cm.IntraClusterDistance = intraSum / float64(count)
		}

		// Compute inter-cluster distance (separation) to nearest cluster
		minInterDist := float64(math.MaxFloat32)
		for j, otherCluster := range result.Clusters {
			if i != j {
				dist := kmc.computeDistance(cluster.Centroid, otherCluster.Centroid)
				if float64(dist) < minInterDist {
					minInterDist = float64(dist)
				}
			}
		}
		cm.InterClusterDistance = minInterDist

		// Silhouette coefficient
		if cm.IntraClusterDistance > 0 {
			cm.Silhouette = (cm.InterClusterDistance - cm.IntraClusterDistance) / 
				math.Max(cm.InterClusterDistance, cm.IntraClusterDistance)
		}

		metrics.Clusters[i] = cm
	}

	// Compute overall metrics
	var totalSilhouette float64
	var maxImbalance float64
	avgSize := float64(len(vectors)) / float64(kmc.k)

	for _, cm := range metrics.Clusters {
		totalSilhouette += cm.Silhouette
		imbalance := math.Abs(float64(cm.Size)-avgSize) / avgSize
		if imbalance > maxImbalance {
			maxImbalance = imbalance
		}
	}

	metrics.OverallSilhouette = totalSilhouette / float64(len(metrics.Clusters))
	metrics.ClusterImbalance = maxImbalance
	metrics.Inertia = result.Inertia

	return metrics
}

// ClusterQualityMetrics contains cluster quality measurements
type ClusterQualityMetrics struct {
	Clusters          []ClusterMetrics
	OverallSilhouette float64
	ClusterImbalance  float64
	Inertia           float64
}

// ClusterMetrics contains metrics for a single cluster
type ClusterMetrics struct {
	ClusterID            int
	Size                 int
	IntraClusterDistance float64 // Average distance within cluster
	InterClusterDistance float64 // Distance to nearest cluster
	Silhouette           float64 // Silhouette coefficient
}