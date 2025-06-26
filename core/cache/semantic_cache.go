package cache

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// SemanticCacheImpl implements semantic similarity-based caching
type SemanticCacheImpl struct {
	*BaseCache
	clusters       []*SemanticCluster
	embeddings     map[string]Vector
	clusterCount   int
	updateInterval time.Duration
	minClusterSize int
	similarityFunc func(a, b []float32) float32
	mu             sync.RWMutex
	stopUpdate     chan struct{}
	clusteringAlgo ClusteringAlgorithm
}

// ClusteringAlgorithm defines the interface for clustering algorithms
type ClusteringAlgorithm interface {
	Cluster(embeddings map[string][]float32, k int) []*SemanticCluster
}

// KMeansClusteringAlgorithm implements k-means clustering
type KMeansClusteringAlgorithm struct {
	maxIterations int
	tolerance     float32
}

// NewKMeansClusteringAlgorithm creates a new k-means clustering algorithm
func NewKMeansClusteringAlgorithm() *KMeansClusteringAlgorithm {
	return &KMeansClusteringAlgorithm{
		maxIterations: 100,
		tolerance:     0.001,
	}
}

// Cluster performs k-means clustering on embeddings
func (k *KMeansClusteringAlgorithm) Cluster(embeddings map[string][]float32, numClusters int) []*SemanticCluster {
	if len(embeddings) < numClusters {
		numClusters = len(embeddings)
	}

	// Convert map to slices for easier processing
	keys := make([]string, 0, len(embeddings))
	vectors := make([][]float32, 0, len(embeddings))

	for key, vec := range embeddings {
		keys = append(keys, key)
		vectors = append(vectors, vec)
	}

	if len(vectors) == 0 {
		return nil
	}

	// Initialize centroids randomly
	centroids := k.initializeCentroids(vectors, numClusters)
	clusters := make([]*SemanticCluster, numClusters)

	// Initialize clusters
	for i := range clusters {
		clusters[i] = &SemanticCluster{
			ID:          fmt.Sprintf("cluster_%d", i),
			Centroid:    centroids[i],
			Members:     make([]string, 0),
			CreatedAt:   time.Now(),
			LastUpdated: time.Now(),
		}
	}

	// K-means iterations
	for iter := 0; iter < k.maxIterations; iter++ {
		// Clear cluster members
		for _, cluster := range clusters {
			cluster.Members = cluster.Members[:0]
		}

		// Assign points to nearest centroid
		for i, vec := range vectors {
			nearestIdx := k.findNearestCentroid(vec, centroids)
			clusters[nearestIdx].Members = append(clusters[nearestIdx].Members, keys[i])
		}

		// Update centroids
		converged := true
		for i, cluster := range clusters {
			if len(cluster.Members) == 0 {
				continue
			}

			newCentroid := k.calculateCentroid(cluster.Members, embeddings)

			// Check convergence
			if k.distance(centroids[i], newCentroid) > k.tolerance {
				converged = false
			}

			centroids[i] = newCentroid
			cluster.Centroid = newCentroid
		}

		if converged {
			break
		}
	}

	// Calculate radius for each cluster
	for _, cluster := range clusters {
		cluster.Radius = k.calculateRadius(cluster, embeddings)
	}

	return clusters
}

// initializeCentroids initializes centroids using k-means++
func (k *KMeansClusteringAlgorithm) initializeCentroids(vectors [][]float32, numClusters int) [][]float32 {
	centroids := make([][]float32, 0, numClusters)

	// Choose first centroid randomly
	firstIdx := 0 // In production, use proper random selection
	centroids = append(centroids, copyVector(vectors[firstIdx]))

	// Choose remaining centroids using k-means++
	for len(centroids) < numClusters {
		distances := make([]float32, len(vectors))

		// Calculate distance to nearest centroid for each point
		for i, vec := range vectors {
			minDist := float32(math.MaxFloat32)
			for _, centroid := range centroids {
				dist := k.distance(vec, centroid)
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist * minDist // Square for probability
		}

		// Select next centroid with probability proportional to squared distance
		nextIdx := k.selectWeighted(distances)
		centroids = append(centroids, copyVector(vectors[nextIdx]))
	}

	return centroids
}

// findNearestCentroid finds the nearest centroid for a vector
func (k *KMeansClusteringAlgorithm) findNearestCentroid(vec []float32, centroids [][]float32) int {
	minDist := float32(math.MaxFloat32)
	nearestIdx := 0

	for i, centroid := range centroids {
		dist := k.distance(vec, centroid)
		if dist < minDist {
			minDist = dist
			nearestIdx = i
		}
	}

	return nearestIdx
}

// calculateCentroid calculates the mean of cluster members
func (k *KMeansClusteringAlgorithm) calculateCentroid(members []string, embeddings map[string][]float32) []float32 {
	if len(members) == 0 {
		return nil
	}

	// Get dimension from first member
	dim := len(embeddings[members[0]])
	centroid := make([]float32, dim)

	// Sum all vectors
	for _, member := range members {
		vec := embeddings[member]
		for i := range centroid {
			centroid[i] += vec[i]
		}
	}

	// Calculate mean
	count := float32(len(members))
	for i := range centroid {
		centroid[i] /= count
	}

	return centroid
}

// calculateRadius calculates the radius of a cluster
func (k *KMeansClusteringAlgorithm) calculateRadius(cluster *SemanticCluster, embeddings map[string][]float32) float32 {
	if len(cluster.Members) == 0 {
		return 0
	}

	maxDist := float32(0)
	for _, member := range cluster.Members {
		vec := embeddings[member]
		dist := k.distance(vec, cluster.Centroid)
		if dist > maxDist {
			maxDist = dist
		}
	}

	return maxDist * 1.2 // Add 20% margin
}

// distance calculates L2 distance between vectors
func (k *KMeansClusteringAlgorithm) distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// selectWeighted selects an index with probability proportional to weights
func (k *KMeansClusteringAlgorithm) selectWeighted(weights []float32) int {
	// Simple weighted selection
	// In production, use proper random selection
	maxWeight := float32(0)
	maxIdx := 0

	for i, w := range weights {
		if w > maxWeight {
			maxWeight = w
			maxIdx = i
		}
	}

	return maxIdx
}

// copyVector creates a copy of a vector
func copyVector(vec []float32) []float32 {
	result := make([]float32, len(vec))
	copy(result, vec)
	return result
}

// NewSemanticCache creates a new semantic cache
func NewSemanticCache(capacity, maxMemory int64, config CacheConfig, clusterCount int) *SemanticCacheImpl {
	baseCache := NewBaseCache(capacity, maxMemory, config.CleanupInterval)

	sc := &SemanticCacheImpl{
		BaseCache:      baseCache,
		embeddings:     make(map[string]Vector),
		clusterCount:   clusterCount,
		updateInterval: 10 * time.Minute,
		minClusterSize: 10,
		similarityFunc: cosineDistance,
		stopUpdate:     make(chan struct{}),
		clusteringAlgo: NewKMeansClusteringAlgorithm(),
	}

	// Start clustering update routine
	go sc.clusterUpdateLoop()

	return sc
}

// GetSimilar finds cached results similar to the query
func (sc *SemanticCacheImpl) GetSimilar(ctx context.Context, query Vector, threshold float64) ([]CachedResult, error) {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if len(sc.clusters) == 0 {
		return nil, nil
	}

	queryVec := query.Values()
	var results []CachedResult

	// Find clusters that might contain similar queries
	for _, cluster := range sc.clusters {
		if cluster.Contains(queryVec, sc.similarityFunc) {
			// Check all members of the cluster
			for _, memberKey := range cluster.Members {
				memberVec, exists := sc.embeddings[memberKey]
				if !exists {
					continue
				}

				// Calculate similarity
				distance := sc.similarityFunc(queryVec, memberVec.Values())
				similarity := 1.0 - distance

				if similarity >= float32(threshold) {
					// Retrieve cached result
					value, found := sc.Get(ctx, memberKey)
					if found {
						if cachedResult, ok := value.(*CachedQueryResult); ok {
							results = append(results, CachedResult{
								QueryHash:    memberKey,
								Results:      []interface{}{cachedResult},
								Timestamp:    cachedResult.Timestamp,
								AccessCount:  cachedResult.AccessCount,
								UserContext:  cachedResult.UserContext,
								Confidence:   float64(similarity),
								SemanticHash: cluster.ID,
							})
						}
					}
				}
			}

			// Update cluster statistics
			cluster.HitCount++
		}
	}

	// Sort by confidence
	sort.Slice(results, func(i, j int) bool {
		return results[i].Confidence > results[j].Confidence
	})

	return results, nil
}

// AddWithEmbedding adds a result with its query embedding
func (sc *SemanticCacheImpl) AddWithEmbedding(ctx context.Context, key string, embedding Vector, result CachedResult) error {
	// Store the embedding
	sc.mu.Lock()
	sc.embeddings[key] = embedding
	sc.mu.Unlock()

	// Store the result in base cache
	return sc.Set(ctx, key, result)
}

// UpdateClusters updates the semantic clustering
func (sc *SemanticCacheImpl) UpdateClusters(ctx context.Context) error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Convert embeddings to format for clustering
	embeddingMap := make(map[string][]float32)
	for key, vec := range sc.embeddings {
		embeddingMap[key] = vec.Values()
	}

	// Perform clustering
	newClusters := sc.clusteringAlgo.Cluster(embeddingMap, sc.clusterCount)

	// Update clusters
	sc.clusters = newClusters

	return nil
}

// GetClusterStats returns clustering statistics
func (sc *SemanticCacheImpl) GetClusterStats() ClusterStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	if len(sc.clusters) == 0 {
		return ClusterStats{}
	}

	totalSize := 0
	minSize := int(^uint(0) >> 1) // Max int
	maxSize := 0
	totalCompactness := float64(0)

	for _, cluster := range sc.clusters {
		size := len(cluster.Members)
		totalSize += size

		if size < minSize {
			minSize = size
		}
		if size > maxSize {
			maxSize = size
		}

		// Calculate compactness (average distance to centroid)
		if size > 0 {
			avgDist := float64(0)
			for _, member := range cluster.Members {
				if vec, exists := sc.embeddings[member]; exists {
					dist := sc.similarityFunc(vec.Values(), cluster.Centroid)
					avgDist += float64(dist)
				}
			}
			avgDist /= float64(size)
			totalCompactness += avgDist
		}
	}

	avgSize := float64(totalSize) / float64(len(sc.clusters))
	avgCompactness := totalCompactness / float64(len(sc.clusters))

	return ClusterStats{
		ClusterCount:    len(sc.clusters),
		AverageSize:     avgSize,
		MinSize:         minSize,
		MaxSize:         maxSize,
		Compactness:     avgCompactness,
		SeparationScore: sc.calculateSeparation(),
	}
}

// calculateSeparation calculates the average separation between clusters
func (sc *SemanticCacheImpl) calculateSeparation() float64 {
	if len(sc.clusters) < 2 {
		return 0
	}

	totalSep := float64(0)
	count := 0

	for i := 0; i < len(sc.clusters)-1; i++ {
		for j := i + 1; j < len(sc.clusters); j++ {
			dist := sc.similarityFunc(sc.clusters[i].Centroid, sc.clusters[j].Centroid)
			totalSep += float64(dist)
			count++
		}
	}

	if count > 0 {
		return totalSep / float64(count)
	}

	return 0
}

// clusterUpdateLoop periodically updates clusters
func (sc *SemanticCacheImpl) clusterUpdateLoop() {
	ticker := time.NewTicker(sc.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ctx := context.Background()
			sc.UpdateClusters(ctx)
		case <-sc.stopUpdate:
			return
		}
	}
}

// Close stops the semantic cache
func (sc *SemanticCacheImpl) Close() {
	close(sc.stopUpdate)
	sc.BaseCache.Close()
}

// cosineDistance calculates cosine distance between vectors
func cosineDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 1.0
	}

	var dotProduct, normA, normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))

	// Convert similarity to distance
	return 1.0 - similarity
}
