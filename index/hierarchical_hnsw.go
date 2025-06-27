package index

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/hierarchical"
)

// HierarchicalHNSW implements a two-level hierarchical HNSW index
// Coarse level for fast approximate routing, fine level for accurate search
type HierarchicalHNSW struct {
	// Coarse level index - lower dimensional representation for fast routing
	coarseIndex core.Index
	// Fine level indexes - multiple smaller HNSW indexes for detailed search
	fineIndexes map[int]core.Index
	// Configuration
	config HierarchicalConfig
	// Vector routing - maps vector IDs to fine index assignments
	vectorRouting map[string]int
	// Quality monitoring
	qualityMonitor *QualityMonitor
	// Full vector dimension
	dimension int
	// Distance metric
	distanceMetric core.DistanceMetric
	// Clustering components
	clusterer  *hierarchical.KMeansClusterer
	rebalancer *hierarchical.ClusterRebalancer
	optimizer  *hierarchical.BackgroundOptimizer
	// Cluster metadata
	clusterMetadata map[int]*ClusterMetadata
	// All vectors for clustering operations
	vectors map[string]core.Vector
	// Thread safety
	mu sync.RWMutex
}

// ClusterMetadata stores metadata about a cluster
type ClusterMetadata struct {
	ID             int
	Centroid       []float32
	CoarseCentroid []float32
	Size           int
	LastUpdated    time.Time
	QualityScore   float64
}

// HierarchicalConfig configures the hierarchical HNSW structure
type HierarchicalConfig struct {
	// Coarse level configuration
	CoarseDimension int        `json:"coarse_dimension"` // Reduced dimension for coarse level
	CoarseConfig    HNSWConfig `json:"coarse_config"`    // HNSW config for coarse level

	// Fine level configuration
	FineConfig           HNSWConfig `json:"fine_config"`             // HNSW config for fine levels
	NumFineClusters      int        `json:"num_fine_clusters"`       // Number of fine-level indexes
	MaxVectorsPerCluster int        `json:"max_vectors_per_cluster"` // Max vectors per fine index

	// Routing configuration
	RoutingOverlap  float64 `json:"routing_overlap"`  // Overlap between clusters (0.0-1.0)
	AdaptiveRouting bool    `json:"adaptive_routing"` // Enable adaptive cluster assignment

	// Quality monitoring
	EnableQualityMonitoring bool    `json:"enable_quality_monitoring"`
	QualityThreshold        float64 `json:"quality_threshold"` // Minimum acceptable recall@k

	// Incremental update configuration
	EnableIncrementalUpdates bool    `json:"enable_incremental_updates"`
	RebalanceThreshold       float64 `json:"rebalance_threshold"` // Cluster size imbalance threshold
}

// DefaultHierarchicalConfig returns sensible defaults for hierarchical HNSW
func DefaultHierarchicalConfig(dimension int) HierarchicalConfig {
	return HierarchicalConfig{
		CoarseDimension:          minInt(64, dimension/4), // Reduce to 1/4 dimension or 64, whichever is smaller
		CoarseConfig:             DefaultHNSWConfig(),
		FineConfig:               DefaultHNSWConfig(),
		NumFineClusters:          int(math.Sqrt(float64(dimension))), // sqrt(d) clusters
		MaxVectorsPerCluster:     10000,
		RoutingOverlap:           0.1, // 10% overlap
		AdaptiveRouting:          true,
		EnableQualityMonitoring:  true,
		QualityThreshold:         0.9, // 90% recall
		EnableIncrementalUpdates: true,
		RebalanceThreshold:       0.3, // 30% size imbalance triggers rebalance
	}
}

// QualityMonitor tracks search quality metrics for incremental updates
type QualityMonitor struct {
	mu                 sync.RWMutex
	searchQueries      []QualityMetric
	lastRebalance      int64
	qualityDegradation float64
}

// QualityMetric represents a single search quality measurement
type QualityMetric struct {
	Timestamp     int64   `json:"timestamp"`
	RecallAtK     float64 `json:"recall_at_k"`
	LatencyMs     float64 `json:"latency_ms"`
	ClusterHits   int     `json:"cluster_hits"`
	TotalClusters int     `json:"total_clusters"`
}

// NewHierarchicalHNSW creates a new hierarchical HNSW index
func NewHierarchicalHNSW(dimension int, distanceMetric core.DistanceMetric, config HierarchicalConfig) *HierarchicalHNSW {
	// Create coarse level index with reduced dimensions
	coarseIndex := NewHNSWIndex(config.CoarseDimension, distanceMetric, config.CoarseConfig)

	// Create clustering components
	clusterer := hierarchical.NewKMeansClusterer(
		config.NumFineClusters,
		dimension,
		distanceMetric,
		hierarchical.WithMaxIterations(50),
		hierarchical.WithBalanceFactor(0.5),
	)

	rebalancerConfig := hierarchical.RebalancerConfig{
		SizeImbalanceThreshold:  config.RebalanceThreshold,
		QualityThreshold:        config.QualityThreshold,
		MinTimeBetweenRebalance: time.Minute * 5,
		MaxVectorsToMove:        1000,
		TargetClusterSize:       config.MaxVectorsPerCluster,
		AllowSplitMerge:         true,
		BackgroundRebalance:     true,
		RebalanceBatchSize:      100,
	}
	rebalancer := hierarchical.NewClusterRebalancer(rebalancerConfig)

	optimizerConfig := hierarchical.OptimizerConfig{
		EnableOptimization:      config.EnableIncrementalUpdates,
		OptimizationInterval:    time.Minute * 10,
		MinTimeBetweenTasks:     time.Minute * 5,
		ClusterQualityWeight:    1.0,
		GraphConnectivityWeight: 0.8,
		MemoryEfficiencyWeight:  0.6,
		MaxCPUPercent:           20.0,
		MaxMemoryMB:             1024,
		MaxConcurrentTasks:      2,
	}
	optimizer := hierarchical.NewBackgroundOptimizer(optimizerConfig)

	h := &HierarchicalHNSW{
		coarseIndex:     coarseIndex,
		fineIndexes:     make(map[int]core.Index),
		config:          config,
		vectorRouting:   make(map[string]int),
		dimension:       dimension,
		distanceMetric:  distanceMetric,
		clusterer:       clusterer,
		rebalancer:      rebalancer,
		optimizer:       optimizer,
		clusterMetadata: make(map[int]*ClusterMetadata),
		vectors:         make(map[string]core.Vector),
		qualityMonitor: &QualityMonitor{
			searchQueries: make([]QualityMetric, 0, 1000),
		},
	}

	// Initialize fine-level indexes
	for i := 0; i < config.NumFineClusters; i++ {
		h.fineIndexes[i] = NewHNSWIndex(dimension, distanceMetric, config.FineConfig)
		h.clusterMetadata[i] = &ClusterMetadata{
			ID:          i,
			Centroid:    make([]float32, dimension),
			LastUpdated: time.Now(),
		}
	}

	// Start background optimizer if enabled
	if config.EnableIncrementalUpdates {
		optimizer.Start(h)
	}

	return h
}

// Add adds a vector to the hierarchical index
func (h *HierarchicalHNSW) Add(vector core.Vector) error {
	if err := core.ValidateVector(vector); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Store vector for clustering operations
	h.vectors[vector.ID] = vector

	// Step 1: Determine fine cluster assignment using proper clustering
	clusterID := h.assignToFineClusterWithClustering(vector)

	// Step 2: Create coarse representation based on cluster centroid
	coarseVector, err := h.createCoarseRepresentationFromCluster(vector, clusterID)
	if err != nil {
		return fmt.Errorf("failed to create coarse representation: %w", err)
	}

	// Step 3: Add to coarse index for routing
	if err := h.coarseIndex.Add(coarseVector); err != nil {
		return fmt.Errorf("failed to add to coarse index: %w", err)
	}

	// Step 4: Add to assigned fine cluster
	if err := h.fineIndexes[clusterID].Add(vector); err != nil {
		return fmt.Errorf("failed to add to fine cluster %d: %w", clusterID, err)
	}

	// Step 5: Update routing table and cluster metadata
	h.vectorRouting[vector.ID] = clusterID
	h.updateClusterMetadata(clusterID)

	// Step 6: Update rebalancer statistics
	vectorIDs := h.getClusterVectorIDs(clusterID)
	h.rebalancer.UpdateClusterStats(clusterID, len(vectorIDs), vectorIDs)

	// Step 7: Check for rebalancing if incremental updates are enabled
	if h.config.EnableIncrementalUpdates {
		if needsRebalance, reason := h.rebalancer.NeedsRebalancing(); needsRebalance {
			go h.performRebalancing(reason)
		}
	}

	return nil
}

// Search performs hierarchical k-nearest neighbor search
func (h *HierarchicalHNSW) Search(query []float32, k int, filter map[string]string) ([]core.SearchResult, error) {
	if len(query) != h.dimension {
		return nil, fmt.Errorf("query dimension %d does not match expected dimension %d", len(query), h.dimension)
	}

	startTime := getCurrentTimeMs()

	h.mu.RLock()
	defer h.mu.RUnlock()

	// Step 1: Find most relevant clusters using centroids
	relevantClusters := h.findRelevantClusters(query, k)

	// Step 2: Create coarse representation based on relevant clusters
	coarseQuery := h.createAdaptiveCoarseQuery(query, relevantClusters)

	// Step 3: Search coarse index for additional routing hints
	clusterCandidates := int(math.Ceil(float64(len(relevantClusters)) * (1.0 + h.config.RoutingOverlap)))
	coarseResults, err := h.coarseIndex.Search(coarseQuery, clusterCandidates, nil)
	if err != nil {
		// Fall back to direct cluster search
		coarseResults = nil
	}

	// Step 4: Combine relevant clusters from both methods
	searchClusters := make(map[int]float32)

	// Add clusters from centroid search with distances
	for _, rc := range relevantClusters {
		searchClusters[rc.clusterID] = rc.distance
	}

	// Add clusters from coarse search
	for _, coarseResult := range coarseResults {
		clusterID := h.mapCoarseResultToCluster(coarseResult.ID)
		if _, exists := searchClusters[clusterID]; !exists {
			searchClusters[clusterID] = coarseResult.Score
		}
	}

	// Step 5: Search relevant fine clusters with priority ordering
	allResults := h.searchClustersWithPriority(query, searchClusters, k, filter)

	// Step 6: Final reranking if needed
	// TODO: Add quantization support when available
	// if h.config.FineConfig.UseQuantization {
	// 	allResults = h.rerankWithFullPrecision(query, allResults, k*2)
	// }

	// Step 7: Return top k results
	if k > len(allResults) {
		k = len(allResults)
	}
	results := allResults[:k]

	// Step 8: Record quality metrics and update search statistics
	if h.config.EnableQualityMonitoring {
		latency := getCurrentTimeMs() - startTime
		h.recordQualityMetric(len(results), latency, len(searchClusters))

		// Update cluster search metrics for adaptive routing
		for clusterID := range searchClusters {
			h.rebalancer.UpdateSearchMetrics(clusterID, latency, float64(len(results))/float64(k))
		}
	}

	return results, nil
}

// findRelevantClusters finds clusters most relevant to the query
func (h *HierarchicalHNSW) findRelevantClusters(query []float32, k int) []struct {
	clusterID int
	distance  float32
} {
	type clusterDist struct {
		clusterID int
		distance  float32
	}

	clusters := make([]clusterDist, 0, h.config.NumFineClusters)

	// Compute distances to all cluster centroids
	for clusterID, metadata := range h.clusterMetadata {
		if len(metadata.Centroid) == len(query) {
			dist := h.computeDistance(query, metadata.Centroid)
			clusters = append(clusters, clusterDist{
				clusterID: clusterID,
				distance:  dist,
			})
		}
	}

	// Sort by distance
	sort.Slice(clusters, func(i, j int) bool {
		return clusters[i].distance < clusters[j].distance
	})

	// Select top clusters based on k and overlap
	numClusters := int(math.Ceil(math.Sqrt(float64(k)) * (1.0 + h.config.RoutingOverlap)))
	if numClusters > len(clusters) {
		numClusters = len(clusters)
	}

	result := make([]struct {
		clusterID int
		distance  float32
	}, numClusters)

	for i := 0; i < numClusters; i++ {
		result[i].clusterID = clusters[i].clusterID
		result[i].distance = clusters[i].distance
	}

	return result
}

// createAdaptiveCoarseQuery creates a coarse query adapted to relevant clusters
func (h *HierarchicalHNSW) createAdaptiveCoarseQuery(query []float32, relevantClusters []struct {
	clusterID int
	distance  float32
}) []float32 {
	if len(relevantClusters) == 0 {
		// Fall back to simple reduction
		coarse := make([]float32, h.config.CoarseDimension)
		copy(coarse, query[:h.config.CoarseDimension])
		return coarse
	}

	// Use weighted average of relevant cluster coarse centroids
	coarseQuery := make([]float32, h.config.CoarseDimension)
	totalWeight := float32(0)

	for _, rc := range relevantClusters {
		if metadata, exists := h.clusterMetadata[rc.clusterID]; exists && len(metadata.CoarseCentroid) > 0 {
			// Weight inversely by distance
			weight := 1.0 / (rc.distance + 0.1)
			for i, val := range metadata.CoarseCentroid {
				coarseQuery[i] += val * weight
			}
			totalWeight += weight
		}
	}

	// Normalize
	if totalWeight > 0 {
		for i := range coarseQuery {
			coarseQuery[i] /= totalWeight
		}
	} else {
		// Fall back to simple reduction
		copy(coarseQuery, query[:h.config.CoarseDimension])
	}

	return coarseQuery
}

// searchClustersWithPriority searches clusters in priority order
func (h *HierarchicalHNSW) searchClustersWithPriority(query []float32, clusters map[int]float32, k int, filter map[string]string) []core.SearchResult {
	// Sort clusters by distance/score
	type clusterPriority struct {
		id    int
		score float32
	}

	priorities := make([]clusterPriority, 0, len(clusters))
	for id, score := range clusters {
		priorities = append(priorities, clusterPriority{id: id, score: score})
	}

	sort.Slice(priorities, func(i, j int) bool {
		return priorities[i].score < priorities[j].score
	})

	// Search clusters in priority order
	var allResults []core.SearchResult
	searched := 0

	for _, cp := range priorities {
		if fineIndex, exists := h.fineIndexes[cp.id]; exists {
			// Adaptive k based on cluster priority
			searchK := k
			if searched > 0 {
				searchK = k / 2 // Search fewer in lower priority clusters
			}

			fineResults, err := fineIndex.Search(query, searchK, filter)
			if err != nil {
				continue
			}

			allResults = append(allResults, fineResults...)
			searched++

			// Early termination if we have enough good results
			if len(allResults) >= k*3 {
				break
			}
		}
	}

	// Sort all results by score
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score < allResults[j].Score
	})

	// Remove duplicates
	seen := make(map[string]bool)
	uniqueResults := make([]core.SearchResult, 0, len(allResults))
	for _, result := range allResults {
		if !seen[result.ID] {
			seen[result.ID] = true
			uniqueResults = append(uniqueResults, result)
		}
	}

	return uniqueResults
}

// rerankWithFullPrecision reranks results using full precision vectors
func (h *HierarchicalHNSW) rerankWithFullPrecision(query []float32, results []core.SearchResult, limit int) []core.SearchResult {
	if limit > len(results) {
		limit = len(results)
	}

	// Recompute distances with full precision
	for i := 0; i < limit; i++ {
		if v, exists := h.vectors[results[i].ID]; exists {
			results[i].Score = h.computeDistance(query, v.Values)
		}
	}

	// Re-sort by updated scores
	sort.Slice(results[:limit], func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	return results
}

// RangeSearch finds all vectors within a distance threshold
func (h *HierarchicalHNSW) RangeSearch(query []float32, radius float32, filter map[string]string, limit int) ([]core.SearchResult, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// For range search, we need to search all clusters since we don't know
	// which clusters might contain vectors within the radius
	var allResults []core.SearchResult

	for _, fineIndex := range h.fineIndexes {
		fineResults, err := fineIndex.RangeSearch(query, radius, filter, limit)
		if err != nil {
			continue // Skip problematic clusters
		}
		allResults = append(allResults, fineResults...)

		// Early termination if we have enough results
		if limit > 0 && len(allResults) >= limit {
			break
		}
	}

	// Sort by distance and apply limit
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score < allResults[j].Score
	})

	if limit > 0 && len(allResults) > limit {
		allResults = allResults[:limit]
	}

	return allResults, nil
}

// Delete removes a vector from the hierarchical index
func (h *HierarchicalHNSW) Delete(id string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Find which cluster the vector belongs to
	clusterID, exists := h.vectorRouting[id]
	if !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Delete from fine cluster
	if err := h.fineIndexes[clusterID].Delete(id); err != nil {
		return fmt.Errorf("failed to delete from fine cluster %d: %w", clusterID, err)
	}

	// Delete from coarse index
	if err := h.coarseIndex.Delete(id); err != nil {
		// Log but don't fail - coarse index inconsistency is recoverable
	}

	// Remove from routing table and vectors map
	delete(h.vectorRouting, id)
	delete(h.vectors, id)

	// Update cluster metadata
	h.updateClusterMetadata(clusterID)

	// Update rebalancer statistics
	vectorIDs := h.getClusterVectorIDs(clusterID)
	h.rebalancer.UpdateClusterStats(clusterID, len(vectorIDs), vectorIDs)

	return nil
}

// Helper methods

// createCoarseRepresentation reduces vector dimensionality for coarse index
func (h *HierarchicalHNSW) createCoarseRepresentation(vector core.Vector) (core.Vector, error) {
	// Simple dimensionality reduction: take first N dimensions
	// In production, could use PCA, random projection, or learned reduction
	coarseDim := h.config.CoarseDimension
	if len(vector.Values) < coarseDim {
		return core.Vector{}, fmt.Errorf("vector dimension %d smaller than coarse dimension %d",
			len(vector.Values), coarseDim)
	}

	coarseValues := make([]float32, coarseDim)
	copy(coarseValues, vector.Values[:coarseDim])

	return core.Vector{
		ID:       vector.ID,
		Values:   coarseValues,
		Metadata: vector.Metadata,
	}, nil
}

// createCoarseQuery reduces query dimensionality
func (h *HierarchicalHNSW) createCoarseQuery(query []float32) ([]float32, error) {
	coarseDim := h.config.CoarseDimension
	if len(query) < coarseDim {
		return nil, fmt.Errorf("query dimension %d smaller than coarse dimension %d",
			len(query), coarseDim)
	}

	coarseQuery := make([]float32, coarseDim)
	copy(coarseQuery, query[:coarseDim])
	return coarseQuery, nil
}

// assignToFineClusterWithClustering uses clustering to assign vectors
func (h *HierarchicalHNSW) assignToFineClusterWithClustering(vector core.Vector) int {
	// If we have enough vectors, use proper clustering
	if len(h.vectors) >= h.config.NumFineClusters*10 {
		// Find nearest cluster centroid
		minDist := float32(math.MaxFloat32)
		bestCluster := 0

		for clusterID, metadata := range h.clusterMetadata {
			if len(metadata.Centroid) == len(vector.Values) {
				dist := h.computeDistance(vector.Values, metadata.Centroid)
				if dist < minDist {
					minDist = dist
					bestCluster = clusterID
				}
			}
		}

		// Check cluster capacity
		if h.fineIndexes[bestCluster].Size() < h.config.MaxVectorsPerCluster {
			return bestCluster
		}

		// Find alternative cluster with capacity
		return h.findAlternativeCluster(vector, bestCluster)
	}

	// Fall back to adaptive assignment for initial vectors
	return h.adaptiveClusterAssignment(vector)
}

// assignToFineCluster determines which fine cluster a vector should be assigned to
func (h *HierarchicalHNSW) assignToFineCluster(vector core.Vector) int {
	if h.config.AdaptiveRouting {
		return h.adaptiveClusterAssignment(vector)
	}

	// Simple hash-based assignment
	hash := 0
	for i, v := range vector.Values {
		hash += int(v*1000) * (i + 1)
	}
	return absInt(hash) % h.config.NumFineClusters
}

// adaptiveClusterAssignment uses load balancing for cluster assignment
func (h *HierarchicalHNSW) adaptiveClusterAssignment(vector core.Vector) int {
	// Find cluster with minimum load
	minLoad := int(^uint(0) >> 1) // Max int
	bestCluster := 0

	for i := 0; i < h.config.NumFineClusters; i++ {
		load := h.fineIndexes[i].Size()
		if load < minLoad {
			minLoad = load
			bestCluster = i
		}
	}

	return bestCluster
}

// mapCoarseResultToCluster maps a coarse search result to a fine cluster ID
func (h *HierarchicalHNSW) mapCoarseResultToCluster(coarseID string) int {
	// In this implementation, coarse IDs map directly to cluster IDs
	// In practice, might need more sophisticated mapping
	if clusterID, exists := h.vectorRouting[coarseID]; exists {
		return clusterID
	}
	return 0 // Default cluster
}

// checkRebalancing checks if clusters need rebalancing
func (h *HierarchicalHNSW) checkRebalancing() {
	if !h.config.EnableIncrementalUpdates {
		return
	}

	// Calculate cluster size statistics
	sizes := make([]int, h.config.NumFineClusters)
	totalSize := 0

	for i := 0; i < h.config.NumFineClusters; i++ {
		sizes[i] = h.fineIndexes[i].Size()
		totalSize += sizes[i]
	}

	if totalSize == 0 {
		return
	}

	avgSize := float64(totalSize) / float64(h.config.NumFineClusters)

	// Check for significant imbalance
	maxImbalance := 0.0
	for _, size := range sizes {
		imbalance := math.Abs(float64(size)-avgSize) / avgSize
		if imbalance > maxImbalance {
			maxImbalance = imbalance
		}
	}

	// Trigger rebalancing if threshold exceeded
	if maxImbalance > h.config.RebalanceThreshold {
		h.rebalanceClusters()
	}
}

// rebalanceClusters redistributes vectors across clusters
func (h *HierarchicalHNSW) rebalanceClusters() {
	// Implementation would collect all vectors and redistribute them
	// For now, this is a placeholder for the rebalancing logic
}

// recordQualityMetric records search quality for monitoring
func (h *HierarchicalHNSW) recordQualityMetric(resultCount int, latencyMs float64, clusterHits int) {
	if !h.config.EnableQualityMonitoring {
		return
	}

	h.qualityMonitor.mu.Lock()
	defer h.qualityMonitor.mu.Unlock()

	metric := QualityMetric{
		Timestamp:     time.Now().UnixMilli(),
		RecallAtK:     float64(resultCount) / 10.0, // Rough recall estimate
		LatencyMs:     latencyMs,
		ClusterHits:   clusterHits,
		TotalClusters: h.config.NumFineClusters,
	}

	h.qualityMonitor.searchQueries = append(h.qualityMonitor.searchQueries, metric)

	// Keep only recent metrics (last 1000)
	if len(h.qualityMonitor.searchQueries) > 1000 {
		h.qualityMonitor.searchQueries = h.qualityMonitor.searchQueries[1:]
	}
}

// Utility functions

func (h *HierarchicalHNSW) getFullDimension() int {
	return h.dimension
}

func getCurrentTimeMs() float64 {
	return float64(time.Now().UnixMilli())
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// createCoarseRepresentationFromCluster creates coarse representation using cluster info
func (h *HierarchicalHNSW) createCoarseRepresentationFromCluster(vector core.Vector, clusterID int) (core.Vector, error) {
	// Use cluster centroid for coarse representation if available
	if metadata, exists := h.clusterMetadata[clusterID]; exists && len(metadata.CoarseCentroid) > 0 {
		return core.Vector{
			ID:       vector.ID,
			Values:   metadata.CoarseCentroid,
			Metadata: vector.Metadata,
		}, nil
	}

	// Fall back to simple dimensionality reduction
	return h.createCoarseRepresentation(vector)
}

// findAlternativeCluster finds an alternative cluster when preferred is full
func (h *HierarchicalHNSW) findAlternativeCluster(vector core.Vector, excludeCluster int) int {
	type clusterDist struct {
		id   int
		dist float32
	}

	candidates := make([]clusterDist, 0, h.config.NumFineClusters-1)

	for clusterID, metadata := range h.clusterMetadata {
		if clusterID == excludeCluster {
			continue
		}

		if h.fineIndexes[clusterID].Size() < h.config.MaxVectorsPerCluster {
			dist := h.computeDistance(vector.Values, metadata.Centroid)
			candidates = append(candidates, clusterDist{id: clusterID, dist: dist})
		}
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	if len(candidates) > 0 {
		return candidates[0].id
	}

	// All clusters full, return least loaded
	return h.adaptiveClusterAssignment(vector)
}

// updateClusterMetadata updates cluster centroid and metadata
func (h *HierarchicalHNSW) updateClusterMetadata(clusterID int) {
	metadata := h.clusterMetadata[clusterID]

	// Get all vectors in cluster
	clusterVectors := make([]core.Vector, 0)
	for vid, cid := range h.vectorRouting {
		if cid == clusterID {
			if v, exists := h.vectors[vid]; exists {
				clusterVectors = append(clusterVectors, v)
			}
		}
	}

	if len(clusterVectors) == 0 {
		return
	}

	// Compute new centroid
	centroid := make([]float32, h.dimension)
	for _, v := range clusterVectors {
		for i, val := range v.Values {
			centroid[i] += val
		}
	}
	for i := range centroid {
		centroid[i] /= float32(len(clusterVectors))
	}

	metadata.Centroid = centroid
	metadata.Size = len(clusterVectors)
	metadata.LastUpdated = time.Now()

	// Update coarse centroid
	coarseCentroid := make([]float32, h.config.CoarseDimension)
	copy(coarseCentroid, centroid[:h.config.CoarseDimension])
	metadata.CoarseCentroid = coarseCentroid
}

// getClusterVectorIDs returns all vector IDs in a cluster
func (h *HierarchicalHNSW) getClusterVectorIDs(clusterID int) []string {
	vectorIDs := make([]string, 0)
	for vid, cid := range h.vectorRouting {
		if cid == clusterID {
			vectorIDs = append(vectorIDs, vid)
		}
	}
	return vectorIDs
}

// performRebalancing performs cluster rebalancing
func (h *HierarchicalHNSW) performRebalancing(reason string) {
	// Create rebalancing plan
	plan := h.rebalancer.PlanRebalancing()
	plan.Reason = reason

	// Execute rebalancing
	if err := h.rebalancer.ExecuteRebalancing(plan, h); err != nil {
		// Log error but don't fail
		return
	}

	// Trigger re-clustering if significant changes
	if len(plan.Moves) > h.config.NumFineClusters*10 {
		go h.reclusterAll()
	}
}

// reclusterAll performs complete re-clustering
func (h *HierarchicalHNSW) reclusterAll() {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Collect all vectors
	allVectors := make([]core.Vector, 0, len(h.vectors))
	for _, v := range h.vectors {
		allVectors = append(allVectors, v)
	}

	if len(allVectors) < h.config.NumFineClusters {
		return
	}

	// Perform clustering
	result, err := h.clusterer.Cluster(allVectors)
	if err != nil {
		return
	}

	// Apply new assignments
	for vid, newClusterID := range result.Assignments {
		oldClusterID, exists := h.vectorRouting[vid]
		if !exists || oldClusterID == newClusterID {
			continue
		}

		// Move vector to new cluster
		if v, exists := h.vectors[vid]; exists {
			h.fineIndexes[oldClusterID].Delete(vid)
			h.fineIndexes[newClusterID].Add(v)
			h.vectorRouting[vid] = newClusterID
		}
	}

	// Update all cluster metadata
	for i := 0; i < h.config.NumFineClusters; i++ {
		h.updateClusterMetadata(i)
	}
}

// computeDistance computes distance between two vectors
func (h *HierarchicalHNSW) computeDistance(v1, v2 []float32) float32 {
	dist, err := core.CalculateDistanceOptimized(v1, v2, h.distanceMetric)
	if err != nil {
		// Fall back to L2 distance on error
		dist, _ = core.EuclideanDistance(v1, v2)
	}
	return dist
}

// Interface compliance for hierarchical.RebalancingIndex
func (h *HierarchicalHNSW) MoveVector(vectorID string, fromCluster, toCluster int) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	v, exists := h.vectors[vectorID]
	if !exists {
		return fmt.Errorf("vector %s not found", vectorID)
	}

	// Remove from old cluster
	if err := h.fineIndexes[fromCluster].Delete(vectorID); err != nil {
		return err
	}

	// Add to new cluster
	if err := h.fineIndexes[toCluster].Add(v); err != nil {
		return err
	}

	// Update routing
	h.vectorRouting[vectorID] = toCluster

	return nil
}

func (h *HierarchicalHNSW) SplitCluster(clusterID int, newCount int) error {
	// TODO: Implementation would split the cluster into newCount sub-clusters
	return fmt.Errorf("cluster splitting not yet implemented")
}

func (h *HierarchicalHNSW) MergeClusters(cluster1, cluster2 int) error {
	// TODO: Implementation would merge two clusters
	return fmt.Errorf("cluster merging not yet implemented")
}

// Interface compliance for hierarchical.OptimizableIndex
func (h *HierarchicalHNSW) GetClusterQuality() float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	totalQuality := 0.0
	count := 0

	for _, metadata := range h.clusterMetadata {
		if metadata.QualityScore > 0 {
			totalQuality += metadata.QualityScore
			count++
		}
	}

	if count == 0 {
		return 0.5 // Default medium quality
	}

	return totalQuality / float64(count)
}

func (h *HierarchicalHNSW) GetClusterAssignments() ([]core.Vector, map[string]int) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	vectors := make([]core.Vector, 0, len(h.vectors))
	for _, v := range h.vectors {
		vectors = append(vectors, v)
	}

	assignments := make(map[string]int)
	for k, v := range h.vectorRouting {
		assignments[k] = v
	}

	return vectors, assignments
}

func (h *HierarchicalHNSW) UpdateClusterAssignments(assignments map[string]int) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	for vid, newClusterID := range assignments {
		oldClusterID, exists := h.vectorRouting[vid]
		if !exists || oldClusterID == newClusterID {
			continue
		}

		if v, exists := h.vectors[vid]; exists {
			h.fineIndexes[oldClusterID].Delete(vid)
			h.fineIndexes[newClusterID].Add(v)
			h.vectorRouting[vid] = newClusterID
		}
	}

	return nil
}

func (h *HierarchicalHNSW) GetClusters() []hierarchical.Cluster {
	h.mu.RLock()
	defer h.mu.RUnlock()

	clusters := make([]hierarchical.Cluster, 0, h.config.NumFineClusters)

	for i := 0; i < h.config.NumFineClusters; i++ {
		members := make([]string, 0)
		for vid, cid := range h.vectorRouting {
			if cid == i {
				members = append(members, vid)
			}
		}

		cluster := hierarchical.Cluster{
			ID:       i,
			Centroid: h.clusterMetadata[i].Centroid,
			Members:  members,
			Size:     len(members),
		}
		clusters = append(clusters, cluster)
	}

	return clusters
}

func (h *HierarchicalHNSW) ComputeCentroid(memberIDs []string) []float32 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	centroid := make([]float32, h.dimension)
	count := 0

	for _, id := range memberIDs {
		if v, exists := h.vectors[id]; exists {
			for i, val := range v.Values {
				centroid[i] += val
			}
			count++
		}
	}

	if count > 0 {
		for i := range centroid {
			centroid[i] /= float32(count)
		}
	}

	return centroid
}

func (h *HierarchicalHNSW) UpdateCentroid(clusterID int, centroid []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if metadata, exists := h.clusterMetadata[clusterID]; exists {
		metadata.Centroid = centroid
		metadata.LastUpdated = time.Now()
	}

	return nil
}

func (h *HierarchicalHNSW) GetGraphConnectivity() float64 {
	// Simplified - would check actual graph connectivity
	return 0.9
}

func (h *HierarchicalHNSW) FindDisconnectedComponents() [][]string {
	// TODO: Implementation would find actual disconnected components
	return [][]string{} // Return empty slice instead of nil
}

func (h *HierarchicalHNSW) RepairComponent(component []string) error {
	// TODO: Implementation would repair disconnected components
	return fmt.Errorf("component repair not yet implemented")
}

func (h *HierarchicalHNSW) PruneRedundantEdges() (int, error) {
	// TODO: Implementation would prune redundant edges
	return 0, fmt.Errorf("edge pruning not yet implemented")
}

func (h *HierarchicalHNSW) GetMemoryEfficiency() float64 {
	// Simplified - would calculate actual memory efficiency
	return 0.8
}

func (h *HierarchicalHNSW) GetMemoryUsage() int64 {
	// Simplified - would calculate actual memory usage
	return int64(len(h.vectors) * h.dimension * 4)
}

func (h *HierarchicalHNSW) CompactMemory() error {
	// Simplified - would compact memory usage
	return nil
}

func (h *HierarchicalHNSW) GetDistanceMetric() core.DistanceMetric {
	return h.distanceMetric
}

// Interface compliance
func (h *HierarchicalHNSW) Rebuild() error { return nil }
func (h *HierarchicalHNSW) Size() int {
	total := 0
	for _, idx := range h.fineIndexes {
		total += idx.Size()
	}
	return total
}
func (h *HierarchicalHNSW) Type() string                  { return "hierarchical_hnsw" }
func (h *HierarchicalHNSW) Serialize() ([]byte, error)    { return nil, nil }
func (h *HierarchicalHNSW) Deserialize(data []byte) error { return nil }
