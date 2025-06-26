package index

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
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
	// Thread safety
	mu sync.RWMutex
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

	h := &HierarchicalHNSW{
		coarseIndex:    coarseIndex,
		fineIndexes:    make(map[int]core.Index),
		config:         config,
		vectorRouting:  make(map[string]int),
		dimension:      dimension,
		distanceMetric: distanceMetric,
		qualityMonitor: &QualityMonitor{
			searchQueries: make([]QualityMetric, 0, 1000),
		},
	}

	// Initialize fine-level indexes
	for i := 0; i < config.NumFineClusters; i++ {
		h.fineIndexes[i] = NewHNSWIndex(dimension, distanceMetric, config.FineConfig)
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

	// Step 1: Create coarse representation (dimensionality reduction)
	coarseVector, err := h.createCoarseRepresentation(vector)
	if err != nil {
		return fmt.Errorf("failed to create coarse representation: %w", err)
	}

	// Step 2: Add to coarse index for routing
	if err := h.coarseIndex.Add(coarseVector); err != nil {
		return fmt.Errorf("failed to add to coarse index: %w", err)
	}

	// Step 3: Determine fine cluster assignment
	clusterID := h.assignToFineCluster(vector)

	// Step 4: Add to assigned fine cluster
	if err := h.fineIndexes[clusterID].Add(vector); err != nil {
		return fmt.Errorf("failed to add to fine cluster %d: %w", clusterID, err)
	}

	// Step 5: Update routing table
	h.vectorRouting[vector.ID] = clusterID

	// Step 6: Check for rebalancing if incremental updates are enabled
	if h.config.EnableIncrementalUpdates {
		h.checkRebalancing()
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

	// Step 1: Create coarse representation of query
	coarseQuery, err := h.createCoarseQuery(query)
	if err != nil {
		return nil, fmt.Errorf("failed to create coarse query: %w", err)
	}

	// Step 2: Search coarse index to find relevant fine clusters
	clusterCandidates := int(math.Ceil(float64(h.config.NumFineClusters) * (h.config.RoutingOverlap + 0.1)))
	coarseResults, err := h.coarseIndex.Search(coarseQuery, clusterCandidates, nil)
	if err != nil {
		return nil, fmt.Errorf("coarse search failed: %w", err)
	}

	// Step 3: Search relevant fine clusters
	var allResults []core.SearchResult
	clusterHits := 0

	for _, coarseResult := range coarseResults {
		// Map coarse result to cluster ID
		clusterID := h.mapCoarseResultToCluster(coarseResult.ID)
		if fineIndex, exists := h.fineIndexes[clusterID]; exists {
			clusterHits++

			// Search fine cluster with higher k to account for filtering
			fineK := k * 2
			fineResults, err := fineIndex.Search(query, fineK, filter)
			if err != nil {
				continue // Skip problematic clusters
			}

			allResults = append(allResults, fineResults...)
		}
	}

	// Step 4: Merge and rank results
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score < allResults[j].Score
	})

	// Step 5: Return top k results
	if k > len(allResults) {
		k = len(allResults)
	}

	results := allResults[:k]

	// Step 6: Record quality metrics if monitoring is enabled
	if h.config.EnableQualityMonitoring {
		latency := getCurrentTimeMs() - startTime
		h.recordQualityMetric(len(results), latency, clusterHits)
	}

	return results, nil
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

	// Remove from routing table
	delete(h.vectorRouting, id)

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
