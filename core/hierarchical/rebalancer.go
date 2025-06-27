package hierarchical

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// ClusterRebalancer manages dynamic cluster rebalancing for hierarchical indexes
type ClusterRebalancer struct {
	mu                sync.RWMutex
	config            RebalancerConfig
	clusterStats      map[int]*ClusterStats
	rebalanceHistory  []RebalanceEvent
	lastRebalanceTime time.Time
}

// RebalancerConfig configures the cluster rebalancing behavior
type RebalancerConfig struct {
	// Rebalancing triggers
	SizeImbalanceThreshold  float64       // Max allowed size imbalance ratio
	QualityThreshold        float64       // Min acceptable search quality
	MinTimeBetweenRebalance time.Duration // Cooldown period between rebalances

	// Rebalancing strategy
	MaxVectorsToMove  int  // Max vectors to move in one rebalance
	TargetClusterSize int  // Ideal cluster size
	AllowSplitMerge   bool // Allow cluster splitting/merging

	// Performance tuning
	BackgroundRebalance bool // Run rebalancing in background
	RebalanceBatchSize  int  // Batch size for moving vectors
}

// ClusterStats tracks statistics for a single cluster
type ClusterStats struct {
	ClusterID        int
	Size             int
	LastUpdated      time.Time
	SearchLatencyP95 float64
	SearchCount      int64
	QualityScore     float64
	VectorIDs        map[string]bool
}

// RebalanceEvent records a rebalancing operation
type RebalanceEvent struct {
	Timestamp        time.Time
	Type             RebalanceType
	VectorsMoved     int
	ClustersAffected []int
	Duration         time.Duration
	Reason           string
}

// RebalanceType represents the type of rebalancing operation
type RebalanceType string

const (
	RebalanceTypeMove  RebalanceType = "move"
	RebalanceTypeSplit RebalanceType = "split"
	RebalanceTypeMerge RebalanceType = "merge"
)

// NewClusterRebalancer creates a new cluster rebalancer
func NewClusterRebalancer(config RebalancerConfig) *ClusterRebalancer {
	return &ClusterRebalancer{
		config:           config,
		clusterStats:     make(map[int]*ClusterStats),
		rebalanceHistory: make([]RebalanceEvent, 0, 100),
	}
}

// UpdateClusterStats updates statistics for a cluster
func (cr *ClusterRebalancer) UpdateClusterStats(clusterID int, size int, vectorIDs []string) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	stats, exists := cr.clusterStats[clusterID]
	if !exists {
		stats = &ClusterStats{
			ClusterID: clusterID,
			VectorIDs: make(map[string]bool),
		}
		cr.clusterStats[clusterID] = stats
	}

	stats.Size = size
	stats.LastUpdated = time.Now()

	// Update vector IDs
	stats.VectorIDs = make(map[string]bool)
	for _, id := range vectorIDs {
		stats.VectorIDs[id] = true
	}
}

// UpdateSearchMetrics updates search performance metrics for a cluster
func (cr *ClusterRebalancer) UpdateSearchMetrics(clusterID int, latency float64, quality float64) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	if stats, exists := cr.clusterStats[clusterID]; exists {
		stats.SearchCount++
		stats.SearchLatencyP95 = latency // Simplified - should use proper percentile tracking
		stats.QualityScore = quality
	}
}

// NeedsRebalancing checks if clusters need rebalancing
func (cr *ClusterRebalancer) NeedsRebalancing() (bool, string) {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	// Check cooldown period
	if time.Since(cr.lastRebalanceTime) < cr.config.MinTimeBetweenRebalance {
		return false, ""
	}

	// Check size imbalance
	if imbalance, reason := cr.checkSizeImbalance(); imbalance {
		return true, reason
	}

	// Check quality degradation
	if degraded, reason := cr.checkQualityDegradation(); degraded {
		return true, reason
	}

	return false, ""
}

// checkSizeImbalance checks for cluster size imbalances
func (cr *ClusterRebalancer) checkSizeImbalance() (bool, string) {
	if len(cr.clusterStats) == 0 {
		return false, ""
	}

	sizes := make([]int, 0, len(cr.clusterStats))
	totalSize := 0
	for _, stats := range cr.clusterStats {
		sizes = append(sizes, stats.Size)
		totalSize += stats.Size
	}

	if totalSize == 0 || len(sizes) == 0 {
		return false, ""
	}

	avgSize := float64(totalSize) / float64(len(sizes))
	maxImbalance := 0.0

	for _, size := range sizes {
		imbalance := math.Abs(float64(size)-avgSize) / avgSize
		if imbalance > maxImbalance {
			maxImbalance = imbalance
		}
	}

	if maxImbalance > cr.config.SizeImbalanceThreshold {
		return true, fmt.Sprintf("size imbalance %.2f exceeds threshold %.2f",
			maxImbalance, cr.config.SizeImbalanceThreshold)
	}

	return false, ""
}

// checkQualityDegradation checks for search quality issues
func (cr *ClusterRebalancer) checkQualityDegradation() (bool, string) {
	totalQuality := 0.0
	count := 0

	for _, stats := range cr.clusterStats {
		if stats.SearchCount > 0 {
			totalQuality += stats.QualityScore
			count++
		}
	}

	if count == 0 {
		return false, ""
	}

	avgQuality := totalQuality / float64(count)
	if avgQuality < cr.config.QualityThreshold {
		return true, fmt.Sprintf("average quality %.2f below threshold %.2f",
			avgQuality, cr.config.QualityThreshold)
	}

	return false, ""
}

// PlanRebalancing creates a rebalancing plan
func (cr *ClusterRebalancer) PlanRebalancing() *RebalancePlan {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	plan := &RebalancePlan{
		Timestamp: time.Now(),
		Moves:     make([]VectorMove, 0),
		Splits:    make([]ClusterSplit, 0),
		Merges:    make([]ClusterMerge, 0),
	}

	// Identify oversized and undersized clusters
	avgSize := cr.calculateAverageSize()
	oversized := make([]int, 0)
	undersized := make([]int, 0)

	for clusterID, stats := range cr.clusterStats {
		deviation := float64(stats.Size) - avgSize
		if deviation > avgSize*cr.config.SizeImbalanceThreshold {
			oversized = append(oversized, clusterID)
		} else if deviation < -avgSize*cr.config.SizeImbalanceThreshold {
			undersized = append(undersized, clusterID)
		}
	}

	// Plan moves from oversized to undersized clusters
	vectorsMoved := 0
	for _, fromCluster := range oversized {
		if vectorsMoved >= cr.config.MaxVectorsToMove {
			break
		}

		fromStats := cr.clusterStats[fromCluster]
		excessVectors := int(float64(fromStats.Size) - avgSize)

		for _, toCluster := range undersized {
			if vectorsMoved >= cr.config.MaxVectorsToMove {
				break
			}

			toStats := cr.clusterStats[toCluster]
			deficit := int(avgSize - float64(toStats.Size))

			// Move vectors
			moveCount := minInt(excessVectors, deficit)
			moveCount = minInt(moveCount, cr.config.MaxVectorsToMove-vectorsMoved)

			if moveCount > 0 {
				// Select vectors to move (simplified - in practice would use distance-based selection)
				vectorsToMove := cr.selectVectorsToMove(fromCluster, toCluster, moveCount)

				for _, vectorID := range vectorsToMove {
					plan.Moves = append(plan.Moves, VectorMove{
						VectorID:    vectorID,
						FromCluster: fromCluster,
						ToCluster:   toCluster,
					})
					vectorsMoved++
				}
			}
		}
	}

	// Plan splits and merges if enabled
	if cr.config.AllowSplitMerge {
		plan.Splits = cr.planClusterSplits(oversized)
		plan.Merges = cr.planClusterMerges(undersized)
	}

	plan.EstimatedImpact = cr.estimateRebalanceImpact(plan)
	return plan
}

// selectVectorsToMove selects vectors to move between clusters
func (cr *ClusterRebalancer) selectVectorsToMove(fromCluster, toCluster int, count int) []string {
	fromStats := cr.clusterStats[fromCluster]
	vectors := make([]string, 0, count)

	// Simple selection - take first N vectors
	// In practice, would select based on distance to target cluster centroid
	i := 0
	for vectorID := range fromStats.VectorIDs {
		if i >= count {
			break
		}
		vectors = append(vectors, vectorID)
		i++
	}

	return vectors
}

// planClusterSplits plans splitting of oversized clusters
func (cr *ClusterRebalancer) planClusterSplits(oversizedClusters []int) []ClusterSplit {
	splits := make([]ClusterSplit, 0)

	for _, clusterID := range oversizedClusters {
		stats := cr.clusterStats[clusterID]
		if stats.Size > cr.config.TargetClusterSize*2 {
			splits = append(splits, ClusterSplit{
				OriginalCluster: clusterID,
				NewClusterCount: 2,
				Method:          "balanced_kmeans",
			})
		}
	}

	return splits
}

// planClusterMerges plans merging of undersized clusters
func (cr *ClusterRebalancer) planClusterMerges(undersizedClusters []int) []ClusterMerge {
	merges := make([]ClusterMerge, 0)

	// Sort by size to merge smallest clusters first
	sort.Slice(undersizedClusters, func(i, j int) bool {
		return cr.clusterStats[undersizedClusters[i]].Size <
			cr.clusterStats[undersizedClusters[j]].Size
	})

	// Pair up clusters for merging
	for i := 0; i < len(undersizedClusters)-1; i += 2 {
		cluster1 := undersizedClusters[i]
		cluster2 := undersizedClusters[i+1]

		combinedSize := cr.clusterStats[cluster1].Size + cr.clusterStats[cluster2].Size
		if combinedSize <= cr.config.TargetClusterSize {
			merges = append(merges, ClusterMerge{
				Cluster1: cluster1,
				Cluster2: cluster2,
			})
		}
	}

	return merges
}

// ExecuteRebalancing executes a rebalancing plan
func (cr *ClusterRebalancer) ExecuteRebalancing(plan *RebalancePlan, index RebalancingIndex) error {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	startTime := time.Now()
	event := RebalanceEvent{
		Timestamp: startTime,
		Type:      RebalanceTypeMove,
		Reason:    plan.Reason,
	}

	// Execute vector moves in batches
	for i := 0; i < len(plan.Moves); i += cr.config.RebalanceBatchSize {
		end := minInt(i+cr.config.RebalanceBatchSize, len(plan.Moves))
		batch := plan.Moves[i:end]

		if err := cr.executeMovesBatch(batch, index); err != nil {
			return fmt.Errorf("failed to execute moves batch: %w", err)
		}

		event.VectorsMoved += len(batch)
	}

	// Execute splits
	for _, split := range plan.Splits {
		if err := cr.executeSplit(split, index); err != nil {
			return fmt.Errorf("failed to execute split: %w", err)
		}
		event.ClustersAffected = append(event.ClustersAffected, split.OriginalCluster)
	}

	// Execute merges
	for _, merge := range plan.Merges {
		if err := cr.executeMerge(merge, index); err != nil {
			return fmt.Errorf("failed to execute merge: %w", err)
		}
		event.ClustersAffected = append(event.ClustersAffected, merge.Cluster1, merge.Cluster2)
	}

	event.Duration = time.Since(startTime)
	cr.rebalanceHistory = append(cr.rebalanceHistory, event)
	cr.lastRebalanceTime = time.Now()

	// Keep only recent history
	if len(cr.rebalanceHistory) > 100 {
		cr.rebalanceHistory = cr.rebalanceHistory[len(cr.rebalanceHistory)-100:]
	}

	return nil
}

// executeMovesBatch executes a batch of vector moves
func (cr *ClusterRebalancer) executeMovesBatch(moves []VectorMove, index RebalancingIndex) error {
	for _, move := range moves {
		if err := index.MoveVector(move.VectorID, move.FromCluster, move.ToCluster); err != nil {
			return fmt.Errorf("failed to move vector %s: %w", move.VectorID, err)
		}

		// Update stats
		if fromStats, exists := cr.clusterStats[move.FromCluster]; exists {
			delete(fromStats.VectorIDs, move.VectorID)
			fromStats.Size--
		}
		if toStats, exists := cr.clusterStats[move.ToCluster]; exists {
			toStats.VectorIDs[move.VectorID] = true
			toStats.Size++
		}
	}
	return nil
}

// executeSplit executes a cluster split operation
func (cr *ClusterRebalancer) executeSplit(split ClusterSplit, index RebalancingIndex) error {
	return index.SplitCluster(split.OriginalCluster, split.NewClusterCount)
}

// executeMerge executes a cluster merge operation
func (cr *ClusterRebalancer) executeMerge(merge ClusterMerge, index RebalancingIndex) error {
	return index.MergeClusters(merge.Cluster1, merge.Cluster2)
}

// calculateAverageSize calculates the average cluster size
func (cr *ClusterRebalancer) calculateAverageSize() float64 {
	if len(cr.clusterStats) == 0 {
		return 0
	}

	total := 0
	for _, stats := range cr.clusterStats {
		total += stats.Size
	}

	return float64(total) / float64(len(cr.clusterStats))
}

// estimateRebalanceImpact estimates the impact of a rebalancing plan
func (cr *ClusterRebalancer) estimateRebalanceImpact(plan *RebalancePlan) RebalanceImpact {
	impact := RebalanceImpact{
		EstimatedDuration: time.Duration(len(plan.Moves)) * time.Millisecond * 10, // Rough estimate
		VectorsAffected:   len(plan.Moves),
		ClustersAffected:  make(map[int]bool),
	}

	for _, move := range plan.Moves {
		impact.ClustersAffected[move.FromCluster] = true
		impact.ClustersAffected[move.ToCluster] = true
	}

	impact.SearchImpact = "minimal" // Simplified - would calculate based on cluster usage
	if len(plan.Moves) > cr.config.MaxVectorsToMove/2 {
		impact.SearchImpact = "moderate"
	}

	return impact
}

// GetRebalanceHistory returns recent rebalancing events
func (cr *ClusterRebalancer) GetRebalanceHistory() []RebalanceEvent {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	history := make([]RebalanceEvent, len(cr.rebalanceHistory))
	copy(history, cr.rebalanceHistory)
	return history
}

// RebalancePlan represents a plan for rebalancing clusters
type RebalancePlan struct {
	Timestamp       time.Time
	Reason          string
	Moves           []VectorMove
	Splits          []ClusterSplit
	Merges          []ClusterMerge
	EstimatedImpact RebalanceImpact
}

// VectorMove represents moving a vector between clusters
type VectorMove struct {
	VectorID    string
	FromCluster int
	ToCluster   int
}

// ClusterSplit represents splitting a cluster
type ClusterSplit struct {
	OriginalCluster int
	NewClusterCount int
	Method          string
}

// ClusterMerge represents merging two clusters
type ClusterMerge struct {
	Cluster1 int
	Cluster2 int
}

// RebalanceImpact estimates the impact of rebalancing
type RebalanceImpact struct {
	EstimatedDuration time.Duration
	VectorsAffected   int
	ClustersAffected  map[int]bool
	SearchImpact      string // "minimal", "moderate", "significant"
}

// RebalancingIndex interface for indexes that support rebalancing operations
type RebalancingIndex interface {
	MoveVector(vectorID string, fromCluster, toCluster int) error
	SplitCluster(clusterID int, newCount int) error
	MergeClusters(cluster1, cluster2 int) error
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
