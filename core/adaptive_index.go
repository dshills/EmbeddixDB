package core

import (
	"fmt"
	"sync"
	"time"
)

// IndexType represents different index implementations
type IndexType string

const (
	IndexTypeFlat IndexType = "flat"
	IndexTypeHNSW IndexType = "hnsw"
	IndexTypeIVF  IndexType = "ivf"
)

// QueryStats tracks performance metrics for adaptive decisions
type QueryStats struct {
	TotalQueries    int64
	AverageLatency  time.Duration
	P95Latency      time.Duration
	ThroughputQPS   float64
	LastUpdated     time.Time
	RecentLatencies []time.Duration
	mutex           sync.RWMutex
}

// NewQueryStats creates a new query statistics tracker
func NewQueryStats() *QueryStats {
	return &QueryStats{
		RecentLatencies: make([]time.Duration, 0, 100), // Keep last 100 queries
		LastUpdated:     time.Now(),
	}
}

// RecordQuery records a query execution time
func (qs *QueryStats) RecordQuery(latency time.Duration) {
	qs.mutex.Lock()
	defer qs.mutex.Unlock()

	qs.TotalQueries++
	qs.RecentLatencies = append(qs.RecentLatencies, latency)

	// Keep only last 100 queries
	if len(qs.RecentLatencies) > 100 {
		qs.RecentLatencies = qs.RecentLatencies[1:]
	}

	// Update statistics
	qs.updateStats()
}

func (qs *QueryStats) updateStats() {
	if len(qs.RecentLatencies) == 0 {
		return
	}

	// Calculate average
	var total time.Duration
	for _, lat := range qs.RecentLatencies {
		total += lat
	}
	qs.AverageLatency = total / time.Duration(len(qs.RecentLatencies))

	// Calculate P95
	if len(qs.RecentLatencies) >= 5 {
		p95Index := int(float64(len(qs.RecentLatencies)) * 0.95)
		qs.P95Latency = qs.RecentLatencies[p95Index]
	}

	// Calculate throughput
	now := time.Now()
	duration := now.Sub(qs.LastUpdated)
	if duration > 0 {
		qs.ThroughputQPS = float64(len(qs.RecentLatencies)) / duration.Seconds()
	}
	qs.LastUpdated = now
}

// GetStats returns a copy of current statistics
func (qs *QueryStats) GetStats() QueryStats {
	qs.mutex.RLock()
	defer qs.mutex.RUnlock()

	return QueryStats{
		TotalQueries:    qs.TotalQueries,
		AverageLatency:  qs.AverageLatency,
		P95Latency:      qs.P95Latency,
		ThroughputQPS:   qs.ThroughputQPS,
		LastUpdated:     qs.LastUpdated,
		RecentLatencies: append([]time.Duration(nil), qs.RecentLatencies...),
	}
}

// IndexPerformanceProfile defines performance characteristics of an index type
type IndexPerformanceProfile struct {
	IndexType          IndexType
	OptimalSizeMin     int     // Minimum collection size for optimal performance
	OptimalSizeMax     int     // Maximum collection size for optimal performance
	QueryLatencyScore  float64 // Lower is better
	InsertLatencyScore float64 // Lower is better
	MemoryEfficiency   float64 // Higher is better
	BuildTimeScore     float64 // Lower is better
	AccuracyScore      float64 // Higher is better (for approximate indices)
}

// DefaultIndexProfiles provides default performance profiles
var DefaultIndexProfiles = []IndexPerformanceProfile{
	{
		IndexType:          IndexTypeFlat,
		OptimalSizeMin:     0,
		OptimalSizeMax:     10000,
		QueryLatencyScore:  1.0, // Linear search
		InsertLatencyScore: 0.1, // Very fast inserts
		MemoryEfficiency:   1.0, // Just stores vectors
		BuildTimeScore:     0.1, // No build time
		AccuracyScore:      1.0, // Exact search
	},
	{
		IndexType:          IndexTypeHNSW,
		OptimalSizeMin:     1000,
		OptimalSizeMax:     10000000,
		QueryLatencyScore:  0.2,  // Very fast approximate search
		InsertLatencyScore: 0.7,  // Moderate insert cost
		MemoryEfficiency:   0.6,  // Graph overhead
		BuildTimeScore:     0.8,  // Expensive to build
		AccuracyScore:      0.95, // High accuracy
	},
	{
		IndexType:          IndexTypeIVF,
		OptimalSizeMin:     100000,
		OptimalSizeMax:     100000000,
		QueryLatencyScore:  0.4,  // Good for large scale
		InsertLatencyScore: 0.5,  // Moderate insert cost
		MemoryEfficiency:   0.8,  // Efficient for large datasets
		BuildTimeScore:     0.6,  // Moderate build time
		AccuracyScore:      0.85, // Good accuracy with proper tuning
	},
}

// AdaptiveIndex automatically selects and switches between index types
type AdaptiveIndex struct {
	currentIndex    Index
	currentType     IndexType
	stats           *QueryStats
	collectionSize  int
	dimension       int
	distanceMetric  DistanceMetric
	profiles        []IndexPerformanceProfile
	switchThreshold time.Duration // Min time between switches
	lastSwitch      time.Time
	mutex           sync.RWMutex
	factory         IndexFactory
}

// NewAdaptiveIndex creates a new adaptive index
func NewAdaptiveIndex(dimension int, distanceMetric DistanceMetric, factory IndexFactory) *AdaptiveIndex {
	ai := &AdaptiveIndex{
		stats:           NewQueryStats(),
		dimension:       dimension,
		distanceMetric:  distanceMetric,
		profiles:        DefaultIndexProfiles,
		switchThreshold: 5 * time.Minute, // Don't switch more than every 5 minutes
		lastSwitch:      time.Now(),
		factory:         factory,
	}

	// Start with flat index
	ai.switchToIndex(IndexTypeFlat)
	return ai
}

// Add adds a vector to the index
func (ai *AdaptiveIndex) Add(vector Vector) error {
	ai.mutex.Lock()
	defer ai.mutex.Unlock()

	ai.collectionSize++

	// Check if we should switch index type
	if time.Since(ai.lastSwitch) > ai.switchThreshold {
		if newType := ai.selectOptimalIndex(); newType != ai.currentType {
			if err := ai.switchToIndex(newType); err != nil {
				// Log error but continue with current index
				fmt.Printf("Failed to switch to index %s: %v\n", newType, err)
			}
		}
	}

	return ai.currentIndex.Add(vector)
}

// Search performs a search and records performance metrics
func (ai *AdaptiveIndex) Search(query []float32, k int, filter map[string]string) ([]SearchResult, error) {
	start := time.Now()

	ai.mutex.RLock()
	index := ai.currentIndex
	ai.mutex.RUnlock()

	results, err := index.Search(query, k, filter)

	// Record query statistics
	latency := time.Since(start)
	ai.stats.RecordQuery(latency)

	return results, err
}

// RangeSearch performs a range search
func (ai *AdaptiveIndex) RangeSearch(query []float32, radius float32, filter map[string]string, limit int) ([]SearchResult, error) {
	start := time.Now()

	ai.mutex.RLock()
	index := ai.currentIndex
	ai.mutex.RUnlock()

	results, err := index.RangeSearch(query, radius, filter, limit)

	// Record query statistics
	latency := time.Since(start)
	ai.stats.RecordQuery(latency)

	return results, err
}

// Delete removes a vector from the index
func (ai *AdaptiveIndex) Delete(id string) error {
	ai.mutex.Lock()
	defer ai.mutex.Unlock()

	ai.collectionSize--
	return ai.currentIndex.Delete(id)
}

// Rebuild rebuilds the current index
func (ai *AdaptiveIndex) Rebuild() error {
	ai.mutex.Lock()
	defer ai.mutex.Unlock()

	return ai.currentIndex.Rebuild()
}

// Size returns the current index size
func (ai *AdaptiveIndex) Size() int {
	ai.mutex.RLock()
	defer ai.mutex.RUnlock()

	return ai.currentIndex.Size()
}

// Type returns the current index type
func (ai *AdaptiveIndex) Type() string {
	ai.mutex.RLock()
	defer ai.mutex.RUnlock()

	return string(ai.currentType)
}

// Serialize serializes the current index
func (ai *AdaptiveIndex) Serialize() ([]byte, error) {
	ai.mutex.RLock()
	defer ai.mutex.RUnlock()

	return ai.currentIndex.Serialize()
}

// Deserialize deserializes the index state
func (ai *AdaptiveIndex) Deserialize(data []byte) error {
	ai.mutex.Lock()
	defer ai.mutex.Unlock()

	return ai.currentIndex.Deserialize(data)
}

// selectOptimalIndex chooses the best index type based on current conditions
func (ai *AdaptiveIndex) selectOptimalIndex() IndexType {
	stats := ai.stats.GetStats()

	bestScore := float64(-1)
	bestType := ai.currentType

	for _, profile := range ai.profiles {
		score := ai.calculateScore(profile, &stats)
		if score > bestScore {
			bestScore = score
			bestType = profile.IndexType
		}
	}

	return bestType
}

// calculateScore calculates a score for an index profile given current conditions
func (ai *AdaptiveIndex) calculateScore(profile IndexPerformanceProfile, stats *QueryStats) float64 {
	// Check if collection size is in optimal range
	if ai.collectionSize < profile.OptimalSizeMin || ai.collectionSize > profile.OptimalSizeMax {
		return 0 // Not suitable for this size
	}

	// Weight factors based on current usage patterns
	queryWeight := 0.4
	insertWeight := 0.2
	memoryWeight := 0.2
	accuracyWeight := 0.2

	// Adjust weights based on query patterns
	if stats.ThroughputQPS > 100 { // High query load
		queryWeight = 0.6
		insertWeight = 0.1
	}

	score := (profile.QueryLatencyScore * queryWeight) +
		(profile.InsertLatencyScore * insertWeight) +
		(profile.MemoryEfficiency * memoryWeight) +
		(profile.AccuracyScore * accuracyWeight)

	return score
}

// switchToIndex switches to a new index type
func (ai *AdaptiveIndex) switchToIndex(indexType IndexType) error {
	// Create new index
	newIndex, err := ai.factory.CreateIndex(string(indexType), ai.dimension, ai.distanceMetric)
	if err != nil {
		return fmt.Errorf("failed to create %s index: %w", indexType, err)
	}

	// Migrate data if we have an existing index
	if ai.currentIndex != nil {
		if err := ai.migrateIndex(ai.currentIndex, newIndex); err != nil {
			return fmt.Errorf("failed to migrate data to %s index: %w", indexType, err)
		}
	}

	ai.currentIndex = newIndex
	ai.currentType = indexType
	ai.lastSwitch = time.Now()

	return nil
}

// migrateIndex migrates vectors from old index to new index
func (ai *AdaptiveIndex) migrateIndex(oldIndex, newIndex Index) error {
	// This is a simplified migration - in practice you'd need to extract
	// all vectors from the old index and add them to the new one
	// For now, we'll trigger a rebuild which should be implemented
	// by the calling code to re-add all vectors
	return newIndex.Rebuild()
}

// GetCurrentStats returns current performance statistics
func (ai *AdaptiveIndex) GetCurrentStats() QueryStats {
	return ai.stats.GetStats()
}

// GetIndexInfo returns information about the current index
func (ai *AdaptiveIndex) GetIndexInfo() map[string]interface{} {
	ai.mutex.RLock()
	defer ai.mutex.RUnlock()

	stats := ai.stats.GetStats()
	return map[string]interface{}{
		"type":            string(ai.currentType),
		"size":            ai.collectionSize,
		"dimension":       ai.dimension,
		"distance_metric": string(ai.distanceMetric),
		"total_queries":   stats.TotalQueries,
		"average_latency": stats.AverageLatency.String(),
		"p95_latency":     stats.P95Latency.String(),
		"throughput_qps":  stats.ThroughputQPS,
		"last_switch":     ai.lastSwitch.Format(time.RFC3339),
	}
}
