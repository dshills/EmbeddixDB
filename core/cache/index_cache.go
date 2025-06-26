package cache

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// IndexPartitionCache implements L3 caching for hot index partitions
type IndexPartitionCache struct {
	*BaseCache
	partitionStats  map[string]*PartitionStats
	hotPartitions   []string
	maxPartitions   int
	partitionSize   int64
	accessThreshold int
	mu              sync.RWMutex
}

// PartitionStats tracks access patterns for index partitions
type PartitionStats struct {
	PartitionID     string
	AccessCount     int64
	LastAccess      time.Time
	AccessTimes     []time.Time
	LoadTime        time.Duration
	SizeBytes       int64
	VectorCount     int
	Temperature     float64 // Hot/cold score
}

// NewIndexPartitionCache creates a new index partition cache
func NewIndexPartitionCache(capacity, maxMemory int64, config CacheConfig, maxPartitions int, partitionSize int64) *IndexPartitionCache {
	baseCache := NewBaseCache(capacity, maxMemory, config.CleanupInterval)
	
	return &IndexPartitionCache{
		BaseCache:       baseCache,
		partitionStats:  make(map[string]*PartitionStats),
		maxPartitions:   maxPartitions,
		partitionSize:   partitionSize,
		accessThreshold: 5, // Minimum accesses to be considered hot
	}
}

// GetPartition retrieves a cached index partition
func (ipc *IndexPartitionCache) GetPartition(ctx context.Context, key IndexCacheKey) (interface{}, bool) {
	hash := key.Hash()
	
	// Update access statistics
	ipc.recordAccess(hash, key)
	
	value, found := ipc.Get(ctx, hash)
	if found {
		ipc.updateTemperature(hash)
	}
	
	return value, found
}

// SetPartition stores an index partition in cache
func (ipc *IndexPartitionCache) SetPartition(ctx context.Context, key IndexCacheKey, partition interface{}, loadTime time.Duration, vectorCount int) error {
	hash := key.Hash()
	
	// Record partition statistics
	ipc.mu.Lock()
	stats, exists := ipc.partitionStats[hash]
	if !exists {
		stats = &PartitionStats{
			PartitionID: hash,
			AccessTimes: make([]time.Time, 0, 100),
			SizeBytes:   ipc.partitionSize,
			VectorCount: vectorCount,
		}
		ipc.partitionStats[hash] = stats
	}
	stats.LoadTime = loadTime
	ipc.mu.Unlock()
	
	// Calculate priority based on access patterns
	priority := ipc.calculatePriority(stats)
	
	options := []CacheOption{
		WithPriority(priority),
		WithTTL(30 * time.Minute), // Longer TTL for index partitions
	}
	
	return ipc.Set(ctx, hash, partition, options...)
}

// recordAccess updates access statistics for a partition
func (ipc *IndexPartitionCache) recordAccess(hash string, key IndexCacheKey) {
	ipc.mu.Lock()
	defer ipc.mu.Unlock()
	
	stats, exists := ipc.partitionStats[hash]
	if !exists {
		stats = &PartitionStats{
			PartitionID: hash,
			AccessTimes: make([]time.Time, 0, 100),
			SizeBytes:   ipc.partitionSize,
		}
		ipc.partitionStats[hash] = stats
	}
	
	now := time.Now()
	stats.AccessCount++
	stats.LastAccess = now
	
	// Keep last 100 access times for pattern analysis
	stats.AccessTimes = append(stats.AccessTimes, now)
	if len(stats.AccessTimes) > 100 {
		stats.AccessTimes = stats.AccessTimes[1:]
	}
}

// updateTemperature updates the hot/cold score for a partition
func (ipc *IndexPartitionCache) updateTemperature(hash string) {
	ipc.mu.Lock()
	defer ipc.mu.Unlock()
	
	stats, exists := ipc.partitionStats[hash]
	if !exists {
		return
	}
	
	// Calculate temperature based on access frequency and recency
	now := time.Now()
	recencyScore := 1.0 / (1.0 + now.Sub(stats.LastAccess).Hours())
	
	// Calculate access frequency (accesses per hour)
	var frequencyScore float64
	if len(stats.AccessTimes) > 1 {
		timeSpan := stats.AccessTimes[len(stats.AccessTimes)-1].Sub(stats.AccessTimes[0])
		if timeSpan > 0 {
			frequencyScore = float64(len(stats.AccessTimes)) / timeSpan.Hours()
		}
	}
	
	// Combine scores
	stats.Temperature = 0.7*recencyScore + 0.3*frequencyScore
}

// calculatePriority calculates cache priority for a partition
func (ipc *IndexPartitionCache) calculatePriority(stats *PartitionStats) int {
	// Higher priority for frequently accessed partitions with high load time
	score := float64(stats.AccessCount) * stats.LoadTime.Seconds()
	
	// Normalize to 0-100
	priority := int(score / 100)
	if priority > 100 {
		priority = 100
	}
	
	return priority
}

// GetHotPartitions returns the list of hot partitions
func (ipc *IndexPartitionCache) GetHotPartitions(count int) []string {
	ipc.mu.RLock()
	defer ipc.mu.RUnlock()
	
	// Create slice of all partitions with sufficient accesses
	candidates := make([]*PartitionStats, 0)
	for _, stats := range ipc.partitionStats {
		if stats.AccessCount >= int64(ipc.accessThreshold) {
			candidates = append(candidates, stats)
		}
	}
	
	// Sort by temperature
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Temperature > candidates[j].Temperature
	})
	
	// Return top N
	result := make([]string, 0, count)
	for i := 0; i < count && i < len(candidates); i++ {
		result = append(result, candidates[i].PartitionID)
	}
	
	return result
}

// PreloadHotPartitions preloads frequently accessed partitions
func (ipc *IndexPartitionCache) PreloadHotPartitions(ctx context.Context, loader func(keys []string) (map[string]interface{}, error)) error {
	hotPartitions := ipc.GetHotPartitions(ipc.maxPartitions)
	if len(hotPartitions) == 0 {
		return nil
	}
	
	// Load partitions
	partitions, err := loader(hotPartitions)
	if err != nil {
		return fmt.Errorf("failed to load hot partitions: %w", err)
	}
	
	// Cache the loaded partitions
	for hash, partition := range partitions {
		// Get stats for priority calculation
		ipc.mu.RLock()
		stats, exists := ipc.partitionStats[hash]
		ipc.mu.RUnlock()
		
		if !exists {
			continue
		}
		
		// Create cache key (simplified - in production would parse properly)
		key := IndexCacheKey{
			PartitionHash: hash,
		}
		
		if err := ipc.SetPartition(ctx, key, partition, stats.LoadTime, stats.VectorCount); err != nil {
			// Log error but continue with other partitions
			continue
		}
	}
	
	return nil
}

// GetPartitionStats returns statistics for all partitions
func (ipc *IndexPartitionCache) GetPartitionStats() map[string]PartitionStats {
	ipc.mu.RLock()
	defer ipc.mu.RUnlock()
	
	// Create copy to avoid lock contention
	result := make(map[string]PartitionStats)
	for k, v := range ipc.partitionStats {
		result[k] = *v
	}
	
	return result
}

// AnalyzeAccessPatterns analyzes partition access patterns
func (ipc *IndexPartitionCache) AnalyzeAccessPatterns() AccessPatternAnalysis {
	ipc.mu.RLock()
	defer ipc.mu.RUnlock()
	
	totalAccesses := int64(0)
	hotCount := 0
	coldCount := 0
	var avgTemperature float64
	
	// Time-based analysis
	hourlyAccesses := make(map[int]int64)
	
	for _, stats := range ipc.partitionStats {
		totalAccesses += stats.AccessCount
		avgTemperature += stats.Temperature
		
		if stats.Temperature > 0.5 {
			hotCount++
		} else {
			coldCount++
		}
		
		// Analyze access times
		for _, accessTime := range stats.AccessTimes {
			hour := accessTime.Hour()
			hourlyAccesses[hour]++
		}
	}
	
	partitionCount := len(ipc.partitionStats)
	if partitionCount > 0 {
		avgTemperature /= float64(partitionCount)
	}
	
	// Find peak hours
	peakHour := 0
	peakAccesses := int64(0)
	for hour, accesses := range hourlyAccesses {
		if accesses > peakAccesses {
			peakHour = hour
			peakAccesses = accesses
		}
	}
	
	return AccessPatternAnalysis{
		TotalPartitions:    partitionCount,
		HotPartitions:      hotCount,
		ColdPartitions:     coldCount,
		TotalAccesses:      totalAccesses,
		AverageTemperature: avgTemperature,
		PeakHour:           peakHour,
		HourlyDistribution: hourlyAccesses,
	}
}

// AccessPatternAnalysis contains partition access pattern analysis
type AccessPatternAnalysis struct {
	TotalPartitions    int              `json:"total_partitions"`
	HotPartitions      int              `json:"hot_partitions"`
	ColdPartitions     int              `json:"cold_partitions"`
	TotalAccesses      int64            `json:"total_accesses"`
	AverageTemperature float64          `json:"average_temperature"`
	PeakHour           int              `json:"peak_hour"`
	HourlyDistribution map[int]int64    `json:"hourly_distribution"`
}

// CleanupColdPartitions removes partitions that haven't been accessed recently
func (ipc *IndexPartitionCache) CleanupColdPartitions(ctx context.Context, maxAge time.Duration) error {
	ipc.mu.Lock()
	defer ipc.mu.Unlock()
	
	now := time.Now()
	toRemove := make([]string, 0)
	
	for hash, stats := range ipc.partitionStats {
		if now.Sub(stats.LastAccess) > maxAge && stats.Temperature < 0.1 {
			toRemove = append(toRemove, hash)
		}
	}
	
	// Remove cold partitions from stats
	for _, hash := range toRemove {
		delete(ipc.partitionStats, hash)
		// Also remove from cache
		ipc.Delete(ctx, hash)
	}
	
	return nil
}