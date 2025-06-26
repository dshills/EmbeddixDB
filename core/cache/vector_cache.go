package cache

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// CachedVector represents a vector for caching (avoiding import cycle)
type CachedVector struct {
	ID         string            `json:"id"`
	Values     []float32         `json:"values"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Collection string            `json:"collection,omitempty"`
}

// VectorCache implements L2 caching for frequently accessed vectors
type VectorCache struct {
	*BaseCache
	costTracker    *ComputeCostTracker
	userHotSets    map[string]*HotSet
	userHotSetSize int
	mu             sync.RWMutex
}

// ComputeCostTracker tracks the computational cost of vectors
type ComputeCostTracker struct {
	mu    sync.RWMutex
	costs map[string]float64
}

// NewComputeCostTracker creates a new cost tracker
func NewComputeCostTracker() *ComputeCostTracker {
	return &ComputeCostTracker{
		costs: make(map[string]float64),
	}
}

// RecordCost records the compute cost for a vector
func (ct *ComputeCostTracker) RecordCost(key string, cost float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	
	// Exponential moving average
	if oldCost, exists := ct.costs[key]; exists {
		ct.costs[key] = 0.7*oldCost + 0.3*cost
	} else {
		ct.costs[key] = cost
	}
}

// GetCost returns the average compute cost for a vector
func (ct *ComputeCostTracker) GetCost(key string) float64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	
	return ct.costs[key]
}

// HotSet tracks frequently accessed items for a user
type HotSet struct {
	items       map[string]*HotSetItem
	maxSize     int
	accessDecay float64
}

// HotSetItem represents an item in the hot set
type HotSetItem struct {
	Key              string
	AccessFrequency  float64
	LastAccess       time.Time
	ComputeCost      float64
}

// NewHotSet creates a new hot set
func NewHotSet(maxSize int) *HotSet {
	return &HotSet{
		items:       make(map[string]*HotSetItem),
		maxSize:     maxSize,
		accessDecay: 0.95, // Decay factor for access frequency
	}
}

// Access records an access and updates frequency
func (hs *HotSet) Access(key string, computeCost float64) {
	now := time.Now()
	
	// Decay all frequencies based on time
	for _, item := range hs.items {
		timeDiff := now.Sub(item.LastAccess).Seconds()
		decayFactor := math.Pow(hs.accessDecay, timeDiff/3600) // Decay per hour
		item.AccessFrequency *= decayFactor
	}
	
	// Update or add the accessed item
	if item, exists := hs.items[key]; exists {
		item.AccessFrequency += 1.0
		item.LastAccess = now
		item.ComputeCost = computeCost
	} else {
		// Add new item if there's space or if it should replace the coldest item
		if len(hs.items) < hs.maxSize {
			hs.items[key] = &HotSetItem{
				Key:             key,
				AccessFrequency: 1.0,
				LastAccess:      now,
				ComputeCost:     computeCost,
			}
		} else {
			// Find coldest item
			var coldestKey string
			coldestScore := math.MaxFloat64
			
			for k, item := range hs.items {
				score := item.AccessFrequency / (1 + item.ComputeCost)
				if score < coldestScore {
					coldestScore = score
					coldestKey = k
				}
			}
			
			// Replace if new item has higher potential value
			newScore := 1.0 / (1 + computeCost)
			if newScore > coldestScore {
				delete(hs.items, coldestKey)
				hs.items[key] = &HotSetItem{
					Key:             key,
					AccessFrequency: 1.0,
					LastAccess:      now,
					ComputeCost:     computeCost,
				}
			}
		}
	}
}

// Contains checks if a key is in the hot set
func (hs *HotSet) Contains(key string) bool {
	_, exists := hs.items[key]
	return exists
}

// GetTopItems returns the top N items by access frequency
func (hs *HotSet) GetTopItems(n int) []string {
	// Create slice of all items
	items := make([]*HotSetItem, 0, len(hs.items))
	for _, item := range hs.items {
		items = append(items, item)
	}
	
	// Sort by access frequency
	// In production, use a more efficient selection algorithm
	for i := 0; i < len(items)-1; i++ {
		for j := i + 1; j < len(items); j++ {
			if items[i].AccessFrequency < items[j].AccessFrequency {
				items[i], items[j] = items[j], items[i]
			}
		}
	}
	
	// Return top N
	result := make([]string, 0, n)
	for i := 0; i < n && i < len(items); i++ {
		result = append(result, items[i].Key)
	}
	
	return result
}

// NewVectorCache creates a new vector cache
func NewVectorCache(capacity, maxMemory int64, config CacheConfig, userHotSetSize int) *VectorCache {
	baseCache := NewBaseCache(capacity, maxMemory, config.CleanupInterval)
	
	// Set up cost-aware eviction policy
	evictionPolicy := NewCostAwareEvictionPolicy()
	baseCache.SetEvictionPolicy(evictionPolicy)
	
	return &VectorCache{
		BaseCache:      baseCache,
		costTracker:    NewComputeCostTracker(),
		userHotSets:    make(map[string]*HotSet),
		userHotSetSize: userHotSetSize,
	}
}

// GetVector retrieves a cached vector
func (vc *VectorCache) GetVector(ctx context.Context, key VectorCacheKey, userID string) (*CachedVector, bool) {
	hash := key.Hash()
	
	// Record user access for hot set tracking
	if userID != "" {
		vc.recordUserAccess(userID, hash)
	}
	
	value, found := vc.Get(ctx, hash)
	if !found {
		return nil, false
	}
	
	vector, ok := value.(*CachedVector)
	if !ok {
		return nil, false
	}
	
	return vector, true
}

// SetVector stores a vector in cache
func (vc *VectorCache) SetVector(ctx context.Context, key VectorCacheKey, vector *CachedVector, computeCost float64, userID string) error {
	hash := key.Hash()
	
	// Record compute cost
	vc.costTracker.RecordCost(hash, computeCost)
	
	// Update user hot set
	if userID != "" {
		vc.recordUserAccess(userID, hash)
	}
	
	// Calculate priority based on compute cost
	priority := int(computeCost / 10) // Simple priority calculation
	
	options := []CacheOption{
		WithPriority(priority),
	}
	
	if userID != "" {
		options = append(options, WithUserContext(userID))
	}
	
	return vc.Set(ctx, hash, vector, options...)
}

// recordUserAccess updates user hot set
func (vc *VectorCache) recordUserAccess(userID, key string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	hotSet, exists := vc.userHotSets[userID]
	if !exists {
		hotSet = NewHotSet(vc.userHotSetSize)
		vc.userHotSets[userID] = hotSet
	}
	
	computeCost := vc.costTracker.GetCost(key)
	hotSet.Access(key, computeCost)
}

// GetUserHotVectors returns the most frequently accessed vectors for a user
func (vc *VectorCache) GetUserHotVectors(userID string, count int) []string {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	
	hotSet, exists := vc.userHotSets[userID]
	if !exists {
		return nil
	}
	
	return hotSet.GetTopItems(count)
}

// PreloadUserHotSet preloads frequently accessed vectors for a user
func (vc *VectorCache) PreloadUserHotSet(ctx context.Context, userID string, loader func(keys []string) (map[string]*CachedVector, error)) error {
	hotKeys := vc.GetUserHotVectors(userID, vc.userHotSetSize)
	if len(hotKeys) == 0 {
		return nil
	}
	
	// Load vectors
	vectors, err := loader(hotKeys)
	if err != nil {
		return fmt.Errorf("failed to load hot vectors: %w", err)
	}
	
	// Cache the loaded vectors
	for key, vector := range vectors {
		// Parse the key back to VectorCacheKey
		// In production, would store structured keys
		cacheKey := VectorCacheKey{
			Collection: vector.Collection,
			VectorID:   vector.ID,
		}
		
		// Use average compute cost for preloaded items
		avgCost := vc.costTracker.GetCost(key)
		if avgCost == 0 {
			avgCost = 10.0 // Default cost
		}
		
		if err := vc.SetVector(ctx, cacheKey, vector, avgCost, userID); err != nil {
			// Log error but continue with other vectors
			continue
		}
	}
	
	return nil
}

// CostAwareEvictionPolicy implements cost-aware eviction
type CostAwareEvictionPolicy struct {
	mu sync.RWMutex
}

// NewCostAwareEvictionPolicy creates a new cost-aware eviction policy
func NewCostAwareEvictionPolicy() *CostAwareEvictionPolicy {
	return &CostAwareEvictionPolicy{}
}

// ShouldEvict determines if an item should be evicted
func (p *CostAwareEvictionPolicy) ShouldEvict(item CachedItem) bool {
	// Items with high compute cost and low access should be kept
	score := float64(item.AccessCount) / (1 + item.ComputeCost)
	return score < 0.1 // Evict if score is very low
}

// SelectVictim chooses which item to evict based on cost and access patterns
func (p *CostAwareEvictionPolicy) SelectVictim(items []CachedItem) CachedItem {
	if len(items) == 0 {
		return CachedItem{}
	}
	
	// Find item with lowest value score
	lowestScore := math.MaxFloat64
	victimIndex := 0
	
	for i, item := range items {
		// Calculate value score: access frequency / (age * compute cost)
		age := time.Since(item.CreatedAt).Seconds()
		score := float64(item.AccessCount) / (age * (1 + item.ComputeCost))
		
		if score < lowestScore {
			lowestScore = score
			victimIndex = i
		}
	}
	
	return items[victimIndex]
}

// OnAccess updates access statistics
func (p *CostAwareEvictionPolicy) OnAccess(key string) {
	// Access tracking handled by base cache
}

// OnEvict handles cleanup when an item is evicted
func (p *CostAwareEvictionPolicy) OnEvict(item CachedItem) {
	// Could log eviction for analysis
}