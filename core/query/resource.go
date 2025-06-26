package query

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ResourceManager manages query execution resources with cancellation support
type ResourceManager struct {
	mu             sync.RWMutex
	activeQueries  map[string]*QueryContext
	maxConcurrent  int
	maxMemoryBytes int64
	currentMemory  atomic.Int64
	metrics        *ResourceMetrics
	limiter        *RateLimiter
}

// QueryContext tracks resources for a single query
type QueryContext struct {
	ID              string
	Context         context.Context
	Cancel          context.CancelFunc
	StartTime       time.Time
	MemoryAllocated int64
	CPUTime         time.Duration
	State           QueryState
	mu              sync.RWMutex
}

// QueryState represents the current state of a query
type QueryState int

const (
	QueryStatePending QueryState = iota
	QueryStateRunning
	QueryStateCompleted
	QueryStateCancelled
	QueryStateFailed
)

// ResourceMetrics tracks resource usage metrics
type ResourceMetrics struct {
	mu                    sync.RWMutex
	TotalQueries          int64
	ActiveQueries         int64
	CancelledQueries      int64
	MemoryPressureEvents  int64
	ThrottledQueries      int64
	AverageMemoryPerQuery int64
	PeakMemoryUsage       int64
	TotalCPUTime          time.Duration
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	tokens     atomic.Int64
	maxTokens  int64
	refillRate int64        // tokens per second
	lastRefill atomic.Int64 // Unix timestamp
}

// ResourceConfig configures resource management
type ResourceConfig struct {
	MaxConcurrentQueries int
	MaxMemoryBytes       int64
	MaxQueryDuration     time.Duration
	RateLimitQPS         int
	MemoryCheckInterval  time.Duration
}

// DefaultResourceConfig returns default resource configuration
func DefaultResourceConfig() ResourceConfig {
	totalMemory := getTotalMemory()
	return ResourceConfig{
		MaxConcurrentQueries: runtime.NumCPU() * 2,
		MaxMemoryBytes:       totalMemory / 4, // Use 25% of system memory
		MaxQueryDuration:     30 * time.Second,
		RateLimitQPS:         1000,
		MemoryCheckInterval:  100 * time.Millisecond,
	}
}

// NewResourceManager creates a new resource manager
func NewResourceManager(config ResourceConfig) *ResourceManager {
	rm := &ResourceManager{
		activeQueries:  make(map[string]*QueryContext),
		maxConcurrent:  config.MaxConcurrentQueries,
		maxMemoryBytes: config.MaxMemoryBytes,
		metrics:        &ResourceMetrics{},
		limiter: &RateLimiter{
			maxTokens:  int64(config.RateLimitQPS),
			refillRate: int64(config.RateLimitQPS),
		},
	}

	// Initialize rate limiter with full tokens
	rm.limiter.tokens.Store(rm.limiter.maxTokens)
	rm.limiter.lastRefill.Store(time.Now().Unix())

	// Start background resource monitor
	go rm.monitorResources(config.MemoryCheckInterval)

	return rm
}

// AcquireQueryContext acquires resources for a new query
func (rm *ResourceManager) AcquireQueryContext(ctx context.Context, queryID string, estimatedMemory int64) (*QueryContext, error) {
	// Check rate limit
	if !rm.limiter.TryAcquire() {
		rm.metrics.mu.Lock()
		rm.metrics.ThrottledQueries++
		rm.metrics.mu.Unlock()
		return nil, fmt.Errorf("rate limit exceeded")
	}

	// Check concurrent query limit
	rm.mu.RLock()
	activeCount := len(rm.activeQueries)
	rm.mu.RUnlock()

	if activeCount >= rm.maxConcurrent {
		return nil, fmt.Errorf("max concurrent queries reached: %d", rm.maxConcurrent)
	}

	// Check memory availability
	currentMem := rm.currentMemory.Load()
	if currentMem+estimatedMemory > rm.maxMemoryBytes {
		rm.metrics.mu.Lock()
		rm.metrics.MemoryPressureEvents++
		rm.metrics.mu.Unlock()
		return nil, fmt.Errorf("insufficient memory: need %d bytes, available %d bytes",
			estimatedMemory, rm.maxMemoryBytes-currentMem)
	}

	// Create query context with cancellation
	queryCtx, cancel := context.WithCancel(ctx)

	qc := &QueryContext{
		ID:              queryID,
		Context:         queryCtx,
		Cancel:          cancel,
		StartTime:       time.Now(),
		MemoryAllocated: estimatedMemory,
		State:           QueryStatePending,
	}

	// Register query
	rm.mu.Lock()
	rm.activeQueries[queryID] = qc
	rm.mu.Unlock()

	// Update metrics
	rm.currentMemory.Add(estimatedMemory)
	rm.metrics.mu.Lock()
	rm.metrics.TotalQueries++
	rm.metrics.ActiveQueries++
	if rm.currentMemory.Load() > rm.metrics.PeakMemoryUsage {
		rm.metrics.PeakMemoryUsage = rm.currentMemory.Load()
	}
	rm.metrics.mu.Unlock()

	// Set query state to running
	qc.mu.Lock()
	qc.State = QueryStateRunning
	qc.mu.Unlock()

	return qc, nil
}

// ReleaseQueryContext releases resources for a completed query
func (rm *ResourceManager) ReleaseQueryContext(queryID string, finalState QueryState) {
	rm.mu.Lock()
	qc, exists := rm.activeQueries[queryID]
	if !exists {
		rm.mu.Unlock()
		return
	}
	delete(rm.activeQueries, queryID)
	rm.mu.Unlock()

	// Update query state
	qc.mu.Lock()
	qc.State = finalState
	qc.CPUTime = time.Since(qc.StartTime)
	qc.mu.Unlock()

	// Cancel context if not already done
	qc.Cancel()

	// Release memory
	rm.currentMemory.Add(-qc.MemoryAllocated)

	// Update metrics
	rm.metrics.mu.Lock()
	rm.metrics.ActiveQueries--
	rm.metrics.TotalCPUTime += qc.CPUTime
	if finalState == QueryStateCancelled {
		rm.metrics.CancelledQueries++
	}

	// Update average memory per query
	if rm.metrics.TotalQueries > 0 {
		rm.metrics.AverageMemoryPerQuery =
			(rm.metrics.AverageMemoryPerQuery*(rm.metrics.TotalQueries-1) + qc.MemoryAllocated) /
				rm.metrics.TotalQueries
	}
	rm.metrics.mu.Unlock()
}

// CancelQuery cancels a running query
func (rm *ResourceManager) CancelQuery(queryID string) error {
	rm.mu.RLock()
	qc, exists := rm.activeQueries[queryID]
	rm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("query %s not found", queryID)
	}

	// Cancel the query context
	qc.Cancel()

	// Update state
	qc.mu.Lock()
	qc.State = QueryStateCancelled
	qc.mu.Unlock()

	// Release resources
	rm.ReleaseQueryContext(queryID, QueryStateCancelled)

	return nil
}

// GetQueryContext retrieves an active query context
func (rm *ResourceManager) GetQueryContext(queryID string) (*QueryContext, bool) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	qc, exists := rm.activeQueries[queryID]
	return qc, exists
}

// ListActiveQueries returns all active queries
func (rm *ResourceManager) ListActiveQueries() []QueryInfo {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	queries := make([]QueryInfo, 0, len(rm.activeQueries))
	for _, qc := range rm.activeQueries {
		qc.mu.RLock()
		info := QueryInfo{
			ID:              qc.ID,
			StartTime:       qc.StartTime,
			Duration:        time.Since(qc.StartTime),
			MemoryAllocated: qc.MemoryAllocated,
			State:           qc.State,
		}
		qc.mu.RUnlock()
		queries = append(queries, info)
	}

	return queries
}

// QueryInfo provides information about a query
type QueryInfo struct {
	ID              string
	StartTime       time.Time
	Duration        time.Duration
	MemoryAllocated int64
	State           QueryState
}

// monitorResources monitors system resources and enforces limits
func (rm *ResourceManager) monitorResources(checkInterval time.Duration) {
	ticker := time.NewTicker(checkInterval)
	defer ticker.Stop()

	for range ticker.C {
		rm.checkResourceLimits()
		rm.refillRateLimiter()
	}
}

// checkResourceLimits checks and enforces resource limits
func (rm *ResourceManager) checkResourceLimits() {
	// Get current memory usage
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Check if we're under memory pressure
	if int64(memStats.Alloc) > rm.maxMemoryBytes*9/10 { // 90% threshold
		rm.handleMemoryPressure()
	}

	// Check for long-running queries
	rm.mu.RLock()
	queries := make([]*QueryContext, 0, len(rm.activeQueries))
	for _, qc := range rm.activeQueries {
		queries = append(queries, qc)
	}
	rm.mu.RUnlock()

	// Cancel queries that exceed time limits
	for _, qc := range queries {
		if time.Since(qc.StartTime) > 30*time.Second {
			rm.CancelQuery(qc.ID)
		}
	}
}

// handleMemoryPressure handles memory pressure situations
func (rm *ResourceManager) handleMemoryPressure() {
	rm.metrics.mu.Lock()
	rm.metrics.MemoryPressureEvents++
	rm.metrics.mu.Unlock()

	// Force garbage collection
	runtime.GC()

	// Cancel oldest queries if still under pressure
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	if int64(memStats.Alloc) > rm.maxMemoryBytes*9/10 {
		rm.cancelOldestQueries(1)
	}
}

// cancelOldestQueries cancels the n oldest queries
func (rm *ResourceManager) cancelOldestQueries(n int) {
	rm.mu.RLock()
	queries := make([]*QueryContext, 0, len(rm.activeQueries))
	for _, qc := range rm.activeQueries {
		queries = append(queries, qc)
	}
	rm.mu.RUnlock()

	// Sort by start time (oldest first)
	// In production, use proper sorting

	// Cancel oldest queries
	cancelled := 0
	for _, qc := range queries {
		if cancelled >= n {
			break
		}
		rm.CancelQuery(qc.ID)
		cancelled++
	}
}

// refillRateLimiter refills rate limiter tokens
func (rm *ResourceManager) refillRateLimiter() {
	now := time.Now().Unix()
	lastRefill := rm.limiter.lastRefill.Load()

	if now > lastRefill {
		elapsed := now - lastRefill
		tokensToAdd := elapsed * rm.limiter.refillRate

		currentTokens := rm.limiter.tokens.Load()
		newTokens := currentTokens + tokensToAdd
		if newTokens > rm.limiter.maxTokens {
			newTokens = rm.limiter.maxTokens
		}

		rm.limiter.tokens.Store(newTokens)
		rm.limiter.lastRefill.Store(now)
	}
}

// TryAcquire attempts to acquire a token from the rate limiter
func (rl *RateLimiter) TryAcquire() bool {
	for {
		current := rl.tokens.Load()
		if current <= 0 {
			return false
		}
		if rl.tokens.CompareAndSwap(current, current-1) {
			return true
		}
	}
}

// GetMetrics returns current resource metrics
func (rm *ResourceManager) GetMetrics() ResourceMetrics {
	rm.metrics.mu.RLock()
	defer rm.metrics.mu.RUnlock()

	return ResourceMetrics{
		TotalQueries:          rm.metrics.TotalQueries,
		ActiveQueries:         rm.metrics.ActiveQueries,
		CancelledQueries:      rm.metrics.CancelledQueries,
		MemoryPressureEvents:  rm.metrics.MemoryPressureEvents,
		ThrottledQueries:      rm.metrics.ThrottledQueries,
		AverageMemoryPerQuery: rm.metrics.AverageMemoryPerQuery,
		PeakMemoryUsage:       rm.metrics.PeakMemoryUsage,
		TotalCPUTime:          rm.metrics.TotalCPUTime,
	}
}

// getTotalMemory returns total system memory in bytes
func getTotalMemory() int64 {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	// This is a simplified approach - in production, use system-specific methods
	return int64(memStats.Sys)
}

// WithQueryContext wraps a function with resource management
func (rm *ResourceManager) WithQueryContext(ctx context.Context, queryID string, estimatedMemory int64, fn func(context.Context) error) error {
	// Acquire resources
	qc, err := rm.AcquireQueryContext(ctx, queryID, estimatedMemory)
	if err != nil {
		return fmt.Errorf("failed to acquire query context: %w", err)
	}

	// Ensure cleanup
	defer func() {
		if qc.State == QueryStateRunning {
			rm.ReleaseQueryContext(queryID, QueryStateCompleted)
		}
	}()

	// Execute function with managed context
	err = fn(qc.Context)
	if err != nil {
		rm.ReleaseQueryContext(queryID, QueryStateFailed)
		return err
	}

	return nil
}
