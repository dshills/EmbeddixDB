package query

import (
	"container/heap"
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// ProgressiveSearch implements progressive result delivery with early termination
type ProgressiveSearch struct {
	mu              sync.RWMutex
	resultHeap      *ResultHeap
	seenIDs         map[string]bool
	targetCount     int
	confidenceLevel float64
	startTime       time.Time
	terminated      atomic.Bool
	metrics         *ProgressiveMetrics
}

// ProgressiveMetrics tracks progressive search performance
type ProgressiveMetrics struct {
	mu                     sync.RWMutex
	ResultsDelivered       int64
	EarlyTerminations      int64
	AverageConfidence      float64
	TimeToFirstResult      time.Duration
	TimeToTargetConfidence time.Duration
}

// ResultHeap implements a min-heap for search results (by score)
type ResultHeap []SearchResult

func (h ResultHeap) Len() int           { return len(h) }
func (h ResultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h ResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *ResultHeap) Push(x interface{}) {
	*h = append(*h, x.(SearchResult))
}

func (h *ResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// ProgressiveConfig configures progressive search behavior
type ProgressiveConfig struct {
	TargetCount         int
	MinConfidence       float64
	MaxLatency          time.Duration
	CheckpointInterval  int // Check termination every N results
	ConfidenceThreshold float64
}

// DefaultProgressiveConfig returns default progressive search configuration
func DefaultProgressiveConfig() ProgressiveConfig {
	return ProgressiveConfig{
		TargetCount:         100,
		MinConfidence:       0.95,
		MaxLatency:          500 * time.Millisecond,
		CheckpointInterval:  10,
		ConfidenceThreshold: 0.98,
	}
}

// NewProgressiveSearch creates a new progressive search instance
func NewProgressiveSearch(config ProgressiveConfig) *ProgressiveSearch {
	h := &ResultHeap{}
	heap.Init(h)

	return &ProgressiveSearch{
		resultHeap:      h,
		seenIDs:         make(map[string]bool),
		targetCount:     config.TargetCount,
		confidenceLevel: 0.0,
		startTime:       time.Now(),
		metrics:         &ProgressiveMetrics{},
	}
}

// AddResult adds a result to the progressive search
func (ps *ProgressiveSearch) AddResult(result SearchResult) bool {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Check if already terminated
	if ps.terminated.Load() {
		return false
	}

	// Skip duplicates
	if ps.seenIDs[result.ID] {
		return false
	}
	ps.seenIDs[result.ID] = true

	// Add to heap
	heap.Push(ps.resultHeap, result)

	// If we have more than target count, remove the lowest scoring result
	if ps.resultHeap.Len() > ps.targetCount {
		heap.Pop(ps.resultHeap)
	}

	// Update metrics
	atomic.AddInt64(&ps.metrics.ResultsDelivered, 1)

	// Record time to first result
	if ps.metrics.TimeToFirstResult == 0 {
		ps.metrics.mu.Lock()
		ps.metrics.TimeToFirstResult = time.Since(ps.startTime)
		ps.metrics.mu.Unlock()
	}

	return true
}

// CheckTermination checks if search can be terminated early
func (ps *ProgressiveSearch) CheckTermination(totalSearched, totalVectors int) bool {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Already terminated
	if ps.terminated.Load() {
		return true
	}

	// Not enough results yet
	if ps.resultHeap.Len() < ps.targetCount/2 {
		return false
	}

	// Calculate confidence based on search progress
	searchProgress := float64(totalSearched) / float64(totalVectors)

	// Get current worst score in our top results
	worstScore := (*ps.resultHeap)[0].Score

	// Estimate confidence that we have the true top-k results
	// This is a simplified model - real implementation would use statistical methods
	ps.confidenceLevel = ps.calculateConfidence(searchProgress, worstScore)

	// Update average confidence metric
	ps.metrics.mu.Lock()
	ps.metrics.AverageConfidence = (ps.metrics.AverageConfidence + ps.confidenceLevel) / 2
	ps.metrics.mu.Unlock()

	// Check if we can terminate
	if ps.confidenceLevel >= 0.98 {
		ps.terminated.Store(true)
		atomic.AddInt64(&ps.metrics.EarlyTerminations, 1)

		ps.metrics.mu.Lock()
		ps.metrics.TimeToTargetConfidence = time.Since(ps.startTime)
		ps.metrics.mu.Unlock()

		return true
	}

	return false
}

// calculateConfidence estimates confidence in current results
func (ps *ProgressiveSearch) calculateConfidence(searchProgress float64, worstScore float32) float64 {
	// Simple confidence model based on:
	// 1. How much of the search space we've covered
	// 2. The score distribution of current results
	// 3. Time elapsed

	baseConfidence := searchProgress * 0.7 // 70% weight on search progress

	// If we have high-scoring results, increase confidence
	if worstScore > 0.9 {
		baseConfidence += 0.2
	} else if worstScore > 0.8 {
		baseConfidence += 0.1
	}

	// Time-based confidence boost
	elapsed := time.Since(ps.startTime)
	if elapsed > 100*time.Millisecond {
		baseConfidence += 0.1
	}

	// Cap at 1.0
	if baseConfidence > 1.0 {
		baseConfidence = 1.0
	}

	return baseConfidence
}

// GetResults returns the current top results
func (ps *ProgressiveSearch) GetResults() []SearchResult {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	// Convert heap to sorted slice
	results := make([]SearchResult, ps.resultHeap.Len())
	copy(results, *ps.resultHeap)

	// Sort by score (descending)
	for i := 0; i < len(results)/2; i++ {
		results[i], results[len(results)-1-i] = results[len(results)-1-i], results[i]
	}

	return results
}

// GetMetrics returns progressive search metrics
func (ps *ProgressiveSearch) GetMetrics() ProgressiveMetrics {
	ps.metrics.mu.RLock()
	defer ps.metrics.mu.RUnlock()

	return ProgressiveMetrics{
		ResultsDelivered:       atomic.LoadInt64(&ps.metrics.ResultsDelivered),
		EarlyTerminations:      atomic.LoadInt64(&ps.metrics.EarlyTerminations),
		AverageConfidence:      ps.metrics.AverageConfidence,
		TimeToFirstResult:      ps.metrics.TimeToFirstResult,
		TimeToTargetConfidence: ps.metrics.TimeToTargetConfidence,
	}
}

// ProgressiveExecutor coordinates progressive search execution
type ProgressiveExecutor struct {
	config   ProgressiveConfig
	executor *ParallelExecutor
	metrics  *ProgressiveExecutorMetrics
}

// ProgressiveExecutorMetrics tracks overall progressive execution metrics
type ProgressiveExecutorMetrics struct {
	mu                  sync.RWMutex
	TotalSearches       int64
	ProgressiveSearches int64
	EarlyTerminations   int64
	AverageSpeedup      float64
}

// NewProgressiveExecutor creates a new progressive search executor
func NewProgressiveExecutor(config ProgressiveConfig, parallelExecutor *ParallelExecutor) *ProgressiveExecutor {
	return &ProgressiveExecutor{
		config:   config,
		executor: parallelExecutor,
		metrics:  &ProgressiveExecutorMetrics{},
	}
}

// ExecuteProgressive performs a progressive search with early termination
func (pe *ProgressiveExecutor) ExecuteProgressive(ctx context.Context, collection string, req SearchRequest) ([]SearchResult, error) {
	atomic.AddInt64(&pe.metrics.TotalSearches, 1)

	// Create progressive search instance
	progressive := NewProgressiveSearch(pe.config)

	// Create a channel for streaming results
	resultChan := make(chan SearchResult, 1000)
	errorChan := make(chan error, 1)

	// Start result collector
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		checkCounter := 0

		for result := range resultChan {
			progressive.AddResult(result)
			checkCounter++

			// Check for early termination periodically
			if checkCounter >= pe.config.CheckpointInterval {
				checkCounter = 0
				// In real implementation, we'd get actual search progress
				if progressive.CheckTermination(1000, 10000) {
					atomic.AddInt64(&pe.metrics.EarlyTerminations, 1)
					// Signal early termination
					break
				}
			}
		}
	}()

	// Execute search with result streaming
	go func() {
		defer close(resultChan)
		// This would be the actual search implementation
		// For now, we'll simulate some results
		for i := 0; i < 100; i++ {
			select {
			case <-ctx.Done():
				errorChan <- ctx.Err()
				return
			case resultChan <- SearchResult{
				ID:    fmt.Sprintf("vec_%d", i),
				Score: float32(0.9 - float64(i)*0.001),
			}:
			}
		}
	}()

	// Wait for completion or timeout
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case err := <-errorChan:
		return nil, err
	case <-time.After(pe.config.MaxLatency):
		// Timeout - return best results so far
		wg.Wait()
		return progressive.GetResults(), nil
	}
}

// StreamingResultWriter implements streaming result delivery
type StreamingResultWriter struct {
	resultChan   chan<- SearchResult
	errorChan    chan<- error
	closed       atomic.Bool
	writtenCount atomic.Int64
}

// NewStreamingResultWriter creates a new streaming result writer
func NewStreamingResultWriter(resultChan chan<- SearchResult, errorChan chan<- error) *StreamingResultWriter {
	return &StreamingResultWriter{
		resultChan: resultChan,
		errorChan:  errorChan,
	}
}

// WriteResult writes a single result to the stream
func (w *StreamingResultWriter) WriteResult(result SearchResult) error {
	if w.closed.Load() {
		return fmt.Errorf("writer is closed")
	}

	select {
	case w.resultChan <- result:
		w.writtenCount.Add(1)
		return nil
	default:
		return fmt.Errorf("result channel is full")
	}
}

// WriteError writes an error to the stream
func (w *StreamingResultWriter) WriteError(err error) {
	if w.closed.Load() {
		return
	}

	select {
	case w.errorChan <- err:
	default:
		// Error channel is full, drop the error
	}
}

// Close closes the streaming writer
func (w *StreamingResultWriter) Close() {
	if w.closed.CompareAndSwap(false, true) {
		close(w.resultChan)
	}
}

// GetWrittenCount returns the number of results written
func (w *StreamingResultWriter) GetWrittenCount() int64 {
	return w.writtenCount.Load()
}
