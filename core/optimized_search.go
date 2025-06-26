package core

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core/performance"
	"github.com/dshills/EmbeddixDB/core/query"
)

// OptimizedVectorStore wraps VectorStoreImpl with query optimization capabilities
type OptimizedVectorStore struct {
	*VectorStoreImpl
	queryPlanner        *query.QueryPlanner
	parallelExecutor    *query.ParallelExecutor
	progressiveExecutor *query.ProgressiveExecutor
	streamingExecutor   *query.StreamingExecutor
	resourceManager     *query.ResourceManager
	config              OptimizationConfig
	mu                  sync.RWMutex
}

// OptimizationConfig configures query optimization behavior
type OptimizationConfig struct {
	EnableParallelExecution bool
	EnableProgressiveSearch bool
	EnableStreamingResults  bool
	EnableQueryPlanCaching  bool
	MaxConcurrentQueries    int
	MaxMemoryBytes          int64
	CacheSizeMB             int
	ParallelWorkers         int
}

// DefaultOptimizationConfig returns default optimization settings
func DefaultOptimizationConfig() OptimizationConfig {
	return OptimizationConfig{
		EnableParallelExecution: true,
		EnableProgressiveSearch: true,
		EnableStreamingResults:  false, // Disabled by default for backward compatibility
		EnableQueryPlanCaching:  true,
		MaxConcurrentQueries:    100,
		MaxMemoryBytes:          1 << 30, // 1GB
		CacheSizeMB:             100,
		ParallelWorkers:         8,
	}
}

// NewOptimizedVectorStore creates a new optimized vector store
func NewOptimizedVectorStore(base *VectorStoreImpl, config OptimizationConfig) *OptimizedVectorStore {
	// Create query optimization components
	queryPlanner := query.NewQueryPlanner(1000) // Max 1000 cached plans
	parallelExecutor := query.NewParallelExecutor(config.ParallelWorkers, config.ParallelWorkers*10)

	progressiveConfig := query.DefaultProgressiveConfig()
	progressiveExecutor := query.NewProgressiveExecutor(progressiveConfig, parallelExecutor)

	streamingConfig := query.DefaultStreamingConfig()
	streamingExecutor := query.NewStreamingExecutor(parallelExecutor, streamingConfig)

	resourceConfig := query.ResourceConfig{
		MaxConcurrentQueries: config.MaxConcurrentQueries,
		MaxMemoryBytes:       config.MaxMemoryBytes,
		MaxQueryDuration:     30 * time.Second,
		RateLimitQPS:         1000,
		MemoryCheckInterval:  100 * time.Millisecond,
	}
	resourceManager := query.NewResourceManager(resourceConfig)

	return &OptimizedVectorStore{
		VectorStoreImpl:     base,
		queryPlanner:        queryPlanner,
		parallelExecutor:    parallelExecutor,
		progressiveExecutor: progressiveExecutor,
		streamingExecutor:   streamingExecutor,
		resourceManager:     resourceManager,
		config:              config,
	}
}

// convertSearchRequest converts core.SearchRequest to query.SearchRequest
func convertSearchRequest(req SearchRequest) query.SearchRequest {
	return query.SearchRequest{
		Query:          req.Query,
		TopK:           req.TopK,
		Filter:         req.Filter,
		IncludeVectors: req.IncludeVectors,
	}
}

// convertSearchResults converts []query.SearchResult to []core.SearchResult
func convertSearchResults(results []query.SearchResult) []SearchResult {
	converted := make([]SearchResult, len(results))
	for i, r := range results {
		converted[i] = SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Metadata: r.Metadata,
		}
		if len(r.Vector) > 0 {
			vec := Vector{
				ID:       r.ID,
				Values:   r.Vector,
				Metadata: r.Metadata,
			}
			converted[i].Vector = &vec
		}
	}
	return converted
}

// OptimizedSearch performs an optimized vector similarity search
func (ovs *OptimizedVectorStore) OptimizedSearch(ctx context.Context, collection string, req SearchRequest) ([]SearchResult, error) {
	// Use resource manager to track query
	queryID := fmt.Sprintf("search_%s_%d", collection, time.Now().UnixNano())
	estimatedMemory := int64(req.TopK * 1000) // Rough estimate

	var results []SearchResult
	err := ovs.resourceManager.WithQueryContext(ctx, queryID, estimatedMemory, func(queryCtx context.Context) error {
		// Track operation with profiler
		tracker := ovs.profiler.TrackOperation("optimized_search")
		defer tracker.Finish()

		// Get collection statistics
		coll, err := ovs.persistence.LoadCollection(queryCtx, collection)
		if err != nil {
			tracker.FinishWithError()
			return fmt.Errorf("collection %s not found: %w", collection, err)
		}

		// Get or update collection stats in query planner
		collStats := &query.CollectionStats{
			Name:        collection,
			VectorCount: 0, // Would be populated from actual data
			Dimension:   coll.Dimension,
			IndexType:   coll.IndexType,
		}

		// Get collection size for stats
		if size, err := ovs.GetCollectionSize(queryCtx, collection); err == nil {
			collStats.VectorCount = size
		}

		ovs.queryPlanner.UpdateCollectionStats(collStats)

		// Convert request to query package type
		queryReq := convertSearchRequest(req)

		// Get optimized query plan
		plan, err := ovs.queryPlanner.GetPlan(collection, queryReq, collStats)
		if err != nil {
			tracker.FinishWithError()
			return fmt.Errorf("failed to create query plan: %w", err)
		}

		// Execute based on configuration and plan
		startTime := time.Now()

		if ovs.config.EnableProgressiveSearch && plan.UseFastPath {
			// Use progressive search for fast-path queries
			queryResults, err := ovs.progressiveExecutor.ExecuteProgressive(queryCtx, collection, queryReq)
			if err == nil {
				results = convertSearchResults(queryResults)
			}
			return err
		} else if ovs.config.EnableParallelExecution && plan.ParallelDegree > 1 {
			// Use parallel execution for complex queries
			queryResults, err := ovs.parallelExecutor.ExecuteParallel(queryCtx, plan, queryReq)
			if err == nil {
				results = convertSearchResults(queryResults)
			}
			return err
		} else {
			// Fall back to standard search
			results, err = ovs.VectorStoreImpl.Search(queryCtx, collection, req)
			return err
		}

		// Record execution metrics
		if err == nil {
			latency := time.Since(startTime)
			ovs.queryPlanner.RecordExecution(plan, latency, len(results))
		}

		return err
	})

	return results, err
}

// StreamingSearch performs a search with streaming results
func (ovs *OptimizedVectorStore) StreamingSearch(ctx context.Context, collection string, req SearchRequest) (query.ResultStream, error) {
	if !ovs.config.EnableStreamingResults {
		return nil, fmt.Errorf("streaming results are not enabled")
	}

	queryID := fmt.Sprintf("stream_%s_%d", collection, time.Now().UnixNano())
	estimatedMemory := int64(req.TopK * 1000)

	// Acquire query context
	qc, err := ovs.resourceManager.AcquireQueryContext(ctx, queryID, estimatedMemory)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire query context: %w", err)
	}

	// Create streaming query
	streamingQuery := query.StreamingQuery{
		Request:       convertSearchRequest(req),
		BatchSize:     100,
		FlushInterval: 50 * time.Millisecond,
		MaxBufferSize: 1000,
	}

	// Execute streaming search
	stream, err := ovs.streamingExecutor.ExecuteStreaming(qc.Context, collection, streamingQuery)
	if err != nil {
		ovs.resourceManager.ReleaseQueryContext(queryID, query.QueryStateFailed)
		return nil, err
	}

	// Wrap stream to ensure cleanup
	return &managedResultStream{
		ResultStream:    stream,
		resourceManager: ovs.resourceManager,
		queryID:         queryID,
	}, nil
}

// managedResultStream wraps a result stream with resource cleanup
type managedResultStream struct {
	query.ResultStream
	resourceManager *query.ResourceManager
	queryID         string
	once            sync.Once
}

// Close ensures proper resource cleanup
func (s *managedResultStream) Close() error {
	var err error
	s.once.Do(func() {
		err = s.ResultStream.Close()
		s.resourceManager.ReleaseQueryContext(s.queryID, query.QueryStateCompleted)
	})
	return err
}

// SetProfiler sets the profiler for performance monitoring
func (ovs *OptimizedVectorStore) SetProfiler(profiler *performance.Profiler) {
	ovs.VectorStoreImpl.SetProfiler(profiler)
}

// GetQueryMetrics returns query optimization metrics
func (ovs *OptimizedVectorStore) GetQueryMetrics() QueryOptimizationMetrics {
	executorMetrics := ovs.parallelExecutor.GetMetrics()
	resourceMetrics := ovs.resourceManager.GetMetrics()

	return QueryOptimizationMetrics{
		QueriesExecuted:      executorMetrics.QueriesExecuted,
		ParallelQueries:      executorMetrics.ParallelQueries,
		CancelledQueries:     resourceMetrics.CancelledQueries,
		ThrottledQueries:     resourceMetrics.ThrottledQueries,
		AverageQueryTime:     time.Duration(0), // Would calculate from metrics
		WorkerUtilization:    executorMetrics.WorkerUtilization,
		MemoryPressureEvents: resourceMetrics.MemoryPressureEvents,
		PeakMemoryUsage:      resourceMetrics.PeakMemoryUsage,
	}
}

// QueryOptimizationMetrics contains query optimization performance metrics
type QueryOptimizationMetrics struct {
	QueriesExecuted      int64
	ParallelQueries      int64
	CancelledQueries     int64
	ThrottledQueries     int64
	AverageQueryTime     time.Duration
	WorkerUtilization    map[string]float64
	MemoryPressureEvents int64
	PeakMemoryUsage      int64
}

// Shutdown gracefully shuts down the optimized vector store
func (ovs *OptimizedVectorStore) Shutdown() error {
	// Shutdown parallel executor
	ovs.parallelExecutor.Shutdown()

	// Close base vector store
	return ovs.VectorStoreImpl.Close()
}
