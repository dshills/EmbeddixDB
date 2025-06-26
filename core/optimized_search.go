package core

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core/cache"
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
	cacheManager        *cache.MultiLevelCacheManager
	config              OptimizationConfig
	mu                  sync.RWMutex
}

// OptimizationConfig configures query optimization behavior
type OptimizationConfig struct {
	EnableParallelExecution bool
	EnableProgressiveSearch bool
	EnableStreamingResults  bool
	EnableQueryPlanCaching  bool
	EnableMultiLevelCache   bool
	MaxConcurrentQueries    int
	MaxMemoryBytes          int64
	CacheSizeMB             int
	ParallelWorkers         int
	CacheConfig             cache.CacheManagerConfig
}

// DefaultOptimizationConfig returns default optimization settings
func DefaultOptimizationConfig() OptimizationConfig {
	return OptimizationConfig{
		EnableParallelExecution: true,
		EnableProgressiveSearch: true,
		EnableStreamingResults:  false, // Disabled by default for backward compatibility
		EnableQueryPlanCaching:  true,
		EnableMultiLevelCache:   true,
		MaxConcurrentQueries:    100,
		MaxMemoryBytes:          1 << 30, // 1GB
		CacheSizeMB:             100,
		ParallelWorkers:         8,
		CacheConfig:             cache.DefaultCacheManagerConfig(),
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

	// Create cache manager if enabled
	var cacheManager *cache.MultiLevelCacheManager
	if config.EnableMultiLevelCache {
		cacheManager = cache.NewMultiLevelCacheManager(config.CacheConfig)
	}

	return &OptimizedVectorStore{
		VectorStoreImpl:     base,
		queryPlanner:        queryPlanner,
		parallelExecutor:    parallelExecutor,
		progressiveExecutor: progressiveExecutor,
		streamingExecutor:   streamingExecutor,
		resourceManager:     resourceManager,
		cacheManager:        cacheManager,
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

	// Check cache first if enabled
	var cacheKey cache.QueryCacheKey
	if ovs.cacheManager != nil && ovs.config.EnableMultiLevelCache {
		cacheKey = cache.QueryCacheKey{
			Collection:     collection,
			QueryVector:    req.Query,
			TopK:           req.TopK,
			Filters:        req.Filter,
			DistanceMetric: string(req.DistanceMetric),
			UserID:         req.UserID,
			IncludeVectors: req.IncludeVectors,
		}
		
		// Try to get from cache with semantic matching
		if cachedResults, found := ovs.cacheManager.GetQueryResult(ctx, cacheKey, 0.85); found {
			// Convert cache.SearchResult to core.SearchResult
			results := make([]SearchResult, len(cachedResults))
			for i, cr := range cachedResults {
				results[i] = SearchResult{
					ID:       cr.ID,
					Score:    cr.Score,
					Metadata: cr.Metadata,
				}
				if cr.Vector != nil {
					if vec, ok := cr.Vector.(*cache.CachedVector); ok {
						results[i].Vector = &Vector{
							ID:       vec.ID,
							Values:   vec.Values,
							Metadata: vec.Metadata,
						}
					}
				}
			}
			return results, nil
		}
	}

	var results []SearchResult
	var err error
	err = ovs.resourceManager.WithQueryContext(ctx, queryID, estimatedMemory, func(queryCtx context.Context) error {
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

		var queryErr error
		if ovs.config.EnableProgressiveSearch && plan.UseFastPath {
			// Use progressive search for fast-path queries
			queryResults, queryErr := ovs.progressiveExecutor.ExecuteProgressive(queryCtx, collection, queryReq)
			if queryErr == nil {
				results = convertSearchResults(queryResults)
			}
		} else if ovs.config.EnableParallelExecution && plan.ParallelDegree > 1 {
			// Use parallel execution for complex queries
			queryResults, queryErr := ovs.parallelExecutor.ExecuteParallel(queryCtx, plan, queryReq)
			if queryErr == nil {
				results = convertSearchResults(queryResults)
			}
		} else {
			// Fall back to standard search
			results, queryErr = ovs.VectorStoreImpl.Search(queryCtx, collection, req)
		}

		// Record execution metrics
		if queryErr == nil {
			latency := time.Since(startTime)
			ovs.queryPlanner.RecordExecution(plan, latency, len(results))
		}

		return queryErr
	})

	// Cache the results if successful and cache is enabled
	if err == nil && ovs.cacheManager != nil && ovs.config.EnableMultiLevelCache && len(results) > 0 {
		// Convert core.SearchResult to cache.SearchResult
		cacheResults := make([]cache.SearchResult, len(results))
		for i, r := range results {
			cacheResults[i] = cache.SearchResult{
				ID:       r.ID,
				Score:    r.Score,
				Metadata: r.Metadata,
			}
			if r.Vector != nil {
				cacheResults[i].Vector = &cache.CachedVector{
					ID:         r.Vector.ID,
					Values:     r.Vector.Values,
					Metadata:   r.Vector.Metadata,
					Collection: collection,
				}
			}
		}
		
		ttl := 5 * time.Minute // Default TTL
		go ovs.cacheManager.SetQueryResult(context.Background(), cacheKey, cacheResults, ttl) // Async cache update
	}

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

	metrics := QueryOptimizationMetrics{
		QueriesExecuted:      executorMetrics.QueriesExecuted,
		ParallelQueries:      executorMetrics.ParallelQueries,
		CancelledQueries:     resourceMetrics.CancelledQueries,
		ThrottledQueries:     resourceMetrics.ThrottledQueries,
		AverageQueryTime:     time.Duration(0), // Would calculate from metrics
		WorkerUtilization:    executorMetrics.WorkerUtilization,
		MemoryPressureEvents: resourceMetrics.MemoryPressureEvents,
		PeakMemoryUsage:      resourceMetrics.PeakMemoryUsage,
	}

	// Add cache metrics if available
	if ovs.cacheManager != nil {
		cacheStats := ovs.cacheManager.GetStats()
		metrics.CacheStats = cacheStats
	}

	return metrics
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
	CacheStats           map[cache.CacheLevel]cache.CacheStats
}

// Shutdown gracefully shuts down the optimized vector store
func (ovs *OptimizedVectorStore) Shutdown() error {
	// Shutdown parallel executor
	ovs.parallelExecutor.Shutdown()

	// Close cache manager if present
	if ovs.cacheManager != nil {
		ovs.cacheManager.Close()
	}

	// Close base vector store
	return ovs.VectorStoreImpl.Close()
}
