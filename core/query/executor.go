package query

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ParallelExecutor manages parallel query execution with worker pools
type ParallelExecutor struct {
	mu          sync.RWMutex
	workerPools map[string]*WorkerPool
	maxWorkers  int
	queueSize   int
	metrics     *ExecutorMetrics
}

// WorkerPool manages a pool of workers for parallel execution
type WorkerPool struct {
	name       string
	workers    int
	taskQueue  chan Task
	resultChan chan TaskResult
	stopChan   chan struct{}
	wg         sync.WaitGroup
	metrics    *PoolMetrics
}

// Task represents a unit of work for parallel execution
type Task struct {
	ID         string
	Type       TaskType
	Priority   int
	Context    context.Context
	ExecuteFn  func(context.Context) (interface{}, error)
	ResultChan chan<- TaskResult
}

// TaskResult contains the result of a task execution
type TaskResult struct {
	TaskID   string
	Result   interface{}
	Error    error
	Duration time.Duration
}

// TaskType defines the type of task
type TaskType string

const (
	TaskTypeVectorSearch TaskType = "vector_search"
	TaskTypeIndexScan    TaskType = "index_scan"
	TaskTypeFilter       TaskType = "filter"
	TaskTypeMerge        TaskType = "merge"
)

// PoolMetrics tracks worker pool performance
type PoolMetrics struct {
	mu               sync.RWMutex
	TasksCompleted   int64
	TasksFailed      int64
	TasksQueued      int64
	ActiveWorkers    int32
	TotalWaitTime    time.Duration
	TotalExecuteTime time.Duration
}

// ExecutorMetrics tracks overall executor performance
type ExecutorMetrics struct {
	mu                sync.RWMutex
	QueriesExecuted   int64
	ParallelQueries   int64
	TotalQueryTime    time.Duration
	WorkerUtilization map[string]float64
}

// NewParallelExecutor creates a new parallel query executor
func NewParallelExecutor(maxWorkers, queueSize int) *ParallelExecutor {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	if queueSize <= 0 {
		queueSize = maxWorkers * 10
	}

	return &ParallelExecutor{
		workerPools: make(map[string]*WorkerPool),
		maxWorkers:  maxWorkers,
		queueSize:   queueSize,
		metrics: &ExecutorMetrics{
			WorkerUtilization: make(map[string]float64),
		},
	}
}

// GetOrCreatePool gets an existing pool or creates a new one
func (e *ParallelExecutor) GetOrCreatePool(name string, workers int) *WorkerPool {
	e.mu.RLock()
	pool, exists := e.workerPools[name]
	e.mu.RUnlock()

	if exists {
		return pool
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Double-check after acquiring write lock
	if pool, exists = e.workerPools[name]; exists {
		return pool
	}

	// Create new pool
	pool = &WorkerPool{
		name:       name,
		workers:    workers,
		taskQueue:  make(chan Task, e.queueSize),
		resultChan: make(chan TaskResult, e.queueSize),
		stopChan:   make(chan struct{}),
		metrics:    &PoolMetrics{},
	}

	// Start workers
	pool.Start()

	e.workerPools[name] = pool
	return pool
}

// ExecuteParallel executes a query plan in parallel
func (e *ParallelExecutor) ExecuteParallel(ctx context.Context, plan *QueryPlan, req SearchRequest) ([]SearchResult, error) {
	if !plan.UseFastPath || plan.ParallelDegree <= 1 {
		// Fall back to sequential execution
		return e.executeSequential(ctx, plan, req)
	}

	atomic.AddInt64(&e.metrics.ParallelQueries, 1)
	startTime := time.Now()
	defer func() {
		atomic.AddInt64(&e.metrics.QueriesExecuted, 1)
		e.metrics.mu.Lock()
		e.metrics.TotalQueryTime += time.Since(startTime)
		e.metrics.mu.Unlock()
	}()

	// Get appropriate worker pool
	pool := e.GetOrCreatePool("search_pool", plan.ParallelDegree)

	// Split work into parallel tasks
	tasks, err := e.createParallelTasks(ctx, plan, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create parallel tasks: %w", err)
	}

	// Submit tasks to pool
	resultChan := make(chan TaskResult, len(tasks))
	for _, task := range tasks {
		task.ResultChan = resultChan
		select {
		case pool.taskQueue <- task:
			atomic.AddInt64(&pool.metrics.TasksQueued, 1)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Collect results
	results := make([]SearchResult, 0)
	errors := make([]error, 0)

	for i := 0; i < len(tasks); i++ {
		select {
		case result := <-resultChan:
			if result.Error != nil {
				errors = append(errors, result.Error)
			} else if searchResults, ok := result.Result.([]SearchResult); ok {
				results = append(results, searchResults...)
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Check for errors
	if len(errors) > 0 {
		return nil, fmt.Errorf("parallel execution failed with %d errors: %v", len(errors), errors[0])
	}

	// Merge and sort results
	return e.mergeResults(results, req.TopK), nil
}

// executeSequential performs sequential execution as fallback
func (e *ParallelExecutor) executeSequential(ctx context.Context, plan *QueryPlan, req SearchRequest) ([]SearchResult, error) {
	// This would be implemented to execute the query sequentially
	// For now, returning empty results as placeholder
	return []SearchResult{}, nil
}

// createParallelTasks splits the search into parallel tasks
func (e *ParallelExecutor) createParallelTasks(ctx context.Context, plan *QueryPlan, req SearchRequest) ([]Task, error) {
	tasks := make([]Task, 0, plan.ParallelDegree)

	// For vector search, we can partition the search space
	for i := 0; i < plan.ParallelDegree; i++ {
		taskID := fmt.Sprintf("search_%s_%d", plan.ID, i)

		task := Task{
			ID:       taskID,
			Type:     TaskTypeVectorSearch,
			Priority: 1,
			Context:  ctx,
			ExecuteFn: func(taskCtx context.Context) (interface{}, error) {
				// This would execute a portion of the search
				// For now, returning empty results as placeholder
				return []SearchResult{}, nil
			},
		}

		tasks = append(tasks, task)
	}

	return tasks, nil
}

// mergeResults merges and sorts results from parallel tasks
func (e *ParallelExecutor) mergeResults(results []SearchResult, topK int) []SearchResult {
	// Sort all results by score
	// This is a simplified version - real implementation would use a heap
	if len(results) <= topK {
		return results
	}
	return results[:topK]
}

// Start begins the worker pool execution
func (p *WorkerPool) Start() {
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}
}

// Stop gracefully shuts down the worker pool
func (p *WorkerPool) Stop() {
	close(p.stopChan)
	p.wg.Wait()
	close(p.taskQueue)
	close(p.resultChan)
}

// worker is the main worker goroutine
func (p *WorkerPool) worker(id int) {
	defer p.wg.Done()
	atomic.AddInt32(&p.metrics.ActiveWorkers, 1)
	defer atomic.AddInt32(&p.metrics.ActiveWorkers, -1)

	for {
		select {
		case task, ok := <-p.taskQueue:
			if !ok {
				return
			}
			p.executeTask(task)
		case <-p.stopChan:
			return
		}
	}
}

// executeTask executes a single task
func (p *WorkerPool) executeTask(task Task) {
	startTime := time.Now()
	waitTime := startTime.Sub(time.Now()) // This would track actual queue wait time

	p.metrics.mu.Lock()
	p.metrics.TotalWaitTime += waitTime
	p.metrics.mu.Unlock()

	// Execute the task
	result, err := task.ExecuteFn(task.Context)
	duration := time.Since(startTime)

	p.metrics.mu.Lock()
	p.metrics.TotalExecuteTime += duration
	p.metrics.mu.Unlock()

	// Update metrics
	if err != nil {
		atomic.AddInt64(&p.metrics.TasksFailed, 1)
	} else {
		atomic.AddInt64(&p.metrics.TasksCompleted, 1)
	}

	// Send result
	if task.ResultChan != nil {
		task.ResultChan <- TaskResult{
			TaskID:   task.ID,
			Result:   result,
			Error:    err,
			Duration: duration,
		}
	}
}

// GetMetrics returns current pool metrics
func (p *WorkerPool) GetMetrics() PoolMetrics {
	p.metrics.mu.RLock()
	defer p.metrics.mu.RUnlock()

	return PoolMetrics{
		TasksCompleted:   atomic.LoadInt64(&p.metrics.TasksCompleted),
		TasksFailed:      atomic.LoadInt64(&p.metrics.TasksFailed),
		TasksQueued:      atomic.LoadInt64(&p.metrics.TasksQueued),
		ActiveWorkers:    atomic.LoadInt32(&p.metrics.ActiveWorkers),
		TotalWaitTime:    p.metrics.TotalWaitTime,
		TotalExecuteTime: p.metrics.TotalExecuteTime,
	}
}

// GetMetrics returns executor metrics
func (e *ParallelExecutor) GetMetrics() *ExecutorMetrics {
	e.metrics.mu.RLock()
	defer e.metrics.mu.RUnlock()

	metrics := &ExecutorMetrics{
		QueriesExecuted:   atomic.LoadInt64(&e.metrics.QueriesExecuted),
		ParallelQueries:   atomic.LoadInt64(&e.metrics.ParallelQueries),
		TotalQueryTime:    e.metrics.TotalQueryTime,
		WorkerUtilization: make(map[string]float64),
	}

	// Calculate worker utilization
	for name, pool := range e.workerPools {
		poolMetrics := pool.GetMetrics()
		if poolMetrics.TasksCompleted > 0 {
			utilization := float64(poolMetrics.ActiveWorkers) / float64(pool.workers)
			metrics.WorkerUtilization[name] = utilization
		}
	}

	return metrics
}

// Shutdown gracefully shuts down all worker pools
func (e *ParallelExecutor) Shutdown() {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, pool := range e.workerPools {
		pool.Stop()
	}

	e.workerPools = make(map[string]*WorkerPool)
}
