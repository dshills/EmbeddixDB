package hierarchical

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// BackgroundOptimizer manages background optimization tasks for hierarchical indexes
type BackgroundOptimizer struct {
	mu               sync.RWMutex
	config           OptimizerConfig
	tasks            []OptimizationTask
	running          bool
	ctx              context.Context
	cancel           context.CancelFunc
	metrics          *OptimizationMetrics
	lastOptimization map[string]time.Time
}

// OptimizerConfig configures the background optimizer
type OptimizerConfig struct {
	// Task scheduling
	EnableOptimization      bool          `json:"enable_optimization"`
	OptimizationInterval    time.Duration `json:"optimization_interval"`
	MinTimeBetweenTasks     time.Duration `json:"min_time_between_tasks"`
	
	// Task priorities
	ClusterQualityWeight    float64       `json:"cluster_quality_weight"`
	GraphConnectivityWeight float64       `json:"graph_connectivity_weight"`
	MemoryEfficiencyWeight  float64       `json:"memory_efficiency_weight"`
	
	// Resource limits
	MaxCPUPercent           float64       `json:"max_cpu_percent"`
	MaxMemoryMB             int64         `json:"max_memory_mb"`
	MaxConcurrentTasks      int           `json:"max_concurrent_tasks"`
}

// OptimizationTask represents a background optimization task
type OptimizationTask struct {
	ID          string
	Type        TaskType
	Priority    float64
	Created     time.Time
	Started     time.Time
	Completed   time.Time
	Status      TaskStatus
	Error       error
	Impact      *TaskImpact
}

// TaskType represents the type of optimization task
type TaskType string

const (
	TaskTypeClusterOptimization  TaskType = "cluster_optimization"
	TaskTypeGraphRepair          TaskType = "graph_repair"
	TaskTypeMemoryCompaction     TaskType = "memory_compaction"
	TaskTypeCentroidRefinement   TaskType = "centroid_refinement"
	TaskTypeEdgePruning          TaskType = "edge_pruning"
)

// TaskStatus represents the status of an optimization task
type TaskStatus string

const (
	TaskStatusPending   TaskStatus = "pending"
	TaskStatusRunning   TaskStatus = "running"
	TaskStatusCompleted TaskStatus = "completed"
	TaskStatusFailed    TaskStatus = "failed"
	TaskStatusCancelled TaskStatus = "cancelled"
)

// TaskImpact measures the impact of an optimization task
type TaskImpact struct {
	VectorsProcessed   int
	MemoryFreedMB      int64
	QualityImprovement float64
	Duration           time.Duration
}

// OptimizationMetrics tracks optimization performance
type OptimizationMetrics struct {
	mu                    sync.RWMutex
	TasksCompleted        int64
	TasksFailed           int64
	TotalOptimizationTime time.Duration
	MemoryFreedTotal      int64
	LastOptimization      time.Time
	CurrentTasks          []string
}

// NewBackgroundOptimizer creates a new background optimizer
func NewBackgroundOptimizer(config OptimizerConfig) *BackgroundOptimizer {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &BackgroundOptimizer{
		config:           config,
		tasks:            make([]OptimizationTask, 0),
		ctx:              ctx,
		cancel:           cancel,
		metrics:          &OptimizationMetrics{},
		lastOptimization: make(map[string]time.Time),
	}
}

// Start begins background optimization
func (bo *BackgroundOptimizer) Start(index OptimizableIndex) error {
	bo.mu.Lock()
	defer bo.mu.Unlock()

	if bo.running {
		return fmt.Errorf("optimizer already running")
	}

	if !bo.config.EnableOptimization {
		return fmt.Errorf("optimization is disabled")
	}

	bo.running = true
	go bo.runOptimizationLoop(index)

	return nil
}

// Stop stops background optimization
func (bo *BackgroundOptimizer) Stop() {
	bo.mu.Lock()
	defer bo.mu.Unlock()

	if bo.running {
		bo.cancel()
		bo.running = false
	}
}

// runOptimizationLoop runs the main optimization loop
func (bo *BackgroundOptimizer) runOptimizationLoop(index OptimizableIndex) {
	ticker := time.NewTicker(bo.config.OptimizationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-bo.ctx.Done():
			return
		case <-ticker.C:
			bo.performOptimizationCycle(index)
		}
	}
}

// performOptimizationCycle performs one optimization cycle
func (bo *BackgroundOptimizer) performOptimizationCycle(index OptimizableIndex) {
	// Identify needed optimizations
	tasks := bo.identifyOptimizationTasks(index)
	if len(tasks) == 0 {
		return
	}

	// Sort by priority
	bo.sortTasksByPriority(tasks)

	// Execute tasks respecting concurrency limits
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, bo.config.MaxConcurrentTasks)

	for _, task := range tasks {
		// Check if enough time has passed since last similar task
		if !bo.canRunTask(task) {
			continue
		}

		wg.Add(1)
		go func(t OptimizationTask) {
			defer wg.Done()
			
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			bo.executeTask(&t, index)
		}(task)
	}

	wg.Wait()
}

// identifyOptimizationTasks identifies tasks that need to be performed
func (bo *BackgroundOptimizer) identifyOptimizationTasks(index OptimizableIndex) []OptimizationTask {
	tasks := make([]OptimizationTask, 0)

	// Check cluster quality
	if quality := index.GetClusterQuality(); quality < 0.8 {
		tasks = append(tasks, OptimizationTask{
			ID:       fmt.Sprintf("cluster_opt_%d", time.Now().Unix()),
			Type:     TaskTypeClusterOptimization,
			Priority: (1.0 - quality) * bo.config.ClusterQualityWeight,
			Created:  time.Now(),
			Status:   TaskStatusPending,
		})
	}

	// Check graph connectivity
	if connectivity := index.GetGraphConnectivity(); connectivity < 0.9 {
		tasks = append(tasks, OptimizationTask{
			ID:       fmt.Sprintf("graph_repair_%d", time.Now().Unix()),
			Type:     TaskTypeGraphRepair,
			Priority: (1.0 - connectivity) * bo.config.GraphConnectivityWeight,
			Created:  time.Now(),
			Status:   TaskStatusPending,
		})
	}

	// Check memory efficiency
	if efficiency := index.GetMemoryEfficiency(); efficiency < 0.7 {
		tasks = append(tasks, OptimizationTask{
			ID:       fmt.Sprintf("mem_compact_%d", time.Now().Unix()),
			Type:     TaskTypeMemoryCompaction,
			Priority: (1.0 - efficiency) * bo.config.MemoryEfficiencyWeight,
			Created:  time.Now(),
			Status:   TaskStatusPending,
		})
	}

	// Periodic centroid refinement
	if bo.shouldRefineCentroids() {
		tasks = append(tasks, OptimizationTask{
			ID:       fmt.Sprintf("centroid_refine_%d", time.Now().Unix()),
			Type:     TaskTypeCentroidRefinement,
			Priority: 0.5,
			Created:  time.Now(),
			Status:   TaskStatusPending,
		})
	}

	// Edge pruning for large graphs
	if index.Size() > 100000 && bo.shouldPruneEdges() {
		tasks = append(tasks, OptimizationTask{
			ID:       fmt.Sprintf("edge_prune_%d", time.Now().Unix()),
			Type:     TaskTypeEdgePruning,
			Priority: 0.3,
			Created:  time.Now(),
			Status:   TaskStatusPending,
		})
	}

	return tasks
}

// executeTask executes a single optimization task
func (bo *BackgroundOptimizer) executeTask(task *OptimizationTask, index OptimizableIndex) {
	bo.updateTaskStatus(task, TaskStatusRunning)
	task.Started = time.Now()

	// Update metrics
	bo.metrics.mu.Lock()
	bo.metrics.CurrentTasks = append(bo.metrics.CurrentTasks, task.ID)
	bo.metrics.mu.Unlock()

	defer func() {
		bo.metrics.mu.Lock()
		// Remove from current tasks
		for i, id := range bo.metrics.CurrentTasks {
			if id == task.ID {
				bo.metrics.CurrentTasks = append(
					bo.metrics.CurrentTasks[:i],
					bo.metrics.CurrentTasks[i+1:]...,
				)
				break
			}
		}
		bo.metrics.mu.Unlock()
	}()

	var err error
	impact := &TaskImpact{}

	switch task.Type {
	case TaskTypeClusterOptimization:
		err = bo.optimizeClusters(index, impact)
	case TaskTypeGraphRepair:
		err = bo.repairGraph(index, impact)
	case TaskTypeMemoryCompaction:
		err = bo.compactMemory(index, impact)
	case TaskTypeCentroidRefinement:
		err = bo.refineCentroids(index, impact)
	case TaskTypeEdgePruning:
		err = bo.pruneEdges(index, impact)
	default:
		err = fmt.Errorf("unknown task type: %s", task.Type)
	}

	task.Completed = time.Now()
	impact.Duration = task.Completed.Sub(task.Started)
	task.Impact = impact

	if err != nil {
		task.Error = err
		task.Status = TaskStatusFailed
		bo.metrics.mu.Lock()
		bo.metrics.TasksFailed++
		bo.metrics.mu.Unlock()
	} else {
		task.Status = TaskStatusCompleted
		bo.metrics.mu.Lock()
		bo.metrics.TasksCompleted++
		bo.metrics.TotalOptimizationTime += impact.Duration
		bo.metrics.MemoryFreedTotal += impact.MemoryFreedMB
		bo.metrics.LastOptimization = time.Now()
		bo.metrics.mu.Unlock()
		
		// Update last optimization time for this task type
		bo.mu.Lock()
		bo.lastOptimization[string(task.Type)] = time.Now()
		bo.mu.Unlock()
	}

	bo.recordTask(*task)
}

// optimizeClusters optimizes cluster assignments
func (bo *BackgroundOptimizer) optimizeClusters(index OptimizableIndex, impact *TaskImpact) error {
	// Get current cluster assignments
	vectors, assignments := index.GetClusterAssignments()
	if len(vectors) == 0 {
		return nil
	}

	// Re-cluster with improved settings
	clusterer := NewKMeansClusterer(
		len(assignments),
		len(vectors[0].Values),
		index.GetDistanceMetric(),
		WithMaxIterations(50),
		WithBalanceFactor(0.7),
	)

	result, err := clusterer.Cluster(vectors)
	if err != nil {
		return fmt.Errorf("clustering failed: %w", err)
	}

	// Apply new assignments if quality improved
	newQuality := clusterer.GetClusterQuality(result, vectors)
	if newQuality.OverallSilhouette > index.GetClusterQuality() {
		if err := index.UpdateClusterAssignments(result.Assignments); err != nil {
			return fmt.Errorf("failed to update assignments: %w", err)
		}
		
		impact.VectorsProcessed = len(vectors)
		impact.QualityImprovement = newQuality.OverallSilhouette - index.GetClusterQuality()
	}

	return nil
}

// repairGraph repairs disconnected components in the graph
func (bo *BackgroundOptimizer) repairGraph(index OptimizableIndex, impact *TaskImpact) error {
	disconnected := index.FindDisconnectedComponents()
	if len(disconnected) == 0 {
		return nil
	}

	repaired := 0
	for _, component := range disconnected {
		if err := index.RepairComponent(component); err != nil {
			continue // Log but continue with other components
		}
		repaired++
	}

	impact.VectorsProcessed = repaired
	return nil
}

// compactMemory compacts memory usage
func (bo *BackgroundOptimizer) compactMemory(index OptimizableIndex, impact *TaskImpact) error {
	initialMemory := index.GetMemoryUsage()
	
	if err := index.CompactMemory(); err != nil {
		return fmt.Errorf("memory compaction failed: %w", err)
	}

	finalMemory := index.GetMemoryUsage()
	impact.MemoryFreedMB = (initialMemory - finalMemory) / (1024 * 1024)
	
	return nil
}

// refineCentroids refines cluster centroids
func (bo *BackgroundOptimizer) refineCentroids(index OptimizableIndex, impact *TaskImpact) error {
	clusters := index.GetClusters()
	refined := 0

	for _, cluster := range clusters {
		oldCentroid := cluster.Centroid
		newCentroid := index.ComputeCentroid(cluster.Members)
		
		// Update if centroid moved significantly
		distance, _ := core.EuclideanDistance(oldCentroid, newCentroid)
		if distance > 0.01 {
			if err := index.UpdateCentroid(cluster.ID, newCentroid); err != nil {
				continue
			}
			refined++
		}
	}

	impact.VectorsProcessed = refined
	return nil
}

// pruneEdges removes unnecessary edges from the graph
func (bo *BackgroundOptimizer) pruneEdges(index OptimizableIndex, impact *TaskImpact) error {
	prunedCount, err := index.PruneRedundantEdges()
	if err != nil {
		return fmt.Errorf("edge pruning failed: %w", err)
	}

	impact.VectorsProcessed = prunedCount
	// Estimate memory saved (rough approximation)
	impact.MemoryFreedMB = int64(prunedCount * 8 / 1024 / 1024) // 8 bytes per edge
	
	return nil
}

// Helper methods

func (bo *BackgroundOptimizer) canRunTask(task OptimizationTask) bool {
	bo.mu.RLock()
	defer bo.mu.RUnlock()

	lastRun, exists := bo.lastOptimization[string(task.Type)]
	if !exists {
		return true
	}

	return time.Since(lastRun) >= bo.config.MinTimeBetweenTasks
}

func (bo *BackgroundOptimizer) shouldRefineCentroids() bool {
	bo.mu.RLock()
	defer bo.mu.RUnlock()

	lastRun, exists := bo.lastOptimization[string(TaskTypeCentroidRefinement)]
	if !exists {
		return true
	}

	// Refine centroids every hour
	return time.Since(lastRun) >= time.Hour
}

func (bo *BackgroundOptimizer) shouldPruneEdges() bool {
	bo.mu.RLock()
	defer bo.mu.RUnlock()

	lastRun, exists := bo.lastOptimization[string(TaskTypeEdgePruning)]
	if !exists {
		return true
	}

	// Prune edges every 6 hours
	return time.Since(lastRun) >= 6*time.Hour
}

func (bo *BackgroundOptimizer) sortTasksByPriority(tasks []OptimizationTask) {
	for i := 0; i < len(tasks); i++ {
		for j := i + 1; j < len(tasks); j++ {
			if tasks[j].Priority > tasks[i].Priority {
				tasks[i], tasks[j] = tasks[j], tasks[i]
			}
		}
	}
}

func (bo *BackgroundOptimizer) updateTaskStatus(task *OptimizationTask, status TaskStatus) {
	bo.mu.Lock()
	defer bo.mu.Unlock()
	task.Status = status
}

func (bo *BackgroundOptimizer) recordTask(task OptimizationTask) {
	bo.mu.Lock()
	defer bo.mu.Unlock()
	
	bo.tasks = append(bo.tasks, task)
	
	// Keep only recent tasks
	if len(bo.tasks) > 1000 {
		bo.tasks = bo.tasks[len(bo.tasks)-1000:]
	}
}

// GetMetrics returns current optimization metrics
func (bo *BackgroundOptimizer) GetMetrics() OptimizationMetrics {
	bo.metrics.mu.RLock()
	defer bo.metrics.mu.RUnlock()
	
	return *bo.metrics
}

// GetRecentTasks returns recent optimization tasks
func (bo *BackgroundOptimizer) GetRecentTasks(limit int) []OptimizationTask {
	bo.mu.RLock()
	defer bo.mu.RUnlock()
	
	start := len(bo.tasks) - limit
	if start < 0 {
		start = 0
	}
	
	result := make([]OptimizationTask, len(bo.tasks[start:]))
	copy(result, bo.tasks[start:])
	return result
}

// OptimizableIndex interface for indexes that support optimization
type OptimizableIndex interface {
	// Cluster operations
	GetClusterQuality() float64
	GetClusterAssignments() ([]core.Vector, map[string]int)
	UpdateClusterAssignments(assignments map[string]int) error
	GetClusters() []Cluster
	ComputeCentroid(memberIDs []string) []float32
	UpdateCentroid(clusterID int, centroid []float32) error
	
	// Graph operations
	GetGraphConnectivity() float64
	FindDisconnectedComponents() [][]string
	RepairComponent(component []string) error
	PruneRedundantEdges() (int, error)
	
	// Memory operations
	GetMemoryEfficiency() float64
	GetMemoryUsage() int64
	CompactMemory() error
	
	// General operations
	Size() int
	GetDistanceMetric() core.DistanceMetric
}