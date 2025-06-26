package performance

import (
	"fmt"
	"net/http"
	_ "net/http/pprof" // Register pprof handlers
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"sync"
	"time"
)

// Profiler manages performance profiling and monitoring
type Profiler struct {
	mu              sync.RWMutex
	enabled         bool
	profileDir      string
	cpuProfile      *os.File
	memProfile      *os.File
	blockProfile    *os.File
	goroutineProfile *os.File
	
	// Metrics collection
	metrics         *Metrics
	metricsInterval time.Duration
	stopChan        chan struct{}
	wg              sync.WaitGroup
}

// Metrics holds performance metrics
type Metrics struct {
	mu                    sync.RWMutex
	StartTime            time.Time
	CPUUsage             []CPUSample
	MemoryUsage          []MemorySample
	GoroutineCount       []GoroutineSample
	GCStats              []GCSample
	OperationMetrics     map[string]*OperationStats
}

// CPUSample represents a CPU usage measurement
type CPUSample struct {
	Timestamp time.Time
	Usage     float64 // Percentage
}

// MemorySample represents memory usage at a point in time
type MemorySample struct {
	Timestamp    time.Time
	Alloc        uint64 // Bytes allocated and still in use
	TotalAlloc   uint64 // Bytes allocated (even if freed)
	Sys          uint64 // Bytes obtained from system
	NumGC        uint32 // Number of garbage collections
	HeapInuse    uint64 // Bytes in in-use spans
	HeapIdle     uint64 // Bytes in idle spans
	StackInuse   uint64 // Bytes in stack spans
	NextGC       uint64 // Next collection will happen when HeapAlloc ≥ this amount
}

// GoroutineSample represents goroutine count at a point in time
type GoroutineSample struct {
	Timestamp time.Time
	Count     int
}

// GCSample represents garbage collection statistics
type GCSample struct {
	Timestamp    time.Time
	NumGC        uint32
	PauseTotal   time.Duration
	LastPause    time.Duration
	PauseHistory []time.Duration
}

// OperationStats tracks performance of specific operations
type OperationStats struct {
	mu           sync.RWMutex
	Count        int64
	TotalTime    time.Duration
	MinTime      time.Duration
	MaxTime      time.Duration
	LastTime     time.Duration
	Errors       int64
}

// ProfilingConfig configures the profiler
type ProfilingConfig struct {
	Enabled         bool
	ProfileDir      string
	MetricsInterval time.Duration
	EnableCPU       bool
	EnableMemory    bool
	EnableBlock     bool
	EnableGoroutine bool
	EnablePprof     bool
	PprofAddr       string
}

// DefaultProfilingConfig returns a default profiling configuration
func DefaultProfilingConfig() ProfilingConfig {
	return ProfilingConfig{
		Enabled:         true,
		ProfileDir:      "./profiles",
		MetricsInterval: time.Second,
		EnableCPU:       true,
		EnableMemory:    true,
		EnableBlock:     true,
		EnableGoroutine: true,
		EnablePprof:     true,
		PprofAddr:       ":6060",
	}
}

// NewProfiler creates a new performance profiler
func NewProfiler(config ProfilingConfig) *Profiler {
	if err := os.MkdirAll(config.ProfileDir, 0755); err != nil {
		panic(fmt.Sprintf("Failed to create profile directory: %v", err))
	}
	
	p := &Profiler{
		enabled:         config.Enabled,
		profileDir:      config.ProfileDir,
		metricsInterval: config.MetricsInterval,
		stopChan:        make(chan struct{}),
		metrics: &Metrics{
			StartTime:        time.Now(),
			OperationMetrics: make(map[string]*OperationStats),
		},
	}
	
	// Enable profiling types based on config
	if config.EnableBlock {
		runtime.SetBlockProfileRate(1)
	}
	
	// Start pprof server if enabled
	if config.EnablePprof {
		go func() {
			if err := http.ListenAndServe(config.PprofAddr, nil); err != nil {
				fmt.Printf("Failed to start pprof server: %v\n", err)
			}
		}()
	}
	
	return p
}

// Start begins profiling and metrics collection
func (p *Profiler) Start() error {
	if !p.enabled {
		return nil
	}
	
	p.mu.Lock()
	defer p.mu.Unlock()
	
	// Start CPU profiling
	cpuFile, err := os.Create(fmt.Sprintf("%s/cpu_%d.prof", p.profileDir, time.Now().Unix()))
	if err != nil {
		return fmt.Errorf("failed to create CPU profile: %w", err)
	}
	p.cpuProfile = cpuFile
	
	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		return fmt.Errorf("failed to start CPU profile: %w", err)
	}
	
	// Start metrics collection
	p.wg.Add(1)
	go p.collectMetrics()
	
	return nil
}

// Stop ends profiling and saves all profiles
func (p *Profiler) Stop() error {
	if !p.enabled {
		return nil
	}
	
	p.mu.Lock()
	defer p.mu.Unlock()
	
	// Stop metrics collection
	close(p.stopChan)
	p.wg.Wait()
	
	// Stop CPU profiling
	if p.cpuProfile != nil {
		pprof.StopCPUProfile()
		p.cpuProfile.Close()
	}
	
	// Write memory profile
	if err := p.writeMemoryProfile(); err != nil {
		return fmt.Errorf("failed to write memory profile: %w", err)
	}
	
	// Write goroutine profile
	if err := p.writeGoroutineProfile(); err != nil {
		return fmt.Errorf("failed to write goroutine profile: %w", err)
	}
	
	// Write block profile
	if err := p.writeBlockProfile(); err != nil {
		return fmt.Errorf("failed to write block profile: %w", err)
	}
	
	return nil
}

// TrackOperation begins tracking a specific operation
func (p *Profiler) TrackOperation(operationName string) *OperationTracker {
	if p == nil || !p.enabled {
		return &OperationTracker{} // No-op tracker
	}
	
	p.metrics.mu.Lock()
	stats, exists := p.metrics.OperationMetrics[operationName]
	if !exists {
		stats = &OperationStats{
			MinTime: time.Hour, // Initialize to large value
		}
		p.metrics.OperationMetrics[operationName] = stats
	}
	p.metrics.mu.Unlock()
	
	return &OperationTracker{
		operationName: operationName,
		stats:         stats,
		startTime:     time.Now(),
	}
}

// OperationTracker tracks the performance of a single operation
type OperationTracker struct {
	operationName string
	stats         *OperationStats
	startTime     time.Time
	finished      bool
}

// Finish completes the operation tracking
func (ot *OperationTracker) Finish() {
	if ot.stats == nil || ot.finished {
		return // No-op tracker or already finished
	}
	
	ot.finished = true
	duration := time.Since(ot.startTime)
	
	ot.stats.mu.Lock()
	ot.stats.Count++
	ot.stats.TotalTime += duration
	ot.stats.LastTime = duration
	
	if duration < ot.stats.MinTime {
		ot.stats.MinTime = duration
	}
	if duration > ot.stats.MaxTime {
		ot.stats.MaxTime = duration
	}
	ot.stats.mu.Unlock()
}

// FinishWithError completes the operation tracking and records an error
func (ot *OperationTracker) FinishWithError() {
	if ot.stats == nil || ot.finished {
		return // No-op tracker or already finished
	}
	
	ot.finished = true
	duration := time.Since(ot.startTime)
	
	ot.stats.mu.Lock()
	ot.stats.Count++
	ot.stats.TotalTime += duration
	ot.stats.LastTime = duration
	ot.stats.Errors++
	
	if duration < ot.stats.MinTime {
		ot.stats.MinTime = duration
	}
	if duration > ot.stats.MaxTime {
		ot.stats.MaxTime = duration
	}
	ot.stats.mu.Unlock()
}

// GetMetrics returns the current performance metrics
func (p *Profiler) GetMetrics() *Metrics {
	p.metrics.mu.RLock()
	defer p.metrics.mu.RUnlock()
	
	// Create a deep copy of metrics to avoid race conditions
	metrics := &Metrics{
		StartTime:        p.metrics.StartTime,
		CPUUsage:         make([]CPUSample, len(p.metrics.CPUUsage)),
		MemoryUsage:      make([]MemorySample, len(p.metrics.MemoryUsage)),
		GoroutineCount:   make([]GoroutineSample, len(p.metrics.GoroutineCount)),
		GCStats:          make([]GCSample, len(p.metrics.GCStats)),
		OperationMetrics: make(map[string]*OperationStats),
	}
	
	copy(metrics.CPUUsage, p.metrics.CPUUsage)
	copy(metrics.MemoryUsage, p.metrics.MemoryUsage)
	copy(metrics.GoroutineCount, p.metrics.GoroutineCount)
	copy(metrics.GCStats, p.metrics.GCStats)
	
	// Copy operation metrics
	for name, stats := range p.metrics.OperationMetrics {
		stats.mu.RLock()
		metrics.OperationMetrics[name] = &OperationStats{
			Count:     stats.Count,
			TotalTime: stats.TotalTime,
			MinTime:   stats.MinTime,
			MaxTime:   stats.MaxTime,
			LastTime:  stats.LastTime,
			Errors:    stats.Errors,
		}
		stats.mu.RUnlock()
	}
	
	return metrics
}

// collectMetrics runs the metrics collection loop
func (p *Profiler) collectMetrics() {
	defer p.wg.Done()
	
	ticker := time.NewTicker(p.metricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-p.stopChan:
			return
		case <-ticker.C:
			p.sampleMetrics()
		}
	}
}

// sampleMetrics collects a single sample of system metrics
func (p *Profiler) sampleMetrics() {
	now := time.Now()
	
	// Collect memory statistics
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	memorySample := MemorySample{
		Timestamp:  now,
		Alloc:      memStats.Alloc,
		TotalAlloc: memStats.TotalAlloc,
		Sys:        memStats.Sys,
		NumGC:      memStats.NumGC,
		HeapInuse:  memStats.HeapInuse,
		HeapIdle:   memStats.HeapIdle,
		StackInuse: memStats.StackInuse,
		NextGC:     memStats.NextGC,
	}
	
	// Collect goroutine count
	goroutineSample := GoroutineSample{
		Timestamp: now,
		Count:     runtime.NumGoroutine(),
	}
	
	// Collect GC statistics
	var gcStats debug.GCStats
	debug.ReadGCStats(&gcStats)
	
	gcSample := GCSample{
		Timestamp:    now,
		NumGC:        uint32(gcStats.NumGC),
		PauseTotal:   gcStats.PauseTotal,
		LastPause:    gcStats.Pause[0],
		PauseHistory: make([]time.Duration, len(gcStats.Pause)),
	}
	copy(gcSample.PauseHistory, gcStats.Pause)
	
	// TODO: Collect CPU usage (requires platform-specific implementation)
	cpuSample := CPUSample{
		Timestamp: now,
		Usage:     0.0, // Placeholder
	}
	
	// Store samples
	p.metrics.mu.Lock()
	p.metrics.CPUUsage = append(p.metrics.CPUUsage, cpuSample)
	p.metrics.MemoryUsage = append(p.metrics.MemoryUsage, memorySample)
	p.metrics.GoroutineCount = append(p.metrics.GoroutineCount, goroutineSample)
	p.metrics.GCStats = append(p.metrics.GCStats, gcSample)
	p.metrics.mu.Unlock()
}

// writeMemoryProfile writes a memory profile to disk
func (p *Profiler) writeMemoryProfile() error {
	filename := fmt.Sprintf("%s/mem_%d.prof", p.profileDir, time.Now().Unix())
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	runtime.GC() // Force GC before memory profile
	return pprof.WriteHeapProfile(file)
}

// writeGoroutineProfile writes a goroutine profile to disk
func (p *Profiler) writeGoroutineProfile() error {
	filename := fmt.Sprintf("%s/goroutine_%d.prof", p.profileDir, time.Now().Unix())
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return pprof.Lookup("goroutine").WriteTo(file, 0)
}

// writeBlockProfile writes a blocking profile to disk
func (p *Profiler) writeBlockProfile() error {
	filename := fmt.Sprintf("%s/block_%d.prof", p.profileDir, time.Now().Unix())
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return pprof.Lookup("block").WriteTo(file, 0)
}

// PrintSummary prints a summary of performance metrics
func (p *Profiler) PrintSummary() {
	metrics := p.GetMetrics()
	
	fmt.Println("=== Performance Summary ===")
	fmt.Printf("Runtime: %v\n", time.Since(metrics.StartTime))
	
	// Memory statistics
	if len(metrics.MemoryUsage) > 0 {
		latest := metrics.MemoryUsage[len(metrics.MemoryUsage)-1]
		fmt.Printf("Memory - Alloc: %d MB, Sys: %d MB, NumGC: %d\n",
			latest.Alloc/(1024*1024),
			latest.Sys/(1024*1024),
			latest.NumGC)
	}
	
	// Goroutine count
	if len(metrics.GoroutineCount) > 0 {
		latest := metrics.GoroutineCount[len(metrics.GoroutineCount)-1]
		fmt.Printf("Goroutines: %d\n", latest.Count)
	}
	
	// Operation statistics
	fmt.Println("Operation Statistics:")
	for name, stats := range metrics.OperationMetrics {
		avgTime := time.Duration(0)
		if stats.Count > 0 {
			avgTime = stats.TotalTime / time.Duration(stats.Count)
		}
		
		errorRate := float64(stats.Errors) / float64(stats.Count) * 100
		
		fmt.Printf("  %s: count=%d, avg=%v, min=%v, max=%v, errors=%.2f%%\n",
			name, stats.Count, avgTime, stats.MinTime, stats.MaxTime, errorRate)
	}
}