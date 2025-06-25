package ai

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// HealthChecker monitors model health and performance
type HealthChecker struct {
	manager       *DefaultModelManager
	checkInterval time.Duration
	timeout       time.Duration
	stopChan      chan struct{}
	running       bool
	mutex         sync.RWMutex
}

// HealthCheckResult contains the result of a health check
type HealthCheckResult struct {
	ModelName    string        `json:"model_name"`
	Healthy      bool          `json:"healthy"`
	Latency      time.Duration `json:"latency"`
	ErrorMessage string        `json:"error_message,omitempty"`
	CPUUsage     float64       `json:"cpu_usage"`
	MemoryUsage  int64         `json:"memory_usage_mb"`
	GPUUsage     float64       `json:"gpu_usage"`
	Timestamp    time.Time     `json:"timestamp"`
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(manager *DefaultModelManager) *HealthChecker {
	return &HealthChecker{
		manager:       manager,
		checkInterval: 2 * time.Minute,
		timeout:       30 * time.Second,
		stopChan:      make(chan struct{}),
	}
}

// Start begins health checking in a background goroutine
func (h *HealthChecker) Start() {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.running {
		return
	}

	h.running = true
	go h.checkLoop()
}

// Stop stops the health checker
func (h *HealthChecker) Stop() {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if !h.running {
		return
	}

	h.running = false
	close(h.stopChan)
}

// CheckModel performs a health check on a specific model
func (h *HealthChecker) CheckModel(modelName string) (*HealthCheckResult, error) {
	h.manager.mutex.RLock()
	loadedModel, exists := h.manager.models[modelName]
	h.manager.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model %s not loaded", modelName)
	}

	result := &HealthCheckResult{
		ModelName: modelName,
		Timestamp: time.Now(),
	}

	// Perform health check with timeout
	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()

	start := time.Now()

	// Test inference with a simple input
	testContent := []string{"Health check test input"}
	_, err := loadedModel.Engine.Embed(ctx, testContent)

	result.Latency = time.Since(start)
	result.Healthy = err == nil

	if err != nil {
		result.ErrorMessage = err.Error()
	}

	// Get system metrics
	h.updateSystemMetrics(result)

	// Update loaded model health
	loadedModel.mutex.Lock()
	loadedModel.Health = ModelHealth{
		ModelName:    modelName,
		Status:       h.getStatusFromResult(result),
		LoadedAt:     loadedModel.LoadedAt,
		LastCheck:    result.Timestamp,
		Latency:      result.Latency,
		ErrorRate:    h.calculateErrorRate(loadedModel),
		MemoryUsage:  result.MemoryUsage,
		CPUUsage:     result.CPUUsage,
		GPUUsage:     result.GPUUsage,
		ErrorMessage: result.ErrorMessage,
	}
	loadedModel.mutex.Unlock()

	return result, nil
}

// CheckAllModels performs health checks on all loaded models
func (h *HealthChecker) CheckAllModels() map[string]*HealthCheckResult {
	h.manager.mutex.RLock()
	modelNames := make([]string, 0, len(h.manager.models))
	for name := range h.manager.models {
		modelNames = append(modelNames, name)
	}
	h.manager.mutex.RUnlock()

	results := make(map[string]*HealthCheckResult)

	// Check models concurrently
	var wg sync.WaitGroup
	resultChan := make(chan struct {
		name   string
		result *HealthCheckResult
		err    error
	}, len(modelNames))

	for _, name := range modelNames {
		wg.Add(1)
		go func(modelName string) {
			defer wg.Done()
			result, err := h.CheckModel(modelName)
			resultChan <- struct {
				name   string
				result *HealthCheckResult
				err    error
			}{modelName, result, err}
		}(name)
	}

	// Close channel when all checks complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for item := range resultChan {
		if item.err == nil {
			results[item.name] = item.result
		}
	}

	return results
}

// SetCheckInterval updates the health check interval
func (h *HealthChecker) SetCheckInterval(interval time.Duration) {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	h.checkInterval = interval
}

// SetTimeout updates the health check timeout
func (h *HealthChecker) SetTimeout(timeout time.Duration) {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	h.timeout = timeout
}

// IsRunning returns whether the health checker is running
func (h *HealthChecker) IsRunning() bool {
	h.mutex.RLock()
	defer h.mutex.RUnlock()
	return h.running
}

// checkLoop runs the periodic health check loop
func (h *HealthChecker) checkLoop() {
	ticker := time.NewTicker(h.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			h.performScheduledCheck()
		case <-h.stopChan:
			return
		}
	}
}

// performScheduledCheck runs health checks on all models
func (h *HealthChecker) performScheduledCheck() {
	results := h.CheckAllModels()

	// Log results and take action on unhealthy models
	for modelName, result := range results {
		if !result.Healthy {
			h.handleUnhealthyModel(modelName, result)
		}
	}
}

// handleUnhealthyModel takes action when a model is unhealthy
func (h *HealthChecker) handleUnhealthyModel(modelName string, result *HealthCheckResult) {
	// For now, just log the issue
	// In a production system, this could:
	// - Send alerts
	// - Attempt to restart the model
	// - Mark model as unavailable
	// - Switch to a backup model

	fmt.Printf("Model %s is unhealthy: %s (latency: %v)\n",
		modelName, result.ErrorMessage, result.Latency)
}

// getStatusFromResult converts health check result to status string
func (h *HealthChecker) getStatusFromResult(result *HealthCheckResult) string {
	if !result.Healthy {
		return "unhealthy"
	}

	// Check latency thresholds
	if result.Latency > 5*time.Second {
		return "degraded"
	}

	return "healthy"
}

// calculateErrorRate computes the error rate for a model
func (h *HealthChecker) calculateErrorRate(model *LoadedModel) float64 {
	// This is a simplified calculation
	// In a real implementation, you'd track error counts over time
	return 0.0
}

// updateSystemMetrics updates system resource usage metrics
func (h *HealthChecker) updateSystemMetrics(result *HealthCheckResult) {
	// Get memory statistics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Convert to MB
	result.MemoryUsage = int64(m.Alloc) / (1024 * 1024)

	// CPU usage would require platform-specific code or external libraries
	// For now, we'll use a placeholder
	result.CPUUsage = 0.0

	// GPU usage would require CUDA/ROCm libraries
	// For now, we'll use a placeholder
	result.GPUUsage = 0.0
}

// GetHealthSummary returns a summary of all model health statuses
func (h *HealthChecker) GetHealthSummary() *HealthSummary {
	results := h.CheckAllModels()

	summary := &HealthSummary{
		TotalModels:     len(results),
		HealthyModels:   0,
		UnhealthyModels: 0,
		DegradedModels:  0,
		LastCheck:       time.Now(),
		Models:          make(map[string]string),
	}

	for name, result := range results {
		status := h.getStatusFromResult(result)
		summary.Models[name] = status

		switch status {
		case "healthy":
			summary.HealthyModels++
		case "unhealthy":
			summary.UnhealthyModels++
		case "degraded":
			summary.DegradedModels++
		}
	}

	// Calculate overall health percentage
	if summary.TotalModels > 0 {
		summary.OverallHealth = float64(summary.HealthyModels) / float64(summary.TotalModels) * 100.0
	}

	return summary
}

// HealthSummary provides an overview of system health
type HealthSummary struct {
	TotalModels     int               `json:"total_models"`
	HealthyModels   int               `json:"healthy_models"`
	UnhealthyModels int               `json:"unhealthy_models"`
	DegradedModels  int               `json:"degraded_models"`
	OverallHealth   float64           `json:"overall_health_percent"`
	LastCheck       time.Time         `json:"last_check"`
	Models          map[string]string `json:"models"` // model_name -> status
}
