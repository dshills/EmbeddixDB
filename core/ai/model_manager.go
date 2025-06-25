package ai

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DefaultModelManager implements the ModelManager interface for embedding model lifecycle
type DefaultModelManager struct {
	models        map[string]*LoadedModel
	configs       map[string]ModelConfig
	registry      *ModelRegistry
	healthChecker *HealthChecker
	mutex         sync.RWMutex
	maxModels     int
	stats         *ManagerStats
}

// LoadedModel represents a model loaded in memory
type LoadedModel struct {
	Name       string
	Engine     EmbeddingEngine
	Config     ModelConfig
	LoadedAt   time.Time
	LastUsed   time.Time
	UsageCount int64
	Health     ModelHealth
	mutex      sync.RWMutex
}

// ManagerStats tracks model manager performance
type ManagerStats struct {
	ModelsLoaded    int64         `json:"models_loaded"`
	ModelsUnloaded  int64         `json:"models_unloaded"`
	TotalRequests   int64         `json:"total_requests"`
	CacheHits       int64         `json:"cache_hits"`
	CacheMisses     int64         `json:"cache_misses"`
	AverageLoadTime time.Duration `json:"average_load_time"`
	MemoryUsage     int64         `json:"memory_usage_mb"`
	ErrorCount      int64         `json:"error_count"`
	mutex           sync.RWMutex
}

// NewModelManager creates a new model manager instance
func NewModelManager(maxModels int) *DefaultModelManager {
	manager := &DefaultModelManager{
		models:    make(map[string]*LoadedModel),
		configs:   make(map[string]ModelConfig),
		maxModels: maxModels,
		stats:     &ManagerStats{},
	}

	// Initialize registry and health checker
	manager.registry = NewModelRegistry()
	manager.healthChecker = NewHealthChecker(manager)

	// Start background tasks
	manager.startBackgroundTasks()

	return manager
}

// LoadModel loads a model for inference
func (m *DefaultModelManager) LoadModel(ctx context.Context, modelName string, config ModelConfig) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if model is already loaded
	if loadedModel, exists := m.models[modelName]; exists {
		loadedModel.mutex.Lock()
		loadedModel.LastUsed = time.Now()
		loadedModel.mutex.Unlock()
		m.stats.recordCacheHit()
		return nil
	}

	// Check model limit
	if len(m.models) >= m.maxModels {
		if err := m.evictLeastUsedModel(); err != nil {
			return fmt.Errorf("failed to evict model for space: %w", err)
		}
	}

	// Load the model
	start := time.Now()
	engine, err := m.createEmbeddingEngine(config)
	if err != nil {
		m.stats.recordError()
		return fmt.Errorf("failed to create embedding engine: %w", err)
	}

	// Create loaded model entry
	loadedModel := &LoadedModel{
		Name:       modelName,
		Engine:     engine,
		Config:     config,
		LoadedAt:   time.Now(),
		LastUsed:   time.Now(),
		UsageCount: 0,
		Health: ModelHealth{
			ModelName: modelName,
			Status:    "loading",
			LoadedAt:  time.Now(),
			LastCheck: time.Now(),
		},
	}

	// Warm up the model
	if err := engine.Warm(ctx); err != nil {
		engine.Close()
		m.stats.recordError()
		return fmt.Errorf("model warmup failed: %w", err)
	}

	// Update health status
	loadedModel.Health.Status = "healthy"
	loadedModel.Health.Latency = time.Since(start)

	// Store the loaded model
	m.models[modelName] = loadedModel
	m.configs[modelName] = config

	// Update statistics
	m.stats.recordModelLoaded(time.Since(start))

	return nil
}

// UnloadModel releases model resources
func (m *DefaultModelManager) UnloadModel(modelName string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	loadedModel, exists := m.models[modelName]
	if !exists {
		return fmt.Errorf("model %s not loaded", modelName)
	}

	// Close the engine
	if err := loadedModel.Engine.Close(); err != nil {
		return fmt.Errorf("failed to close engine: %w", err)
	}

	// Remove from maps
	delete(m.models, modelName)
	delete(m.configs, modelName)

	// Update statistics
	m.stats.recordModelUnloaded()

	return nil
}

// GetEngine returns a loaded embedding engine
func (m *DefaultModelManager) GetEngine(modelName string) (EmbeddingEngine, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	loadedModel, exists := m.models[modelName]
	if !exists {
		m.stats.recordCacheMiss()
		return nil, fmt.Errorf("model %s not loaded", modelName)
	}

	// Update usage statistics
	loadedModel.mutex.Lock()
	loadedModel.LastUsed = time.Now()
	loadedModel.UsageCount++
	loadedModel.mutex.Unlock()

	m.stats.recordRequest()
	return loadedModel.Engine, nil
}

// ListModels returns available models
func (m *DefaultModelManager) ListModels() []ModelInfo {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	models := make([]ModelInfo, 0, len(m.models))
	for _, loadedModel := range m.models {
		models = append(models, loadedModel.Engine.GetModelInfo())
	}

	return models
}

// GetModelHealth returns model health status
func (m *DefaultModelManager) GetModelHealth(modelName string) (ModelHealth, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	loadedModel, exists := m.models[modelName]
	if !exists {
		return ModelHealth{}, fmt.Errorf("model %s not loaded", modelName)
	}

	loadedModel.mutex.RLock()
	health := loadedModel.Health
	loadedModel.mutex.RUnlock()

	return health, nil
}

// GetStats returns manager statistics
func (m *DefaultModelManager) GetStats() *ManagerStats {
	m.stats.mutex.RLock()
	defer m.stats.mutex.RUnlock()

	// Create a copy to avoid concurrent access issues
	stats := &ManagerStats{
		ModelsLoaded:    m.stats.ModelsLoaded,
		ModelsUnloaded:  m.stats.ModelsUnloaded,
		TotalRequests:   m.stats.TotalRequests,
		CacheHits:       m.stats.CacheHits,
		CacheMisses:     m.stats.CacheMisses,
		AverageLoadTime: m.stats.AverageLoadTime,
		MemoryUsage:     m.stats.MemoryUsage,
		ErrorCount:      m.stats.ErrorCount,
	}

	return stats
}

// Close releases all loaded models and stops background tasks
func (m *DefaultModelManager) Close() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Stop health checker
	if m.healthChecker != nil {
		m.healthChecker.Stop()
	}

	// Unload all models
	var lastError error
	for modelName := range m.models {
		if err := m.unloadModelUnsafe(modelName); err != nil {
			lastError = err
		}
	}

	return lastError
}

// createEmbeddingEngine creates an embedding engine based on config
func (m *DefaultModelManager) createEmbeddingEngine(config ModelConfig) (EmbeddingEngine, error) {
	switch config.Type {
	case "onnx":
		// For now, return a mock implementation to avoid import cycle
		// In a real implementation, we'd have a factory pattern or dependency injection
		return NewMockEmbeddingEngine(config), nil
	default:
		return nil, fmt.Errorf("unsupported model type: %s", config.Type)
	}
}

// evictLeastUsedModel removes the least recently used model to make space
func (m *DefaultModelManager) evictLeastUsedModel() error {
	var oldestModel *LoadedModel
	var oldestName string
	oldestTime := time.Now()

	for name, model := range m.models {
		model.mutex.RLock()
		lastUsed := model.LastUsed
		model.mutex.RUnlock()

		if lastUsed.Before(oldestTime) {
			oldestTime = lastUsed
			oldestModel = model
			oldestName = name
		}
	}

	if oldestModel == nil {
		return fmt.Errorf("no models to evict")
	}

	return m.unloadModelUnsafe(oldestName)
}

// unloadModelUnsafe unloads a model without acquiring locks (internal use)
func (m *DefaultModelManager) unloadModelUnsafe(modelName string) error {
	loadedModel, exists := m.models[modelName]
	if !exists {
		return fmt.Errorf("model %s not loaded", modelName)
	}

	if err := loadedModel.Engine.Close(); err != nil {
		return fmt.Errorf("failed to close engine: %w", err)
	}

	delete(m.models, modelName)
	delete(m.configs, modelName)
	m.stats.recordModelUnloaded()

	return nil
}

// startBackgroundTasks starts periodic maintenance tasks
func (m *DefaultModelManager) startBackgroundTasks() {
	// Start health checking
	go m.healthChecker.Start()

	// Start periodic cleanup
	go m.periodicCleanup()
}

// periodicCleanup runs periodic maintenance tasks
func (m *DefaultModelManager) periodicCleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		m.updateMemoryUsage()
		m.cleanupUnhealthyModels()
	}
}

// updateMemoryUsage estimates memory usage of loaded models
func (m *DefaultModelManager) updateMemoryUsage() {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var totalMemory int64
	for _, model := range m.models {
		info := model.Engine.GetModelInfo()
		totalMemory += info.Size
	}

	m.stats.mutex.Lock()
	m.stats.MemoryUsage = totalMemory / (1024 * 1024) // Convert to MB
	m.stats.mutex.Unlock()
}

// cleanupUnhealthyModels removes models that have been unhealthy for too long
func (m *DefaultModelManager) cleanupUnhealthyModels() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	unhealthyThreshold := 10 * time.Minute
	now := time.Now()

	for name, model := range m.models {
		model.mutex.RLock()
		isUnhealthy := model.Health.Status == "unhealthy"
		lastCheck := model.Health.LastCheck
		model.mutex.RUnlock()

		if isUnhealthy && now.Sub(lastCheck) > unhealthyThreshold {
			m.unloadModelUnsafe(name)
		}
	}
}

// ManagerStats methods

func (s *ManagerStats) recordModelLoaded(loadTime time.Duration) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.ModelsLoaded++

	// Update average load time
	if s.AverageLoadTime == 0 {
		s.AverageLoadTime = loadTime
	} else {
		s.AverageLoadTime = (s.AverageLoadTime + loadTime) / 2
	}
}

func (s *ManagerStats) recordModelUnloaded() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.ModelsUnloaded++
}

func (s *ManagerStats) recordRequest() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.TotalRequests++
}

func (s *ManagerStats) recordCacheHit() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.CacheHits++
}

func (s *ManagerStats) recordCacheMiss() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.CacheMisses++
}

func (s *ManagerStats) recordError() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.ErrorCount++
}

// GetCacheHitRate returns the cache hit rate as a percentage
func (s *ManagerStats) GetCacheHitRate() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	total := s.CacheHits + s.CacheMisses
	if total == 0 {
		return 0.0
	}

	return float64(s.CacheHits) / float64(total) * 100.0
}
