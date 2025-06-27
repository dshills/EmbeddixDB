package ai

import (
	"context"
	"testing"
	"time"
)

// setupTestModelManager creates a model manager with mock engine factory
func setupTestModelManager(maxModels int) *DefaultModelManager {
	manager := NewModelManager(maxModels)
	manager.SetEngineFactory(func(cfg ModelConfig) (EmbeddingEngine, error) {
		return NewMockEmbeddingEngine(cfg), nil
	})
	return manager
}

func TestModelManager_LoadModel(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping model loading test in short mode")
	}
	manager := setupTestModelManager(2)
	defer manager.Close()

	config := ModelConfig{
		Name:                "test-model",
		Type:                "onnx",
		Path:                "/tmp/test.onnx",
		BatchSize:           32,
		NormalizeEmbeddings: true,
	}

	// Test loading a model
	err := manager.LoadModel(context.Background(), "test-model", config)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Verify model is loaded
	engine, err := manager.GetEngine("test-model")
	if err != nil {
		t.Fatalf("Failed to get engine: %v", err)
	}
	if engine == nil {
		t.Fatal("Engine is nil")
	}

	// Test loading same model again (should use cache)
	err = manager.LoadModel(context.Background(), "test-model", config)
	if err != nil {
		t.Fatalf("Failed to reload model: %v", err)
	}

	// Verify stats
	stats := manager.GetStats()
	if stats.ModelsLoaded != 1 {
		t.Errorf("Expected 1 model loaded, got %d", stats.ModelsLoaded)
	}
	if stats.CacheHits != 1 {
		t.Errorf("Expected 1 cache hit, got %d", stats.CacheHits)
	}
}

func TestModelManager_UnloadModel(t *testing.T) {
	manager := setupTestModelManager(2)
	defer manager.Close()

	config := ModelConfig{
		Name: "test-model",
		Type: "onnx",
		Path: "/tmp/test.onnx",
	}

	// Load model
	err := manager.LoadModel(context.Background(), "test-model", config)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Unload model
	err = manager.UnloadModel("test-model")
	if err != nil {
		t.Fatalf("Failed to unload model: %v", err)
	}

	// Verify model is unloaded
	_, err = manager.GetEngine("test-model")
	if err == nil {
		t.Fatal("Expected error when getting unloaded model")
	}

	// Verify stats
	stats := manager.GetStats()
	if stats.ModelsUnloaded != 1 {
		t.Errorf("Expected 1 model unloaded, got %d", stats.ModelsUnloaded)
	}
}

func TestModelManager_EvictLeastUsed(t *testing.T) {
	manager := setupTestModelManager(2) // Max 2 models
	defer manager.Close()

	// Load first model
	config1 := ModelConfig{
		Name: "model-1",
		Type: "onnx",
		Path: "/tmp/model1.onnx",
	}
	err := manager.LoadModel(context.Background(), "model-1", config1)
	if err != nil {
		t.Fatalf("Failed to load model 1: %v", err)
	}

	// Load second model
	config2 := ModelConfig{
		Name: "model-2",
		Type: "onnx",
		Path: "/tmp/model2.onnx",
	}
	err = manager.LoadModel(context.Background(), "model-2", config2)
	if err != nil {
		t.Fatalf("Failed to load model 2: %v", err)
	}

	// Use model 2 to make it more recently used
	_, err = manager.GetEngine("model-2")
	if err != nil {
		t.Fatalf("Failed to get model 2: %v", err)
	}

	// Sleep to ensure different timestamps
	time.Sleep(10 * time.Millisecond)

	// Load third model (should evict model 1)
	config3 := ModelConfig{
		Name: "model-3",
		Type: "onnx",
		Path: "/tmp/model3.onnx",
	}
	err = manager.LoadModel(context.Background(), "model-3", config3)
	if err != nil {
		t.Fatalf("Failed to load model 3: %v", err)
	}

	// Verify model 1 was evicted
	_, err = manager.GetEngine("model-1")
	if err == nil {
		t.Fatal("Expected model 1 to be evicted")
	}

	// Verify models 2 and 3 are still loaded
	_, err = manager.GetEngine("model-2")
	if err != nil {
		t.Fatalf("Model 2 should still be loaded: %v", err)
	}

	_, err = manager.GetEngine("model-3")
	if err != nil {
		t.Fatalf("Model 3 should be loaded: %v", err)
	}
}

func TestModelManager_GetModelHealth(t *testing.T) {
	manager := setupTestModelManager(2)
	defer manager.Close()

	config := ModelConfig{
		Name: "test-model",
		Type: "onnx",
		Path: "/tmp/test.onnx",
	}

	// Load model
	err := manager.LoadModel(context.Background(), "test-model", config)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Get health
	health, err := manager.GetModelHealth("test-model")
	if err != nil {
		t.Fatalf("Failed to get model health: %v", err)
	}

	if health.ModelName != "test-model" {
		t.Errorf("Expected model name 'test-model', got '%s'", health.ModelName)
	}

	if health.Status != "healthy" {
		t.Errorf("Expected status 'healthy', got '%s'", health.Status)
	}
}

func TestModelManager_ListModels(t *testing.T) {
	manager := setupTestModelManager(2)
	defer manager.Close()

	// Load two models
	config1 := ModelConfig{
		Name: "model-1",
		Type: "onnx",
		Path: "/tmp/model1.onnx",
	}
	err := manager.LoadModel(context.Background(), "model-1", config1)
	if err != nil {
		t.Fatalf("Failed to load model 1: %v", err)
	}

	config2 := ModelConfig{
		Name: "model-2",
		Type: "onnx",
		Path: "/tmp/model2.onnx",
	}
	err = manager.LoadModel(context.Background(), "model-2", config2)
	if err != nil {
		t.Fatalf("Failed to load model 2: %v", err)
	}

	// List models
	models := manager.ListModels()
	if len(models) != 2 {
		t.Errorf("Expected 2 models, got %d", len(models))
	}

	// Verify model names are present
	found1, found2 := false, false
	for _, model := range models {
		if model.Name == "model-1" {
			found1 = true
		}
		if model.Name == "model-2" {
			found2 = true
		}
	}

	if !found1 {
		t.Error("Model 1 not found in list")
	}
	if !found2 {
		t.Error("Model 2 not found in list")
	}
}

func TestManagerStats_CacheHitRate(t *testing.T) {
	stats := &ManagerStats{}

	// Record some hits and misses
	stats.recordCacheHit()
	stats.recordCacheHit()
	stats.recordCacheMiss()

	hitRate := stats.GetCacheHitRate()
	expected := 66.66666666666667 // 2/3 * 100

	// Use approximate comparison for floating point
	if abs(hitRate-expected) > 0.001 {
		t.Errorf("Expected cache hit rate %.2f, got %.2f", expected, hitRate)
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

func TestManagerStats_EmptyStats(t *testing.T) {
	stats := &ManagerStats{}

	// Test empty stats
	hitRate := stats.GetCacheHitRate()
	if hitRate != 0.0 {
		t.Errorf("Expected 0.0 hit rate for empty stats, got %.2f", hitRate)
	}
}

func BenchmarkModelManager_GetEngine(b *testing.B) {
	manager := setupTestModelManager(1)
	defer manager.Close()

	config := ModelConfig{
		Name: "bench-model",
		Type: "onnx",
		Path: "/tmp/bench.onnx",
	}

	// Load model
	err := manager.LoadModel(context.Background(), "bench-model", config)
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := manager.GetEngine("bench-model")
		if err != nil {
			b.Fatalf("Failed to get engine: %v", err)
		}
	}
}

func BenchmarkModelManager_LoadModel(b *testing.B) {
	for i := 0; i < b.N; i++ {
		manager := setupTestModelManager(1)
		config := ModelConfig{
			Name: "bench-model",
			Type: "onnx",
			Path: "/tmp/bench.onnx",
		}

		err := manager.LoadModel(context.Background(), "bench-model", config)
		if err != nil {
			b.Fatalf("Failed to load model: %v", err)
		}

		manager.Close()
	}
}
