package ai

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelRegistry_RegisterModel(t *testing.T) {
	registry := NewModelRegistry()

	entry := &ModelEntry{
		Info: ModelInfo{
			Name:      "test-model",
			Version:   "1.0",
			Dimension: 384,
			Languages: []string{"en"},
		},
		Config: ModelConfig{
			Name: "test-model",
			Type: "onnx",
		},
		Source: ModelSource{
			Type: "local",
			Path: "/tmp/test.onnx",
		},
		Tags: []string{"test", "small"},
	}

	// Register model
	err := registry.RegisterModel("test-model", entry)
	if err != nil {
		t.Fatalf("Failed to register model: %v", err)
	}

	// Retrieve model
	retrieved, err := registry.GetModel("test-model")
	if err != nil {
		t.Fatalf("Failed to get model: %v", err)
	}

	if retrieved.Info.Name != "test-model" {
		t.Errorf("Expected name 'test-model', got '%s'", retrieved.Info.Name)
	}

	if retrieved.Info.Dimension != 384 {
		t.Errorf("Expected dimension 384, got %d", retrieved.Info.Dimension)
	}
}

func TestModelRegistry_GetNonExistentModel(t *testing.T) {
	registry := NewModelRegistry()

	_, err := registry.GetModel("non-existent")
	if err == nil {
		t.Fatal("Expected error for non-existent model")
	}
}

func TestModelRegistry_ListModels(t *testing.T) {
	registry := NewModelRegistry()

	// Should have default models
	models := registry.ListModels()
	if len(models) < 2 {
		t.Errorf("Expected at least 2 default models, got %d", len(models))
	}

	// Check for expected default models
	_, exists1 := models["all-MiniLM-L6-v2"]
	_, exists2 := models["all-mpnet-base-v2"]

	if !exists1 {
		t.Error("Expected default model 'all-MiniLM-L6-v2' not found")
	}
	if !exists2 {
		t.Error("Expected default model 'all-mpnet-base-v2' not found")
	}
}

func TestModelRegistry_SearchModels(t *testing.T) {
	registry := NewModelRegistry()

	// Search by language
	criteria := SearchCriteria{
		Languages: []string{"en"},
	}

	results := registry.SearchModels(criteria)
	if len(results) == 0 {
		t.Error("Expected at least one English model")
	}

	// Verify all results match criteria
	for _, result := range results {
		found := false
		for _, lang := range result.Entry.Info.Languages {
			if lang == "en" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Model %s doesn't support English", result.Name)
		}
	}
}

func TestModelRegistry_SearchByDimension(t *testing.T) {
	registry := NewModelRegistry()

	// Search by dimension range
	criteria := SearchCriteria{
		MinDimension: 300,
		MaxDimension: 500,
	}

	results := registry.SearchModels(criteria)

	// Verify all results are within dimension range
	for _, result := range results {
		dim := result.Entry.Info.Dimension
		if dim < 300 || dim > 500 {
			t.Errorf("Model %s dimension %d outside range [300, 500]", result.Name, dim)
		}
	}
}

func TestModelRegistry_SearchByTags(t *testing.T) {
	registry := NewModelRegistry()

	// Search by tags
	criteria := SearchCriteria{
		Tags: []string{"fast"},
	}

	results := registry.SearchModels(criteria)

	// Verify all results have the tag
	for _, result := range results {
		found := false
		for _, tag := range result.Entry.Tags {
			if tag == "fast" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Model %s doesn't have 'fast' tag", result.Name)
		}
	}
}

func TestModelRegistry_IsModelDownloaded(t *testing.T) {
	registry := NewModelRegistry()

	// Test with non-existent model
	if registry.IsModelDownloaded("non-existent") {
		t.Error("Expected false for non-existent model")
	}

	// Create a temporary file to simulate downloaded model
	tempDir := os.TempDir()
	modelPath := filepath.Join(tempDir, "test-model")
	file, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	file.Close()
	defer os.Remove(modelPath)

	// Test with existing file
	registry.downloadPath = tempDir
	if !registry.IsModelDownloaded("test-model") {
		t.Error("Expected true for existing model file")
	}
}

func TestModelRegistry_GetLocalPath(t *testing.T) {
	registry := NewModelRegistry()

	path := registry.GetLocalPath("test/model")
	expected := filepath.Join(registry.downloadPath, "test_model")

	if path != expected {
		t.Errorf("Expected path '%s', got '%s'", expected, path)
	}
}

func TestModelRegistry_ValidateModelEntry(t *testing.T) {
	registry := NewModelRegistry()

	// Test valid entry
	validEntry := &ModelEntry{
		Info: ModelInfo{
			Name:      "valid-model",
			Dimension: 384,
		},
		Source: ModelSource{
			Type: "local",
		},
	}

	err := registry.validateModelEntry(validEntry)
	if err != nil {
		t.Errorf("Valid entry should not produce error: %v", err)
	}

	// Test invalid entry - missing name
	invalidEntry1 := &ModelEntry{
		Info: ModelInfo{
			Dimension: 384,
		},
		Source: ModelSource{
			Type: "local",
		},
	}

	err = registry.validateModelEntry(invalidEntry1)
	if err == nil {
		t.Error("Expected error for missing model name")
	}

	// Test invalid entry - zero dimension
	invalidEntry2 := &ModelEntry{
		Info: ModelInfo{
			Name:      "invalid-model",
			Dimension: 0,
		},
		Source: ModelSource{
			Type: "local",
		},
	}

	err = registry.validateModelEntry(invalidEntry2)
	if err == nil {
		t.Error("Expected error for zero dimension")
	}

	// Test invalid entry - missing source type
	invalidEntry3 := &ModelEntry{
		Info: ModelInfo{
			Name:      "invalid-model",
			Dimension: 384,
		},
		Source: ModelSource{},
	}

	err = registry.validateModelEntry(invalidEntry3)
	if err == nil {
		t.Error("Expected error for missing source type")
	}
}

func TestSearchCriteria_EmptyCriteria(t *testing.T) {
	registry := NewModelRegistry()

	// Empty criteria should return all models
	criteria := SearchCriteria{}
	results := registry.SearchModels(criteria)

	allModels := registry.ListModels()
	if len(results) != len(allModels) {
		t.Errorf("Empty criteria should return all models. Expected %d, got %d",
			len(allModels), len(results))
	}
}

func TestModelSearchResult_Relevance(t *testing.T) {
	registry := NewModelRegistry()

	// Test relevance calculation
	criteria := SearchCriteria{
		Tags: []string{"fast", "small"},
	}

	results := registry.SearchModels(criteria)

	// Results should be sorted by relevance (descending)
	for i := 1; i < len(results); i++ {
		if results[i].Relevance > results[i-1].Relevance {
			t.Error("Results should be sorted by relevance in descending order")
		}
	}
}

func BenchmarkModelRegistry_SearchModels(b *testing.B) {
	registry := NewModelRegistry()
	criteria := SearchCriteria{
		Languages: []string{"en"},
		Tags:      []string{"general"},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		registry.SearchModels(criteria)
	}
}

func BenchmarkModelRegistry_GetModel(b *testing.B) {
	registry := NewModelRegistry()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := registry.GetModel("all-MiniLM-L6-v2")
		if err != nil {
			b.Fatalf("Failed to get model: %v", err)
		}
	}
}
