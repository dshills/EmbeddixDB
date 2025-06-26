package embedding

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectModelArchitecture(t *testing.T) {
	testCases := []struct {
		modelPath            string
		expectedArchitecture ModelArchitecture
	}{
		{"bert-base-uncased.onnx", ArchitectureBERT},
		{"distilbert-base-uncased.onnx", ArchitectureDistilBERT},
		{"roberta-base.onnx", ArchitectureRoBERTa},
		{"sentence-t5-base.onnx", ArchitectureSentenceT5},
		{"all-MiniLM-L6-v2.onnx", ArchitectureAllMiniLM},
		{"e5-base-v2.onnx", ArchitectureE5},
		{"bge-base-en-v1.5.onnx", ArchitectureBGE},
		{"custom-model.onnx", ArchitectureUnknown},
		{"some/path/to/bert-large.onnx", ArchitectureBERT},
		{"/absolute/path/to/distilbert.onnx", ArchitectureDistilBERT},
	}

	for _, tc := range testCases {
		t.Run(tc.modelPath, func(t *testing.T) {
			arch := DetectModelArchitecture(tc.modelPath)
			if arch != tc.expectedArchitecture {
				t.Errorf("Expected architecture %s, got %s", tc.expectedArchitecture, arch)
			}
		})
	}
}

func TestGetDefaultConfigForArchitecture(t *testing.T) {
	testCases := []struct {
		architecture      ModelArchitecture
		expectedMaxTokens int
		expectedPooling   string
		expectedNormalize bool
	}{
		{ArchitectureBERT, 512, "cls", true},
		{ArchitectureDistilBERT, 512, "cls", true},
		{ArchitectureRoBERTa, 512, "cls", true},
		{ArchitectureSentenceT5, 512, "mean", true},
		{ArchitectureAllMiniLM, 256, "mean", true},
		{ArchitectureE5, 512, "mean", true},
		{ArchitectureBGE, 512, "cls", true},
		{ArchitectureUnknown, 512, "cls", true},
	}

	for _, tc := range testCases {
		t.Run(string(tc.architecture), func(t *testing.T) {
			config := GetDefaultConfigForArchitecture(tc.architecture)

			if config.MaxTokens != tc.expectedMaxTokens {
				t.Errorf("Expected MaxTokens %d, got %d", tc.expectedMaxTokens, config.MaxTokens)
			}

			if config.PoolingStrategy != tc.expectedPooling {
				t.Errorf("Expected PoolingStrategy %s, got %s", tc.expectedPooling, config.PoolingStrategy)
			}

			if config.NormalizeEmbeddings != tc.expectedNormalize {
				t.Errorf("Expected NormalizeEmbeddings %v, got %v", tc.expectedNormalize, config.NormalizeEmbeddings)
			}

			if config.Name == "" {
				t.Error("Expected non-empty model name")
			}

			if config.BatchSize <= 0 {
				t.Error("Expected positive batch size")
			}
		})
	}
}

func TestValidateModelFile(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Test cases
	testCases := []struct {
		name        string
		setupFunc   func() string
		expectError bool
	}{
		{
			name: "empty path",
			setupFunc: func() string {
				return ""
			},
			expectError: true,
		},
		{
			name: "non-existent file",
			setupFunc: func() string {
				return filepath.Join(tempDir, "nonexistent.onnx")
			},
			expectError: true,
		},
		{
			name: "directory instead of file",
			setupFunc: func() string {
				dirPath := filepath.Join(tempDir, "model_dir.onnx")
				os.Mkdir(dirPath, 0755)
				return dirPath
			},
			expectError: true,
		},
		{
			name: "wrong file extension",
			setupFunc: func() string {
				filePath := filepath.Join(tempDir, "model.txt")
				os.WriteFile(filePath, []byte("dummy content"), 0644)
				return filePath
			},
			expectError: true,
		},
		{
			name: "file too small",
			setupFunc: func() string {
				filePath := filepath.Join(tempDir, "small.onnx")
				os.WriteFile(filePath, []byte("x"), 0644)
				return filePath
			},
			expectError: true,
		},
		{
			name: "valid file",
			setupFunc: func() string {
				filePath := filepath.Join(tempDir, "valid.onnx")
				content := make([]byte, 2048) // Create a file larger than minimum size
				for i := range content {
					content[i] = byte(i % 256)
				}
				os.WriteFile(filePath, content, 0644)
				return filePath
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			modelPath := tc.setupFunc()
			err := ValidateModelFile(modelPath)

			if tc.expectError && err == nil {
				t.Error("Expected error but got none")
			} else if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestGetRecommendedBatchSize(t *testing.T) {
	testCases := []struct {
		architecture ModelArchitecture
		memoryMB     int
		minExpected  int
		maxExpected  int
	}{
		{ArchitectureBERT, 1024, 1, 4},         // Low memory, memory-intensive model
		{ArchitectureBERT, 8192, 6, 12},        // Normal memory
		{ArchitectureBERT, 32768, 12, 24},      // High memory
		{ArchitectureDistilBERT, 1024, 4, 12},  // Low memory, efficient model
		{ArchitectureDistilBERT, 8192, 24, 40}, // Normal memory
		{ArchitectureAllMiniLM, 1024, 4, 12},   // Low memory, efficient model
		{ArchitectureAllMiniLM, 8192, 24, 40},  // Normal memory
		{ArchitectureUnknown, 4096, 12, 20},    // Default case
	}

	for _, tc := range testCases {
		t.Run(string(tc.architecture), func(t *testing.T) {
			batchSize := GetRecommendedBatchSize(tc.architecture, tc.memoryMB)

			if batchSize < tc.minExpected {
				t.Errorf("Batch size %d is below minimum expected %d", batchSize, tc.minExpected)
			}

			if batchSize > tc.maxExpected {
				t.Errorf("Batch size %d is above maximum expected %d", batchSize, tc.maxExpected)
			}

			if batchSize <= 0 {
				t.Error("Batch size should be positive")
			}

			t.Logf("Architecture: %s, Memory: %dMB -> Batch size: %d", tc.architecture, tc.memoryMB, batchSize)
		})
	}
}

func TestEstimateModelMemoryUsage(t *testing.T) {
	// Create a temporary model file for testing
	tempDir, err := os.MkdirTemp("", "memory_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	modelPath := filepath.Join(tempDir, "test.onnx")
	modelContent := make([]byte, 1024*1024) // 1MB model
	err = os.WriteFile(modelPath, modelContent, 0644)
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	testCases := []struct {
		batchSize int
		maxTokens int
		minMemory int64
		maxMemory int64
	}{
		{1, 128, 1024 * 1024, 10 * 1024 * 1024},        // Small batch
		{16, 512, 5 * 1024 * 1024, 50 * 1024 * 1024},   // Medium batch
		{32, 512, 10 * 1024 * 1024, 100 * 1024 * 1024}, // Large batch
	}

	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			memUsage, err := EstimateModelMemoryUsage(modelPath, tc.batchSize, tc.maxTokens)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if memUsage < tc.minMemory {
				t.Errorf("Memory usage %d is below minimum expected %d", memUsage, tc.minMemory)
			}

			if memUsage > tc.maxMemory {
				t.Errorf("Memory usage %d is above maximum expected %d", memUsage, tc.maxMemory)
			}

			t.Logf("Batch: %d, Tokens: %d -> Memory: %d bytes (%.2f MB)",
				tc.batchSize, tc.maxTokens, memUsage, float64(memUsage)/1024/1024)
		})
	}
}

func TestEstimateModelMemoryUsage_NonExistentFile(t *testing.T) {
	_, err := EstimateModelMemoryUsage("/nonexistent/path.onnx", 16, 512)
	if err == nil {
		t.Error("Expected error for non-existent file")
	}
}

func TestCreateConfigFromModel(t *testing.T) {
	// Create a temporary model file for testing
	tempDir, err := os.MkdirTemp("", "config_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Test with a recognizable model name
	modelPath := filepath.Join(tempDir, "bert-base-uncased.onnx")
	modelContent := make([]byte, 2048) // Create a valid-sized file
	err = os.WriteFile(modelPath, modelContent, 0644)
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}

	// This test will fail when it tries to create an ONNX session with invalid content
	// But we can still test the architecture detection part
	t.Run("architecture detection", func(t *testing.T) {
		arch := DetectModelArchitecture(modelPath)
		if arch != ArchitectureBERT {
			t.Errorf("Expected BERT architecture, got %s", arch)
		}
	})

	// Test with non-existent file
	t.Run("non-existent file", func(t *testing.T) {
		_, err := CreateConfigFromModel("/nonexistent/model.onnx")
		if err == nil {
			t.Error("Expected error for non-existent file")
		}
	})
}

func TestMax(t *testing.T) {
	testCases := []struct {
		a, b, expected int
	}{
		{1, 2, 2},
		{5, 3, 5},
		{0, 0, 0},
		{-1, 1, 1},
		{10, 10, 10},
	}

	for _, tc := range testCases {
		result := max(tc.a, tc.b)
		if result != tc.expected {
			t.Errorf("max(%d, %d) = %d, expected %d", tc.a, tc.b, result, tc.expected)
		}
	}
}
