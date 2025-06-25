package embedding

import (
	"os"
	"testing"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

func initONNXRuntime() error {
	if !onnxruntime.IsInitialized() {
		return onnxruntime.InitializeEnvironment()
	}
	return nil
}

func TestNewRealONNXTensor(t *testing.T) {
	if err := initONNXRuntime(); err != nil {
		t.Skipf("Failed to initialize ONNX Runtime: %v", err)
	}
	
	testCases := []struct {
		name        string
		data        interface{}
		shape       []int64
		expectError bool
	}{
		{
			name:        "int64 data",
			data:        []int64{1, 2, 3, 4},
			shape:       []int64{2, 2},
			expectError: false,
		},
		{
			name:        "float32 data",
			data:        []float32{1.0, 2.0, 3.0, 4.0},
			shape:       []int64{2, 2},
			expectError: false,
		},
		{
			name:        "int32 data",
			data:        []int32{1, 2, 3, 4},
			shape:       []int64{2, 2},
			expectError: false,
		},
		{
			name:        "unsupported data type",
			data:        []string{"a", "b", "c"},
			shape:       []int64{3},
			expectError: true,
		},
		{
			name:        "empty data",
			data:        []int64{},
			shape:       []int64{0},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := NewRealONNXTensor(tc.data, tc.shape)
			
			if tc.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			
			if tensor == nil {
				t.Fatal("Expected tensor to be created")
			}
			
			// Verify shape
			actualShape := tensor.GetShape()
			if len(actualShape) != len(tc.shape) {
				t.Errorf("Expected shape length %d, got %d", len(tc.shape), len(actualShape))
			}
			
			for i, dim := range tc.shape {
				if i < len(actualShape) && actualShape[i] != dim {
					t.Errorf("Expected shape[%d] = %d, got %d", i, dim, actualShape[i])
				}
			}
			
			// Clean up
			tensor.Destroy()
		})
	}
}

func TestCreateInputTensorFromTokens(t *testing.T) {
	if err := initONNXRuntime(); err != nil {
		t.Skipf("Failed to initialize ONNX Runtime: %v", err)
	}
	
	testCases := []struct {
		name        string
		tokens      [][]int64
		inputName   string
		expectError bool
	}{
		{
			name:        "valid tokens",
			tokens:      [][]int64{{1, 2, 3}, {4, 5, 6}},
			inputName:   "input_ids",
			expectError: false,
		},
		{
			name:        "single sequence",
			tokens:      [][]int64{{1, 2, 3, 4, 5}},
			inputName:   "input_ids",
			expectError: false,
		},
		{
			name:        "empty tokens",
			tokens:      [][]int64{},
			inputName:   "input_ids",
			expectError: true,
		},
		{
			name:        "mismatched sequence lengths",
			tokens:      [][]int64{{1, 2, 3}, {4, 5}},
			inputName:   "input_ids",
			expectError: true,
		},
		{
			name:        "empty sequence",
			tokens:      [][]int64{{}},
			inputName:   "input_ids",
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := CreateInputTensorFromTokens(tc.tokens, tc.inputName)
			
			if tc.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			
			if tensor == nil {
				t.Fatal("Expected tensor to be created")
			}
			
			// Verify shape
			shape := tensor.GetShape()
			if len(shape) != 2 {
				t.Errorf("Expected 2D tensor, got shape: %v", shape)
			}
			
			if len(tc.tokens) > 0 {
				expectedBatchSize := int64(len(tc.tokens))
				expectedSeqLen := int64(len(tc.tokens[0]))
				
				if shape[0] != expectedBatchSize {
					t.Errorf("Expected batch size %d, got %d", expectedBatchSize, shape[0])
				}
				
				if shape[1] != expectedSeqLen {
					t.Errorf("Expected sequence length %d, got %d", expectedSeqLen, shape[1])
				}
			}
			
			// Clean up
			tensor.Destroy()
		})
	}
}

func TestCreateAttentionMaskTensor(t *testing.T) {
	if err := initONNXRuntime(); err != nil {
		t.Skipf("Failed to initialize ONNX Runtime: %v", err)
	}
	
	testCases := []struct {
		name        string
		masks       [][]int64
		expectError bool
	}{
		{
			name:        "valid masks",
			masks:       [][]int64{{1, 1, 0}, {1, 1, 1}},
			expectError: false,
		},
		{
			name:        "single mask",
			masks:       [][]int64{{1, 1, 1, 0, 0}},
			expectError: false,
		},
		{
			name:        "empty masks",
			masks:       [][]int64{},
			expectError: true,
		},
		{
			name:        "empty mask sequence",
			masks:       [][]int64{{}},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := CreateAttentionMaskTensor(tc.masks)
			
			if tc.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			
			if tensor == nil {
				t.Fatal("Expected tensor to be created")
			}
			
			// Verify shape
			shape := tensor.GetShape()
			if len(shape) != 2 {
				t.Errorf("Expected 2D tensor, got shape: %v", shape)
			}
			
			if len(tc.masks) > 0 {
				expectedBatchSize := int64(len(tc.masks))
				expectedSeqLen := int64(len(tc.masks[0]))
				
				if shape[0] != expectedBatchSize {
					t.Errorf("Expected batch size %d, got %d", expectedBatchSize, shape[0])
				}
				
				if shape[1] != expectedSeqLen {
					t.Errorf("Expected sequence length %d, got %d", expectedSeqLen, shape[1])
				}
			}
			
			// Clean up
			tensor.Destroy()
		})
	}
}

func TestExtractEmbeddingsFromTensor(t *testing.T) {
	if err := initONNXRuntime(); err != nil {
		t.Skipf("Failed to initialize ONNX Runtime: %v", err)
	}
	
	testCases := []struct {
		name            string
		data            []float32
		shape           []int64
		poolingStrategy string
		expectError     bool
		expectedBatch   int
		expectedDim     int
	}{
		{
			name:            "2D tensor - already pooled",
			data:            []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			shape:           []int64{2, 3},
			poolingStrategy: "cls",
			expectError:     false,
			expectedBatch:   2,
			expectedDim:     3,
		},
		{
			name:            "3D tensor - CLS pooling",
			data:            []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
			shape:           []int64{2, 2, 3}, // batch_size=2, seq_len=2, hidden_size=3
			poolingStrategy: "cls",
			expectError:     false,
			expectedBatch:   2,
			expectedDim:     3,
		},
		{
			name:            "3D tensor - mean pooling",
			data:            []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
			shape:           []int64{2, 2, 3},
			poolingStrategy: "mean",
			expectError:     false,
			expectedBatch:   2,
			expectedDim:     3,
		},
		{
			name:            "3D tensor - max pooling",
			data:            []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
			shape:           []int64{2, 2, 3},
			poolingStrategy: "max",
			expectError:     false,
			expectedBatch:   2,
			expectedDim:     3,
		},
		{
			name:            "unsupported pooling strategy",
			data:            []float32{1.0, 2.0, 3.0, 4.0},
			shape:           []int64{1, 2, 2},
			poolingStrategy: "invalid",
			expectError:     true,
		},
		{
			name:            "unsupported shape",
			data:            []float32{1.0, 2.0, 3.0, 4.0},
			shape:           []int64{2, 2, 1, 1}, // 4D not supported
			poolingStrategy: "cls",
			expectError:     true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a real tensor for testing
			tensor, err := NewRealONNXTensor(tc.data, tc.shape)
			if err != nil {
				t.Fatalf("Failed to create test tensor: %v", err)
			}
			defer tensor.Destroy()
			
			embeddings, err := ExtractEmbeddingsFromTensor(tensor, tc.poolingStrategy)
			
			if tc.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			
			if len(embeddings) != tc.expectedBatch {
				t.Errorf("Expected %d embeddings, got %d", tc.expectedBatch, len(embeddings))
			}
			
			for i, embedding := range embeddings {
				if len(embedding) != tc.expectedDim {
					t.Errorf("Embedding %d: expected dimension %d, got %d", i, tc.expectedDim, len(embedding))
				}
			}
			
			t.Logf("Extracted embeddings shape: [%d, %d]", len(embeddings), len(embeddings[0]))
		})
	}
}

func TestGetONNXRuntimeVersion(t *testing.T) {
	version := GetONNXRuntimeVersion()
	// The version might be empty if ONNX Runtime is not properly initialized
	// This is expected in test environments without the actual library
	if version == "" {
		t.Skip("ONNX Runtime version not available - library may not be installed")
	}
	t.Logf("ONNX Runtime version: %s", version)
}

func TestGetAvailableProviders(t *testing.T) {
	providers := GetAvailableProviders()
	if len(providers) == 0 {
		t.Error("Expected at least one execution provider")
	}
	
	// Should at least have CPU provider
	hasCPU := false
	for _, provider := range providers {
		if provider == "CPUExecutionProvider" {
			hasCPU = true
			break
		}
	}
	
	if !hasCPU {
		t.Error("Expected CPUExecutionProvider to be available")
	}
	
	t.Logf("Available providers: %v", providers)
}

// TestNewRealONNXSession tests session creation with a non-existent model file
// In a real scenario, you would use an actual ONNX model file
func TestNewRealONNXSession_NonExistentModel(t *testing.T) {
	nonExistentPath := "/path/to/nonexistent/model.onnx"
	
	session, err := NewRealONNXSession(nonExistentPath)
	if err == nil {
		t.Error("Expected error for non-existent model file")
		if session != nil {
			session.Destroy()
		}
	}
	
	if session != nil {
		t.Error("Expected session to be nil for non-existent model")
	}
}

// TestRealONNXSession_EmptyPath tests session creation with empty path
func TestNewRealONNXSession_EmptyPath(t *testing.T) {
	session, err := NewRealONNXSession("")
	if err == nil {
		t.Error("Expected error for empty model path")
		if session != nil {
			session.Destroy()
		}
	}
	
	if session != nil {
		t.Error("Expected session to be nil for empty path")
	}
}

// Integration test that would work with a real ONNX model
func TestRealONNXSession_Integration(t *testing.T) {
	// Skip this test unless we have a real model file
	modelPath := os.Getenv("TEST_ONNX_MODEL_PATH")
	if modelPath == "" {
		t.Skip("Skipping integration test - set TEST_ONNX_MODEL_PATH environment variable to test with real model")
	}
	
	session, err := NewRealONNXSession(modelPath)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()
	
	// Test session properties
	if session.GetInputCount() <= 0 {
		t.Error("Expected at least one input")
	}
	
	if session.GetOutputCount() <= 0 {
		t.Error("Expected at least one output")
	}
	
	// Test input/output names
	inputName := session.GetInputName(0)
	if inputName == "" {
		t.Error("Expected non-empty input name")
	}
	
	outputName := session.GetOutputName(0)
	if outputName == "" {
		t.Error("Expected non-empty output name")
	}
	
	t.Logf("Model inputs: %d, outputs: %d", session.GetInputCount(), session.GetOutputCount())
	t.Logf("First input: %s, first output: %s", inputName, outputName)
}