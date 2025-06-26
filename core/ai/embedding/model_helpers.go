package embedding

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// ModelArchitecture represents common transformer architectures
type ModelArchitecture string

const (
	ArchitectureBERT       ModelArchitecture = "bert"
	ArchitectureDistilBERT ModelArchitecture = "distilbert"
	ArchitectureRoBERTa    ModelArchitecture = "roberta"
	ArchitectureSentenceT5 ModelArchitecture = "sentence-t5"
	ArchitectureAllMiniLM  ModelArchitecture = "all-minilm"
	ArchitectureE5         ModelArchitecture = "e5"
	ArchitectureBGE        ModelArchitecture = "bge"
	ArchitectureUnknown    ModelArchitecture = "unknown"
)

// ModelInfo contains metadata about an ONNX model
type ModelInfo struct {
	Path         string            `json:"path"`
	Architecture ModelArchitecture `json:"architecture"`
	Dimension    int               `json:"dimension"`
	MaxLength    int               `json:"max_length"`
	Inputs       []string          `json:"inputs"`
	Outputs      []string          `json:"outputs"`
	Size         int64             `json:"size_bytes"`
	Checksum     string            `json:"checksum,omitempty"`
}

// DetectModelArchitecture attempts to detect the model architecture from its path or metadata
func DetectModelArchitecture(modelPath string) ModelArchitecture {
	// Normalize path for comparison
	lowerPath := strings.ToLower(filepath.Base(modelPath))

	// Check for common model names/patterns - order matters!
	if strings.Contains(lowerPath, "distilbert") {
		return ArchitectureDistilBERT
	}
	if strings.Contains(lowerPath, "roberta") {
		return ArchitectureRoBERTa
	}
	if strings.Contains(lowerPath, "bert") {
		return ArchitectureBERT
	}
	if strings.Contains(lowerPath, "sentence-t5") || strings.Contains(lowerPath, "st5") {
		return ArchitectureSentenceT5
	}
	if strings.Contains(lowerPath, "all-minilm") || strings.Contains(lowerPath, "minilm") {
		return ArchitectureAllMiniLM
	}
	if strings.Contains(lowerPath, "e5-") || strings.Contains(lowerPath, "e5_") {
		return ArchitectureE5
	}
	if strings.Contains(lowerPath, "bge-") || strings.Contains(lowerPath, "bge_") {
		return ArchitectureBGE
	}

	return ArchitectureUnknown
}

// GetDefaultConfigForArchitecture returns default configuration for known architectures
func GetDefaultConfigForArchitecture(arch ModelArchitecture) ai.ModelConfig {
	baseConfig := ai.ModelConfig{
		MaxTokens:           512,
		BatchSize:           16,
		NormalizeEmbeddings: true,
		PoolingStrategy:     "cls",
	}

	switch arch {
	case ArchitectureBERT:
		baseConfig.Name = "bert-base-uncased"
		baseConfig.MaxTokens = 512
		baseConfig.PoolingStrategy = "cls"

	case ArchitectureDistilBERT:
		baseConfig.Name = "distilbert-base-uncased"
		baseConfig.MaxTokens = 512
		baseConfig.PoolingStrategy = "cls"

	case ArchitectureRoBERTa:
		baseConfig.Name = "roberta-base"
		baseConfig.MaxTokens = 512
		baseConfig.PoolingStrategy = "cls"

	case ArchitectureSentenceT5:
		baseConfig.Name = "sentence-t5-base"
		baseConfig.MaxTokens = 512
		baseConfig.PoolingStrategy = "mean"

	case ArchitectureAllMiniLM:
		baseConfig.Name = "all-MiniLM-L6-v2"
		baseConfig.MaxTokens = 256
		baseConfig.PoolingStrategy = "mean"
		baseConfig.NormalizeEmbeddings = true

	case ArchitectureE5:
		baseConfig.Name = "e5-base-v2"
		baseConfig.MaxTokens = 512
		baseConfig.PoolingStrategy = "mean"
		baseConfig.NormalizeEmbeddings = true

	case ArchitectureBGE:
		baseConfig.Name = "bge-base-en-v1.5"
		baseConfig.MaxTokens = 512
		baseConfig.PoolingStrategy = "cls"
		baseConfig.NormalizeEmbeddings = true

	default:
		baseConfig.Name = "unknown-model"
		baseConfig.PoolingStrategy = "cls"
	}

	return baseConfig
}

// ValidateModelFile checks if the model file exists and is readable
func ValidateModelFile(modelPath string) error {
	if modelPath == "" {
		return fmt.Errorf("model path is empty")
	}

	// Check if file exists
	info, err := os.Stat(modelPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("model file does not exist: %s", modelPath)
		}
		return fmt.Errorf("cannot access model file: %w", err)
	}

	// Check if it's a regular file
	if !info.Mode().IsRegular() {
		return fmt.Errorf("model path is not a regular file: %s", modelPath)
	}

	// Check file extension
	ext := strings.ToLower(filepath.Ext(modelPath))
	if ext != ".onnx" {
		return fmt.Errorf("model file must have .onnx extension, got: %s", ext)
	}

	// Check minimum file size (ONNX models should be at least a few KB)
	if info.Size() < 1024 {
		return fmt.Errorf("model file seems too small (%d bytes), may be corrupted", info.Size())
	}

	return nil
}

// InspectONNXModel attempts to load a model and extract metadata
func InspectONNXModel(modelPath string) (*ModelInfo, error) {
	// Validate the file first
	if err := ValidateModelFile(modelPath); err != nil {
		return nil, err
	}

	// Get file info
	fileInfo, err := os.Stat(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	// Try to create a session to inspect the model
	session, err := NewRealONNXSession(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect model: %w", err)
	}
	defer session.Destroy()

	// Extract input and output information
	inputs := make([]string, session.GetInputCount())
	for i := 0; i < session.GetInputCount(); i++ {
		inputs[i] = session.GetInputName(i)
	}

	outputs := make([]string, session.GetOutputCount())
	for i := 0; i < session.GetOutputCount(); i++ {
		outputs[i] = session.GetOutputName(i)
	}

	// Detect architecture
	architecture := DetectModelArchitecture(modelPath)

	// Get absolute path
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		absPath = modelPath
	}

	return &ModelInfo{
		Path:         absPath,
		Architecture: architecture,
		Dimension:    0,   // Will be determined during first inference
		MaxLength:    512, // Default, may be overridden by config
		Inputs:       inputs,
		Outputs:      outputs,
		Size:         fileInfo.Size(),
	}, nil
}

// CreateConfigFromModel creates an appropriate configuration based on model inspection
func CreateConfigFromModel(modelPath string) (ai.ModelConfig, error) {
	modelInfo, err := InspectONNXModel(modelPath)
	if err != nil {
		return ai.ModelConfig{}, err
	}

	// Get default config for detected architecture
	config := GetDefaultConfigForArchitecture(modelInfo.Architecture)

	// Override name with the model file name
	config.Name = strings.TrimSuffix(filepath.Base(modelPath), ".onnx")

	return config, nil
}

// ValidateModelCompatibility checks if a model is compatible with the engine
func ValidateModelCompatibility(modelPath string) error {
	modelInfo, err := InspectONNXModel(modelPath)
	if err != nil {
		return err
	}

	// Check for required inputs
	hasInputIds := false
	for _, input := range modelInfo.Inputs {
		if strings.Contains(strings.ToLower(input), "input_ids") {
			hasInputIds = true
			break
		}
	}

	if !hasInputIds {
		return fmt.Errorf("model does not have required 'input_ids' input, found inputs: %v", modelInfo.Inputs)
	}

	// Check for expected outputs
	if len(modelInfo.Outputs) == 0 {
		return fmt.Errorf("model has no outputs")
	}

	// Check architecture support
	if modelInfo.Architecture == ArchitectureUnknown {
		// This is a warning, not an error - we can still try to use the model
		fmt.Printf("Warning: Unknown model architecture, using default configuration\n")
	}

	return nil
}

// GetRecommendedBatchSize returns the recommended batch size for a model architecture
func GetRecommendedBatchSize(arch ModelArchitecture, availableMemoryMB int) int {
	baseSize := 16 // Default batch size

	switch arch {
	case ArchitectureBERT, ArchitectureRoBERTa:
		// BERT and RoBERTa are memory-intensive
		baseSize = 8
	case ArchitectureDistilBERT, ArchitectureAllMiniLM:
		// Smaller models can handle larger batches
		baseSize = 32
	case ArchitectureSentenceT5:
		// T5 models vary widely
		baseSize = 16
	case ArchitectureE5, ArchitectureBGE:
		// Modern sentence transformers
		baseSize = 24
	}

	// Adjust based on available memory
	if availableMemoryMB < 2048 {
		baseSize = max(1, baseSize/4)
	} else if availableMemoryMB < 4096 {
		baseSize = max(1, baseSize/2)
	} else if availableMemoryMB > 16384 {
		baseSize = baseSize * 2
	}

	return baseSize
}

// EstimateModelMemoryUsage estimates the memory usage for a model and batch size
func EstimateModelMemoryUsage(modelPath string, batchSize int, maxTokens int) (int64, error) {
	fileInfo, err := os.Stat(modelPath)
	if err != nil {
		return 0, err
	}

	modelSize := fileInfo.Size()

	// Rough estimation: model size + activation memory
	// Activation memory depends on batch size, sequence length, and hidden dimensions
	activationMemory := int64(batchSize * maxTokens * 768 * 4) // Assume 768 hidden dim, 4 bytes per float

	// Add some overhead for ONNX Runtime
	overhead := modelSize / 10

	return modelSize + activationMemory + overhead, nil
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
