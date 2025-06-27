package embedding

import (
	"fmt"

	"github.com/dshills/EmbeddixDB/core/ai"
)

// CreateEngine creates an embedding engine based on the model configuration
func CreateEngine(config ai.ModelConfig) (ai.EmbeddingEngine, error) {
	switch config.Type {
	case ai.ModelTypeONNX:
		return NewONNXEmbeddingEngine(config.Path, config)
	case ai.ModelTypeOllama:
		return NewOllamaEmbeddingEngine(&config)
	default:
		return nil, fmt.Errorf("unsupported embedding engine type: %q", config.Type)
	}
}

// SupportedEngineTypes returns the list of supported embedding engine types
func SupportedEngineTypes() []string {
	return []string{ai.ModelTypeONNX, ai.ModelTypeOllama}
}

// ValidateConfig validates the model configuration for the specified engine type
func ValidateConfig(config ai.ModelConfig) error {
	switch config.Type {
	case ai.ModelTypeONNX:
		if config.Path == "" {
			return fmt.Errorf("ONNX engine requires model path")
		}
	case ai.ModelTypeOllama:
		if config.Path == "" {
			return fmt.Errorf("Ollama engine requires model name in Path field")
		}
	default:
		return fmt.Errorf("unsupported engine type: %q", config.Type)
	}
	return nil
}