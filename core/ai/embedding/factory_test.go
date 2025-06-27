package embedding

import (
	"testing"

	"github.com/dshills/EmbeddixDB/core/ai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCreateEngine(t *testing.T) {
	tests := []struct {
		name    string
		config  ai.ModelConfig
		wantErr bool
	}{
		{
			name: "ollama engine",
			config: ai.ModelConfig{
				Type:           "ollama",
				Path:           "nomic-embed-text",
				OllamaEndpoint: "http://localhost:11434",
			},
			wantErr: false,
		},
		{
			name: "onnx engine",
			config: ai.ModelConfig{
				Type: "onnx",
				Path: "", // Empty path for mock ONNX engine
			},
			wantErr: false,
		},
		{
			name: "unsupported engine",
			config: ai.ModelConfig{
				Type: "unknown",
				Path: "model",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine, err := CreateEngine(tt.config)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, engine)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, engine)
			}
		})
	}
}

func TestSupportedEngineTypes(t *testing.T) {
	types := SupportedEngineTypes()
	assert.Contains(t, types, "onnx")
	assert.Contains(t, types, "ollama")
	assert.Len(t, types, 2)
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  ai.ModelConfig
		wantErr bool
	}{
		{
			name: "valid onnx config",
			config: ai.ModelConfig{
				Type: "onnx",
				Path: "model.onnx",
			},
			wantErr: false,
		},
		{
			name: "onnx missing path",
			config: ai.ModelConfig{
				Type: "onnx",
			},
			wantErr: true,
		},
		{
			name: "valid ollama config",
			config: ai.ModelConfig{
				Type: "ollama",
				Path: "nomic-embed-text",
			},
			wantErr: false,
		},
		{
			name: "ollama missing path",
			config: ai.ModelConfig{
				Type: "ollama",
			},
			wantErr: true,
		},
		{
			name: "unsupported type",
			config: ai.ModelConfig{
				Type: "unknown",
				Path: "model",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConfig(tt.config)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestOllamaIntegration(t *testing.T) {
	// This test verifies that the Ollama engine can be created through the factory
	config := ai.ModelConfig{
		Type:           "ollama",
		Path:           "nomic-embed-text",
		OllamaEndpoint: "http://localhost:11434",
		Dimensions:     768,
		BatchSize:      32,
	}

	engine, err := CreateEngine(config)
	require.NoError(t, err)
	require.NotNil(t, engine)

	// Verify it's the correct type
	ollamaEngine, ok := engine.(*OllamaEmbeddingEngine)
	assert.True(t, ok)
	assert.NotNil(t, ollamaEngine)

	// Check model info
	info := engine.GetModelInfo()
	assert.Equal(t, "nomic-embed-text", info.Name)
	assert.Equal(t, 768, info.Dimension)
}