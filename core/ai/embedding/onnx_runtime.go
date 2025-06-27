package embedding

import (
	"fmt"
	"os"
	"path/filepath"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// RealONNXSession implements ONNXSession using actual ONNX Runtime
type RealONNXSession struct {
	session     *onnxruntime.DynamicAdvancedSession
	inputNames  []string
	outputNames []string
}

// NewRealONNXSession creates a new ONNX Runtime session from a model file
func NewRealONNXSession(modelPath string) (*RealONNXSession, error) {
	// Initialize ONNX Runtime (only needs to be done once)
	if !onnxruntime.IsInitialized() {
		err := onnxruntime.InitializeEnvironment()
		if err != nil {
			return nil, NewEmbeddingError("initialize", "onnx", ErrModelInitFailed, err.Error(), false)
		}
	}

	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, NewEmbeddingError("load", filepath.Base(modelPath), ErrModelNotFound, modelPath, false)
	}

	// Get absolute path
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, NewEmbeddingError("load", filepath.Base(modelPath), err, "failed to get absolute path", false)
	}

	// Get input/output info to determine names
	inputInfo, outputInfo, err := onnxruntime.GetInputOutputInfo(absPath)
	if err != nil {
		return nil, NewEmbeddingError("load", filepath.Base(modelPath), ErrCorruptedModel, err.Error(), false)
	}

	// Extract input and output names
	inputNames := make([]string, len(inputInfo))
	for i, info := range inputInfo {
		inputNames[i] = info.Name
	}

	outputNames := make([]string, len(outputInfo))
	for i, info := range outputInfo {
		outputNames[i] = info.Name
	}

	// Create session options for optimization
	options, err := onnxruntime.NewSessionOptions()
	if err != nil {
		return nil, NewEmbeddingError("initialize", filepath.Base(modelPath), ErrModelInitFailed, err.Error(), false)
	}
	defer options.Destroy()

	// Configure session for optimal inference performance
	err = options.SetIntraOpNumThreads(1) // Single thread for deterministic results
	if err != nil {
		return nil, fmt.Errorf("failed to set intra-op threads: %w", err)
	}

	err = options.SetInterOpNumThreads(1) // Single thread for deterministic results
	if err != nil {
		return nil, fmt.Errorf("failed to set inter-op threads: %w", err)
	}

	// Set optimization level
	err = options.SetGraphOptimizationLevel(onnxruntime.GraphOptimizationLevelEnableAll)
	if err != nil {
		return nil, fmt.Errorf("failed to set optimization level: %w", err)
	}

	// Create the dynamic session
	session, err := onnxruntime.NewDynamicAdvancedSession(absPath, inputNames, outputNames, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &RealONNXSession{
		session:     session,
		inputNames:  inputNames,
		outputNames: outputNames,
	}, nil
}

// Run executes inference on the ONNX model
func (s *RealONNXSession) Run(inputs []ONNXValue) ([]ONNXValue, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}

	// Convert our ONNXValue interface to actual ONNX Runtime values
	onnxInputs := make([]onnxruntime.Value, len(inputs))
	for i, input := range inputs {
		realTensor, ok := input.(*RealONNXTensor)
		if !ok {
			return nil, fmt.Errorf("input %d is not a RealONNXTensor", i)
		}
		onnxInputs[i] = realTensor.value
	}

	// Run inference - prepare empty outputs slice that will be allocated by the session
	outputs := make([]onnxruntime.Value, len(s.outputNames))
	err := s.session.Run(onnxInputs, outputs)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Convert outputs back to our interface
	result := make([]ONNXValue, len(outputs))
	for i, output := range outputs {
		result[i] = &RealONNXTensor{value: output}
	}

	return result, nil
}

// GetInputCount returns the number of model inputs
func (s *RealONNXSession) GetInputCount() int {
	return len(s.inputNames)
}

// GetOutputCount returns the number of model outputs
func (s *RealONNXSession) GetOutputCount() int {
	return len(s.outputNames)
}

// GetInputName returns the name of the input at the given index
func (s *RealONNXSession) GetInputName(index int) string {
	if index < 0 || index >= len(s.inputNames) {
		return ""
	}
	return s.inputNames[index]
}

// GetOutputName returns the name of the output at the given index
func (s *RealONNXSession) GetOutputName(index int) string {
	if index < 0 || index >= len(s.outputNames) {
		return ""
	}
	return s.outputNames[index]
}

// Destroy releases the ONNX session resources
func (s *RealONNXSession) Destroy() {
	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}
}

// RealONNXTensor implements ONNXValue using actual ONNX Runtime tensors
type RealONNXTensor struct {
	value onnxruntime.Value
}

// NewRealONNXTensor creates a new tensor from Go data
func NewRealONNXTensor(data interface{}, shape []int64) (*RealONNXTensor, error) {
	var value onnxruntime.Value
	var err error

	switch d := data.(type) {
	case []int64:
		tensor, err := onnxruntime.NewTensor(onnxruntime.NewShape(shape...), d)
		if err != nil {
			return nil, err
		}
		value = tensor
	case []float32:
		tensor, err := onnxruntime.NewTensor(onnxruntime.NewShape(shape...), d)
		if err != nil {
			return nil, err
		}
		value = tensor
	case []int32:
		tensor, err := onnxruntime.NewTensor(onnxruntime.NewShape(shape...), d)
		if err != nil {
			return nil, err
		}
		value = tensor
	default:
		return nil, fmt.Errorf("unsupported data type: %T", data)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create tensor: %w", err)
	}

	return &RealONNXTensor{value: value}, nil
}

// GetData returns the tensor data
func (t *RealONNXTensor) GetData() interface{} {
	// Try to cast to different tensor types
	if tensor, ok := t.value.(*onnxruntime.Tensor[float32]); ok {
		return tensor.GetData()
	}
	if tensor, ok := t.value.(*onnxruntime.Tensor[int64]); ok {
		return tensor.GetData()
	}
	if tensor, ok := t.value.(*onnxruntime.Tensor[int32]); ok {
		return tensor.GetData()
	}
	return nil
}

// GetShape returns the tensor shape
func (t *RealONNXTensor) GetShape() []int64 {
	// Try to cast to different tensor types
	if tensor, ok := t.value.(*onnxruntime.Tensor[float32]); ok {
		return []int64(tensor.GetShape())
	}
	if tensor, ok := t.value.(*onnxruntime.Tensor[int64]); ok {
		return []int64(tensor.GetShape())
	}
	if tensor, ok := t.value.(*onnxruntime.Tensor[int32]); ok {
		return []int64(tensor.GetShape())
	}
	return nil
}

// Destroy releases the tensor resources
func (t *RealONNXTensor) Destroy() {
	if t.value != nil {
		t.value.Destroy()
		t.value = nil
	}
}

// CreateInputTensorFromTokens creates an input tensor from tokenized text
func CreateInputTensorFromTokens(tokens [][]int64, inputName string) (*RealONNXTensor, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens provided")
	}

	batchSize := int64(len(tokens))
	seqLen := int64(len(tokens[0]))

	// Validate that all sequences have the same length
	for i, seq := range tokens {
		if int64(len(seq)) != seqLen {
			return nil, fmt.Errorf("sequence %d has length %d, expected %d", i, len(seq), seqLen)
		}
	}

	// Flatten tokens for tensor creation
	flatTokens := make([]int64, batchSize*seqLen)
	for i, seq := range tokens {
		copy(flatTokens[i*int(seqLen):(i+1)*int(seqLen)], seq)
	}

	// Create tensor with shape [batch_size, sequence_length]
	shape := []int64{batchSize, seqLen}
	return NewRealONNXTensor(flatTokens, shape)
}

// CreateAttentionMaskTensor creates an attention mask tensor
func CreateAttentionMaskTensor(masks [][]int64) (*RealONNXTensor, error) {
	if len(masks) == 0 {
		return nil, fmt.Errorf("no attention masks provided")
	}

	batchSize := int64(len(masks))
	seqLen := int64(len(masks[0]))

	// Flatten masks
	flatMasks := make([]int64, batchSize*seqLen)
	for i, mask := range masks {
		copy(flatMasks[i*int(seqLen):(i+1)*int(seqLen)], mask)
	}

	// Create tensor with shape [batch_size, sequence_length]
	shape := []int64{batchSize, seqLen}
	return NewRealONNXTensor(flatMasks, shape)
}

// ExtractEmbeddingsFromTensor extracts embeddings from model output tensor
func ExtractEmbeddingsFromTensor(output *RealONNXTensor, poolingStrategy string) ([][]float32, error) {
	data := output.GetData()
	shape := output.GetShape()

	// Ensure we have float32 data
	embeddings, ok := data.([]float32)
	if !ok {
		return nil, fmt.Errorf("expected float32 output, got %T", data)
	}

	// Handle different output shapes
	var batchSize, seqLen, hiddenSize int64

	switch len(shape) {
	case 2:
		// Shape: [batch_size, hidden_size] - already pooled
		batchSize = shape[0]
		hiddenSize = shape[1]
		seqLen = 1
	case 3:
		// Shape: [batch_size, sequence_length, hidden_size] - need pooling
		batchSize = shape[0]
		seqLen = shape[1]
		hiddenSize = shape[2]
	default:
		return nil, fmt.Errorf("unsupported output shape: %v", shape)
	}

	result := make([][]float32, batchSize)

	if seqLen == 1 {
		// Already pooled - just reshape
		for i := int64(0); i < batchSize; i++ {
			start := i * hiddenSize
			end := start + hiddenSize
			result[i] = make([]float32, hiddenSize)
			copy(result[i], embeddings[start:end])
		}
	} else {
		// Need to apply pooling strategy
		switch poolingStrategy {
		case "cls", "first":
			// Use first token (CLS token) embeddings
			for i := int64(0); i < batchSize; i++ {
				start := i*seqLen*hiddenSize + 0*hiddenSize // First token
				end := start + hiddenSize
				result[i] = make([]float32, hiddenSize)
				copy(result[i], embeddings[start:end])
			}
		case "mean", "average":
			// Mean pooling over sequence length
			for i := int64(0); i < batchSize; i++ {
				result[i] = make([]float32, hiddenSize)
				for j := int64(0); j < seqLen; j++ {
					start := i*seqLen*hiddenSize + j*hiddenSize
					for k := int64(0); k < hiddenSize; k++ {
						result[i][k] += embeddings[start+k]
					}
				}
				// Divide by sequence length for mean
				for k := int64(0); k < hiddenSize; k++ {
					result[i][k] /= float32(seqLen)
				}
			}
		case "max":
			// Max pooling over sequence length
			for i := int64(0); i < batchSize; i++ {
				result[i] = make([]float32, hiddenSize)
				// Initialize with first token values
				start := i * seqLen * hiddenSize
				copy(result[i], embeddings[start:start+hiddenSize])

				// Find max across sequence
				for j := int64(1); j < seqLen; j++ {
					tokenStart := i*seqLen*hiddenSize + j*hiddenSize
					for k := int64(0); k < hiddenSize; k++ {
						if embeddings[tokenStart+k] > result[i][k] {
							result[i][k] = embeddings[tokenStart+k]
						}
					}
				}
			}
		default:
			return nil, fmt.Errorf("unsupported pooling strategy: %s", poolingStrategy)
		}
	}

	return result, nil
}

// GetONNXRuntimeVersion returns the version of ONNX Runtime being used
func GetONNXRuntimeVersion() string {
	return onnxruntime.GetVersion()
}

// GetAvailableProviders returns the available execution providers
func GetAvailableProviders() []string {
	// The onnxruntime_go library doesn't expose this function directly
	// Return a default list of commonly available providers
	return []string{"CPUExecutionProvider"}
}
