# Model Integration Guide

## Supported Model Formats

EmbeddixDB supports multiple model formats and frameworks for maximum flexibility:

### ONNX Runtime (Primary)
- **Format**: `.onnx` files
- **Benefits**: Cross-platform, optimized inference, broad model support
- **GPU Support**: CUDA, DirectML, CoreML
- **Recommended for**: Production deployments

### Hugging Face Transformers
- **Format**: PyTorch/TensorFlow models
- **Benefits**: Largest model ecosystem, easy integration
- **Conversion**: Automatic ONNX conversion available
- **Recommended for**: Development and experimentation

### Custom Models
- **Format**: Various (with adapters)
- **Benefits**: Full control over inference pipeline
- **Requirements**: Implementation of `EmbeddingEngine` interface
- **Recommended for**: Specialized use cases

## Built-in Model Registry

### Text Embedding Models

#### General Purpose
```yaml
# sentence-transformers/all-MiniLM-L6-v2
name: "sentence-transformers/all-MiniLM-L6-v2"
version: "1.0"
type: "text_embedding"
dimension: 384
max_tokens: 512
languages: ["en"]
size_mb: 22
use_case: "general_purpose"
accuracy_score: 0.85
speed_tokens_per_sec: 2500
license: "apache-2.0"
source: "huggingface"
tags: ["sentence-transformers", "lightweight", "fast"]

# sentence-transformers/all-mpnet-base-v2  
name: "sentence-transformers/all-mpnet-base-v2"
version: "1.0"
type: "text_embedding"
dimension: 768
max_tokens: 514
languages: ["en"]
size_mb: 420
use_case: "high_quality"
accuracy_score: 0.92
speed_tokens_per_sec: 1200
license: "apache-2.0"
source: "huggingface"
tags: ["sentence-transformers", "high-quality", "robust"]
```

#### Multilingual
```yaml
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
version: "1.0"
type: "text_embedding"
dimension: 384
max_tokens: 512
languages: ["en", "de", "fr", "es", "it", "zh", "ja", "ko", "ru"]
size_mb: 118
use_case: "multilingual"
accuracy_score: 0.88
speed_tokens_per_sec: 2000
license: "apache-2.0"
source: "huggingface"
tags: ["multilingual", "sentence-transformers"]

# sentence-transformers/distiluse-base-multilingual-cased-v2
name: "sentence-transformers/distiluse-base-multilingual-cased-v2"
version: "1.0"
type: "text_embedding"
dimension: 512
max_tokens: 512
languages: ["en", "de", "fr", "es", "it", "nl", "pl", "pt", "ru", "zh"]
size_mb: 540
use_case: "multilingual_quality"
accuracy_score: 0.90
speed_tokens_per_sec: 1500
license: "apache-2.0"
source: "huggingface"
```

#### Domain-Specific
```yaml
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
name: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
version: "1.0"
type: "text_embedding"
dimension: 768
max_tokens: 512
languages: ["en"]
size_mb: 420
use_case: "biomedical"
accuracy_score: 0.94
domain: "healthcare"
speed_tokens_per_sec: 1100
license: "mit"
source: "huggingface"
tags: ["biomedical", "healthcare", "pubmed"]

# sentence-transformers/allenai-specter
name: "sentence-transformers/allenai-specter"
version: "1.0"
type: "text_embedding"
dimension: 768
max_tokens: 512
languages: ["en"]
size_mb: 420
use_case: "scientific"
domain: "research"
accuracy_score: 0.91
speed_tokens_per_sec: 1200
license: "apache-2.0"
source: "huggingface"
tags: ["scientific", "research", "papers"]
```

### Multimodal Models

#### CLIP Models
```yaml
# openai/clip-vit-base-patch32
name: "openai/clip-vit-base-patch32"
version: "1.0"
type: "multimodal_embedding"
dimension: 512
modalities: ["text", "image"]
max_tokens: 77
languages: ["en"]
size_mb: 151
use_case: "text_image"
accuracy_score: 0.88
speed_tokens_per_sec: 800
license: "mit"
source: "openai"
tags: ["clip", "multimodal", "vision"]

# openai/clip-vit-large-patch14
name: "openai/clip-vit-large-patch14"  
version: "1.0"
type: "multimodal_embedding"
dimension: 768
modalities: ["text", "image"]
max_tokens: 77
languages: ["en"]
size_mb: 1700
use_case: "high_quality_multimodal"
accuracy_score: 0.93
speed_tokens_per_sec: 300
license: "mit"
source: "openai"
tags: ["clip", "multimodal", "high-quality"]
```

## Model Integration Implementation

### ONNX Model Loading

```go
package embedding

import (
    "context"
    "fmt"
    "github.com/yalue/onnxruntime_go"
)

type ONNXEmbeddingEngine struct {
    session    *onnxruntime.Session
    tokenizer  *Tokenizer
    config     ModelConfig
    stats      *InferenceStats
    warmupDone bool
    mutex      sync.RWMutex
}

func NewONNXEmbeddingEngine(modelPath string, config ModelConfig) (*ONNXEmbeddingEngine, error) {
    // Initialize ONNX Runtime
    onnxruntime.SetDefaultLogger(onnxruntime.LogSeverityWarning, "onnx")
    
    // Create session options
    options, err := onnxruntime.NewSessionOptions()
    if err != nil {
        return nil, fmt.Errorf("failed to create session options: %w", err)
    }
    defer options.Destroy()
    
    // Configure GPU if available
    if config.EnableGPU {
        err = options.AppendExecutionProviderCUDA(0)
        if err != nil {
            // Fall back to CPU
            fmt.Printf("GPU not available, using CPU: %v\n", err)
        }
    }
    
    // Set optimization level
    options.SetIntraOpNumThreads(config.NumThreads)
    options.SetGraphOptimizationLevel(onnxruntime.GraphOptimizationLevel(config.OptimizationLevel))
    
    // Load the model
    session, err := onnxruntime.NewSession(modelPath, options)
    if err != nil {
        return nil, fmt.Errorf("failed to create ONNX session: %w", err)
    }
    
    // Initialize tokenizer
    tokenizer, err := NewTokenizer(config.TokenizerPath)
    if err != nil {
        session.Destroy()
        return nil, fmt.Errorf("failed to initialize tokenizer: %w", err)
    }
    
    engine := &ONNXEmbeddingEngine{
        session:   session,
        tokenizer: tokenizer,
        config:    config,
        stats:     NewInferenceStats(),
    }
    
    return engine, nil
}

func (e *ONNXEmbeddingEngine) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    start := time.Now()
    defer func() {
        e.stats.RecordInference(len(texts), time.Since(start))
    }()
    
    // Ensure model is warmed up
    if !e.warmupDone {
        if err := e.Warm(ctx); err != nil {
            return nil, fmt.Errorf("model warmup failed: %w", err)
        }
    }
    
    // Tokenize inputs
    tokens, err := e.tokenizer.TokenizeBatch(texts, e.config.MaxTokens)
    if err != nil {
        return nil, fmt.Errorf("tokenization failed: %w", err)
    }
    
    // Prepare input tensors
    inputTensor, err := e.createInputTensor(tokens)
    if err != nil {
        return nil, fmt.Errorf("failed to create input tensor: %w", err)
    }
    defer inputTensor.Destroy()
    
    // Run inference
    outputs, err := e.session.Run([]onnxruntime.Value{inputTensor})
    if err != nil {
        return nil, fmt.Errorf("inference failed: %w", err)
    }
    defer func() {
        for _, output := range outputs {
            output.Destroy()
        }
    }()
    
    // Extract embeddings
    embeddings, err := e.extractEmbeddings(outputs[0])
    if err != nil {
        return nil, fmt.Errorf("failed to extract embeddings: %w", err)
    }
    
    // Normalize if configured
    if e.config.NormalizeEmbeddings {
        embeddings = normalizeEmbeddings(embeddings)
    }
    
    return embeddings, nil
}

func (e *ONNXEmbeddingEngine) createInputTensor(tokens [][]int64) (onnxruntime.Value, error) {
    batchSize := len(tokens)
    seqLen := len(tokens[0]) // Assuming all sequences have same length after padding
    
    // Flatten tokens for tensor creation
    flatTokens := make([]int64, batchSize*seqLen)
    for i, seq := range tokens {
        copy(flatTokens[i*seqLen:(i+1)*seqLen], seq)
    }
    
    // Create tensor
    shape := []int64{int64(batchSize), int64(seqLen)}
    tensor, err := onnxruntime.NewTensor(shape, flatTokens)
    if err != nil {
        return nil, fmt.Errorf("failed to create tensor: %w", err)
    }
    
    return tensor, nil
}

func (e *ONNXEmbeddingEngine) extractEmbeddings(output onnxruntime.Value) ([][]float32, error) {
    // Get output as float32 slice
    data := output.GetData().([]float32)
    shape := output.GetShape()
    
    batchSize := int(shape[0])
    embeddingDim := int(shape[len(shape)-1])
    
    // Reshape to batch of embeddings
    embeddings := make([][]float32, batchSize)
    for i := 0; i < batchSize; i++ {
        start := i * embeddingDim
        end := start + embeddingDim
        embeddings[i] = make([]float32, embeddingDim)
        copy(embeddings[i], data[start:end])
    }
    
    return embeddings, nil
}

func (e *ONNXEmbeddingEngine) Warm(ctx context.Context) error {
    e.mutex.Lock()
    defer e.mutex.Unlock()
    
    if e.warmupDone {
        return nil
    }
    
    // Run a dummy inference to warm up the model
    warmupTexts := []string{"This is a warmup text for model initialization."}
    _, err := e.Embed(ctx, warmupTexts)
    if err != nil {
        return fmt.Errorf("warmup inference failed: %w", err)
    }
    
    e.warmupDone = true
    return nil
}
```

### Hugging Face Integration

```go
package embedding

import (
    "context"
    "fmt"
    "os/exec"
    "encoding/json"
)

type HuggingFaceEngine struct {
    modelName   string
    pythonPath  string
    scriptPath  string
    config      ModelConfig
    process     *exec.Cmd
    stdin       io.WriteCloser
    stdout      io.ReadCloser
    ready       bool
    mutex       sync.RWMutex
}

func NewHuggingFaceEngine(modelName string, config ModelConfig) (*HuggingFaceEngine, error) {
    engine := &HuggingFaceEngine{
        modelName:  modelName,
        pythonPath: config.PythonPath,
        scriptPath: config.HFScriptPath,
        config:     config,
    }
    
    if err := engine.start(); err != nil {
        return nil, fmt.Errorf("failed to start HuggingFace engine: %w", err)
    }
    
    return engine, nil
}

func (h *HuggingFaceEngine) start() error {
    // Start Python subprocess with embedding script
    cmd := exec.Command(h.pythonPath, h.scriptPath, h.modelName)
    
    stdin, err := cmd.StdinPipe()
    if err != nil {
        return err
    }
    h.stdin = stdin
    
    stdout, err := cmd.StdoutPipe()
    if err != nil {
        return err
    }
    h.stdout = stdout
    
    if err := cmd.Start(); err != nil {
        return err
    }
    h.process = cmd
    
    // Wait for ready signal
    decoder := json.NewDecoder(h.stdout)
    var response map[string]interface{}
    if err := decoder.Decode(&response); err != nil {
        return fmt.Errorf("failed to read ready signal: %w", err)
    }
    
    if status, ok := response["status"].(string); !ok || status != "ready" {
        return fmt.Errorf("unexpected ready response: %v", response)
    }
    
    h.ready = true
    return nil
}

func (h *HuggingFaceEngine) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    h.mutex.RLock()
    defer h.mutex.RUnlock()
    
    if !h.ready {
        return nil, fmt.Errorf("engine not ready")
    }
    
    // Send request to Python process
    request := map[string]interface{}{
        "action": "embed",
        "texts":  texts,
    }
    
    encoder := json.NewEncoder(h.stdin)
    if err := encoder.Encode(request); err != nil {
        return nil, fmt.Errorf("failed to send request: %w", err)
    }
    
    // Read response
    decoder := json.NewDecoder(h.stdout)
    var response map[string]interface{}
    if err := decoder.Decode(&response); err != nil {
        return nil, fmt.Errorf("failed to read response: %w", err)
    }
    
    if errMsg, ok := response["error"].(string); ok {
        return nil, fmt.Errorf("embedding error: %s", errMsg)
    }
    
    // Parse embeddings
    embeddingsRaw, ok := response["embeddings"].([]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid embeddings format")
    }
    
    embeddings := make([][]float32, len(embeddingsRaw))
    for i, embRaw := range embeddingsRaw {
        embSlice := embRaw.([]interface{})
        embedding := make([]float32, len(embSlice))
        for j, val := range embSlice {
            embedding[j] = float32(val.(float64))
        }
        embeddings[i] = embedding
    }
    
    return embeddings, nil
}
```

### Python Embedding Script (for HuggingFace)

```python
#!/usr/bin/env python3
"""
Embedding service for HuggingFace models
Communicates with Go service via JSON over stdin/stdout
"""

import sys
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class EmbeddingService:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Try SentenceTransformers first
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_type = 'sentence_transformer'
        except:
            # Fall back to raw transformers
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model_type = 'transformer'
        
        # Warm up
        self.embed(["warmup text"])
    
    def embed(self, texts):
        if self.model_type == 'sentence_transformer':
            return self._embed_sentence_transformer(texts)
        else:
            return self._embed_transformer(texts)
    
    def _embed_sentence_transformer(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32
            )
        return embeddings.tolist()
    
    def _embed_transformer(self, texts):
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding or mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                # Normalize
                embedding = embedding / embedding.norm()
                embeddings.append(embedding.cpu().numpy().tolist())
        
        return embeddings

def main():
    if len(sys.argv) != 2:
        print("Usage: embedding_service.py <model_name>", file=sys.stderr)
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    try:
        service = EmbeddingService(model_name)
        
        # Signal ready
        print(json.dumps({"status": "ready"}))
        sys.stdout.flush()
        
        # Process requests
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                
                if request["action"] == "embed":
                    texts = request["texts"]
                    embeddings = service.embed(texts)
                    response = {"embeddings": embeddings}
                else:
                    response = {"error": f"Unknown action: {request['action']}"}
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except Exception as e:
                error_response = {"error": str(e)}
                print(json.dumps(error_response))
                sys.stdout.flush()
                
    except Exception as e:
        print(json.dumps({"error": f"Failed to initialize model: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Model Management

### Model Registry Implementation

```go
package models

type ModelManager struct {
    registry   *ModelRegistry
    cache      *ModelCache
    downloader *ModelDownloader
    engines    map[string]EmbeddingEngine
    mutex      sync.RWMutex
}

func NewModelManager(config ManagerConfig) *ModelManager {
    return &ModelManager{
        registry:   NewModelRegistry(config.RegistryPath),
        cache:      NewModelCache(config.CacheDir, config.MaxCacheSize),
        downloader: NewModelDownloader(config.DownloadDir),
        engines:    make(map[string]EmbeddingEngine),
    }
}

func (mm *ModelManager) LoadModel(ctx context.Context, modelName string, config ModelConfig) error {
    mm.mutex.Lock()
    defer mm.mutex.Unlock()
    
    // Check if already loaded
    if _, exists := mm.engines[modelName]; exists {
        return nil
    }
    
    // Get model metadata
    modelInfo, err := mm.registry.GetModel(modelName)
    if err != nil {
        return fmt.Errorf("model not found in registry: %w", err)
    }
    
    // Download if not cached
    modelPath, err := mm.ensureModelAvailable(ctx, modelInfo)
    if err != nil {
        return fmt.Errorf("failed to ensure model availability: %w", err)
    }
    
    // Create appropriate engine based on model type
    var engine EmbeddingEngine
    switch modelInfo.Type {
    case "onnx":
        engine, err = NewONNXEmbeddingEngine(modelPath, config)
    case "huggingface":
        engine, err = NewHuggingFaceEngine(modelName, config)
    case "custom":
        engine, err = mm.createCustomEngine(modelName, modelPath, config)
    default:
        return fmt.Errorf("unsupported model type: %s", modelInfo.Type)
    }
    
    if err != nil {
        return fmt.Errorf("failed to create embedding engine: %w", err)
    }
    
    // Warm up the model
    if err := engine.Warm(ctx); err != nil {
        engine.Close()
        return fmt.Errorf("model warmup failed: %w", err)
    }
    
    mm.engines[modelName] = engine
    return nil
}

func (mm *ModelManager) ensureModelAvailable(ctx context.Context, modelInfo ModelInfo) (string, error) {
    // Check cache first
    if cachedPath, exists := mm.cache.GetPath(modelInfo.Name); exists {
        return cachedPath, nil
    }
    
    // Download model
    downloadPath, err := mm.downloader.Download(ctx, modelInfo)
    if err != nil {
        return "", fmt.Errorf("download failed: %w", err)
    }
    
    // Verify checksum
    if err := mm.verifyChecksum(downloadPath, modelInfo.Checksum); err != nil {
        return "", fmt.Errorf("checksum verification failed: %w", err)
    }
    
    // Add to cache
    mm.cache.Add(modelInfo.Name, downloadPath)
    
    return downloadPath, nil
}

func (mm *ModelManager) GetEngine(modelName string) (EmbeddingEngine, error) {
    mm.mutex.RLock()
    defer mm.mutex.RUnlock()
    
    engine, exists := mm.engines[modelName]
    if !exists {
        return nil, fmt.Errorf("model not loaded: %s", modelName)
    }
    
    return engine, nil
}
```

### Model Performance Monitoring

```go
type ModelMonitor struct {
    engines map[string]*EngineMonitor
    metrics *MetricsCollector
    alerts  *AlertManager
    mutex   sync.RWMutex
}

type EngineMonitor struct {
    modelName      string
    engine         EmbeddingEngine
    healthChecker  *HealthChecker
    perfTracker    *PerformanceTracker
    resourceTracker *ResourceTracker
    lastCheck      time.Time
}

func (em *EngineMonitor) CheckHealth() HealthStatus {
    start := time.Now()
    
    // Test inference
    testTexts := []string{"health check test"}
    _, err := em.engine.Embed(context.Background(), testTexts)
    
    latency := time.Since(start)
    
    status := HealthStatus{
        ModelName:  em.modelName,
        Healthy:    err == nil,
        Latency:    latency,
        Timestamp:  time.Now(),
        Error:      err,
    }
    
    if err != nil {
        status.Healthy = false
        status.ErrorMessage = err.Error()
    }
    
    // Check resource usage
    resources := em.resourceTracker.GetCurrentUsage()
    status.CPUUsage = resources.CPU
    status.MemoryUsage = resources.Memory
    status.GPUUsage = resources.GPU
    
    // Determine overall health
    if latency > 5*time.Second {
        status.Healthy = false
        status.ErrorMessage = "high latency detected"
    }
    
    if resources.Memory > 0.9 { // 90% memory usage
        status.Healthy = false
        status.ErrorMessage = "high memory usage"
    }
    
    return status
}
```

## Model Optimization

### ONNX Optimization

```go
func OptimizeONNXModel(inputPath, outputPath string, config OptimizationConfig) error {
    // Load model
    model, err := onnx.LoadModel(inputPath)
    if err != nil {
        return err
    }
    
    // Apply optimizations
    optimizer := onnx.NewOptimizer()
    
    if config.FuseLayers {
        optimizer.AddPass(onnx.LayerFusionPass{})
    }
    
    if config.QuantizeWeights {
        optimizer.AddPass(onnx.QuantizationPass{
            Mode: config.QuantizationMode,
            Bits: config.QuantizationBits,
        })
    }
    
    if config.OptimizeForInference {
        optimizer.AddPass(onnx.InferenceOptimizationPass{})
    }
    
    // Run optimization
    optimizedModel, err := optimizer.Optimize(model)
    if err != nil {
        return err
    }
    
    // Save optimized model
    return onnx.SaveModel(optimizedModel, outputPath)
}
```

### Dynamic Batching

```go
type BatchingEngine struct {
    baseEngine    EmbeddingEngine
    batchSize     int
    maxWaitTime   time.Duration
    requestQueue  chan EmbedRequest
    responseChans map[string]chan EmbedResponse
    mutex         sync.RWMutex
}

type EmbedRequest struct {
    ID       string
    Texts    []string
    Response chan EmbedResponse
}

type EmbedResponse struct {
    Embeddings [][]float32
    Error      error
}

func (be *BatchingEngine) Start() {
    go be.processBatches()
}

func (be *BatchingEngine) processBatches() {
    ticker := time.NewTicker(be.maxWaitTime)
    defer ticker.Stop()
    
    var batch []EmbedRequest
    
    for {
        select {
        case req := <-be.requestQueue:
            batch = append(batch, req)
            
            if len(batch) >= be.batchSize {
                be.processBatch(batch)
                batch = nil
            }
            
        case <-ticker.C:
            if len(batch) > 0 {
                be.processBatch(batch)
                batch = nil
            }
        }
    }
}

func (be *BatchingEngine) processBatch(batch []EmbedRequest) {
    // Combine all texts
    var allTexts []string
    var requestSizes []int
    
    for _, req := range batch {
        allTexts = append(allTexts, req.Texts...)
        requestSizes = append(requestSizes, len(req.Texts))
    }
    
    // Process batch
    embeddings, err := be.baseEngine.Embed(context.Background(), allTexts)
    
    // Distribute results
    offset := 0
    for i, req := range batch {
        size := requestSizes[i]
        
        if err != nil {
            req.Response <- EmbedResponse{Error: err}
        } else {
            reqEmbeddings := embeddings[offset : offset+size]
            req.Response <- EmbedResponse{Embeddings: reqEmbeddings}
        }
        
        offset += size
        close(req.Response)
    }
}
```

This model integration guide provides comprehensive coverage of embedding model support, from basic ONNX integration to advanced optimization techniques. The modular design allows for easy extension to support new model formats and optimization strategies.