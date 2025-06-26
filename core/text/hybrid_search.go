package text

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/core/ai"
)

// HybridSearchManager implements hybrid vector and text search
type HybridSearchManager struct {
	mu           sync.RWMutex
	vectorStore  core.VectorStore
	textIndex    *EnhancedBM25Index
	modelManager ai.ModelManager
	defaultModel string

	// Fusion configuration
	defaultWeights ai.SearchWeights
	fusionAlgo     ai.FusionAlgorithm

	// Caching
	queryCache   *QueryCache
	cacheEnabled bool
	cacheTTL     time.Duration

	// Statistics
	stats *SearchStatistics
}

// NewHybridSearchManager creates a new hybrid search manager
func NewHybridSearchManager(vectorStore core.VectorStore, modelManager ai.ModelManager) *HybridSearchManager {
	return &HybridSearchManager{
		vectorStore:  vectorStore,
		textIndex:    NewEnhancedBM25Index(),
		modelManager: modelManager,
		defaultModel: "all-MiniLM-L6-v2",
		defaultWeights: ai.SearchWeights{
			Vector:    0.7,
			Text:      0.3,
			Freshness: 0.0,
			Authority: 0.0,
		},
		fusionAlgo:   NewReciprocalRankFusion(),
		queryCache:   NewQueryCache(1000), // Cache up to 1000 queries
		cacheEnabled: true,
		cacheTTL:     5 * time.Minute,
		stats:        NewSearchStatistics(),
	}
}

// Search performs hybrid search combining vector and text results
func (hsm *HybridSearchManager) Search(ctx context.Context, req ai.HybridSearchRequest) (ai.HybridSearchResult, error) {
	startTime := time.Now()

	// Check cache if enabled
	if hsm.cacheEnabled {
		if cached, found := hsm.queryCache.Get(req); found {
			hsm.stats.RecordCacheHit()
			return cached, nil
		}
	}

	// Process query
	queryInfo := hsm.processQuery(req.Query, req.Options)

	// Perform parallel searches
	var vectorResults, textResults []ai.SearchResult
	var vectorErr, textErr error
	var vectorTime, textTime time.Duration

	var wg sync.WaitGroup
	wg.Add(2)

	// Vector search
	go func() {
		defer wg.Done()
		start := time.Now()
		vectorResults, vectorErr = hsm.performVectorSearch(ctx, req, queryInfo)
		vectorTime = time.Since(start)
	}()

	// Text search
	go func() {
		defer wg.Done()
		start := time.Now()
		textResults, textErr = hsm.performTextSearch(ctx, req, queryInfo)
		textTime = time.Since(start)
	}()

	wg.Wait()

	// Handle errors
	if vectorErr != nil && textErr != nil {
		return ai.HybridSearchResult{}, fmt.Errorf("both searches failed: vector=%v, text=%v", vectorErr, textErr)
	}

	// Fuse results
	fusionStart := time.Now()
	weights := req.Weights
	if weights.Vector == 0 && weights.Text == 0 {
		weights = hsm.defaultWeights
	}

	var fusedResults []ai.SearchResult
	if req.Options.FusionAlgorithm == "linear" {
		fusedResults = hsm.linearFusion(vectorResults, textResults, weights)
	} else {
		fusedResults = hsm.fusionAlgo.Fuse(vectorResults, textResults, weights)
	}
	fusionTime := time.Since(fusionStart)

	// Re-rank if requested
	var rerankTime time.Duration
	if req.Options.Rerank {
		rerankStart := time.Now()
		fusedResults = hsm.rerankResults(fusedResults, queryInfo)
		rerankTime = time.Since(rerankStart)
	}

	// Apply limit
	if req.Limit > 0 && len(fusedResults) > req.Limit {
		fusedResults = fusedResults[:req.Limit]
	}

	// Build result
	result := ai.HybridSearchResult{
		Results:   fusedResults,
		QueryInfo: queryInfo,
		Performance: ai.SearchPerf{
			TotalTimeMs:    time.Since(startTime).Milliseconds(),
			VectorSearchMs: vectorTime.Milliseconds(),
			TextSearchMs:   textTime.Milliseconds(),
			FusionMs:       fusionTime.Milliseconds(),
			RerankMs:       rerankTime.Milliseconds(),
		},
	}

	// Add debug info if requested
	if req.Options.IncludeExplanation {
		result.DebugInfo = &ai.DebugInfo{
			VectorCandidates: len(vectorResults),
			TextCandidates:   len(textResults),
			FusionCandidates: len(fusedResults),
			FinalResults:     len(result.Results),
		}
	}

	// Cache result
	if hsm.cacheEnabled {
		hsm.queryCache.Set(req, result, hsm.cacheTTL)
	}

	// Update statistics
	hsm.stats.RecordSearch(time.Since(startTime), len(result.Results) > 0)

	return result, nil
}

// AddDocuments adds documents to both vector and text indices
func (hsm *HybridSearchManager) AddDocuments(ctx context.Context, docs []ai.Document) error {
	hsm.mu.Lock()
	defer hsm.mu.Unlock()

	// Get embedding engine
	engine, err := hsm.modelManager.GetEngine(hsm.defaultModel)
	if err != nil {
		return fmt.Errorf("failed to get embedding engine: %w", err)
	}

	// Process documents in batches
	batchSize := 32
	for i := 0; i < len(docs); i += batchSize {
		end := i + batchSize
		if end > len(docs) {
			end = len(docs)
		}

		batch := docs[i:end]

		// Generate embeddings
		contents := make([]string, len(batch))
		for j, doc := range batch {
			contents[j] = doc.Content
		}

		embeddings, err := engine.EmbedBatch(ctx, contents, batchSize)
		if err != nil {
			return fmt.Errorf("failed to generate embeddings: %w", err)
		}

		// Add to vector store
		vectors := make([]core.Vector, len(batch))
		for j, doc := range batch {
			// Convert metadata to map[string]string for vector store
			metadata := make(map[string]string)
			for k, v := range doc.Metadata {
				if str, ok := v.(string); ok {
					metadata[k] = str
				} else {
					metadata[k] = fmt.Sprintf("%v", v)
				}
			}

			vectors[j] = core.Vector{
				ID:       doc.ID,
				Values:   embeddings[j],
				Metadata: metadata,
			}
		}

		if err := hsm.vectorStore.AddVectorsBatch(ctx, "default", vectors); err != nil {
			return fmt.Errorf("failed to add vectors: %w", err)
		}

		// Add to text index
		if err := hsm.textIndex.Index(ctx, batch); err != nil {
			return fmt.Errorf("failed to index text: %w", err)
		}
	}

	// Clear cache after adding documents
	hsm.queryCache.Clear()

	return nil
}

// UpdateFusionWeights updates the fusion weights
func (hsm *HybridSearchManager) UpdateFusionWeights(weights ai.SearchWeights) error {
	hsm.mu.Lock()
	defer hsm.mu.Unlock()

	// Validate weights
	total := weights.Vector + weights.Text + weights.Freshness + weights.Authority
	if math.Abs(total-1.0) > 0.01 {
		return fmt.Errorf("weights must sum to 1.0, got %f", total)
	}

	hsm.defaultWeights = weights
	hsm.queryCache.Clear() // Clear cache when weights change

	return nil
}

// GetStats returns search statistics
func (hsm *HybridSearchManager) GetStats() ai.SearchStats {
	return hsm.stats.GetStats()
}

// performVectorSearch executes vector similarity search
func (hsm *HybridSearchManager) performVectorSearch(ctx context.Context, req ai.HybridSearchRequest, queryInfo ai.QueryInfo) ([]ai.SearchResult, error) {
	// Get embedding engine
	engine, err := hsm.modelManager.GetEngine(hsm.defaultModel)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding engine: %w", err)
	}

	// Generate query embedding
	embeddings, err := engine.Embed(ctx, []string{queryInfo.Processed})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embedding generated")
	}

	// Convert filters from map[string]interface{} to map[string]string
	filters := make(map[string]string)
	for k, v := range req.Filters {
		if str, ok := v.(string); ok {
			filters[k] = str
		}
	}

	// Perform vector search
	searchReq := core.SearchRequest{
		Query:          embeddings[0],
		TopK:           req.Limit * 2, // Get more candidates for fusion
		Filter:         filters,
		IncludeVectors: false,
	}

	vectorResults, err := hsm.vectorStore.Search(ctx, "default", searchReq)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// Convert to AI search results
	results := make([]ai.SearchResult, len(vectorResults))
	for i, vr := range vectorResults {
		// Convert metadata from map[string]string to map[string]interface{}
		metadata := make(map[string]interface{})
		for k, v := range vr.Metadata {
			metadata[k] = v
		}

		results[i] = ai.SearchResult{
			ID:       vr.ID,
			Score:    float64(vr.Score),
			Metadata: metadata,
			Explanation: &ai.SearchExplanation{
				VectorScore: float64(vr.Score),
				SemanticSim: float64(vr.Score),
			},
		}

		// Retrieve content from text index
		if doc, exists := hsm.textIndex.documents[vr.ID]; exists {
			results[i].Content = doc.Content
		}
	}

	return results, nil
}

// performTextSearch executes BM25 text search
func (hsm *HybridSearchManager) performTextSearch(ctx context.Context, req ai.HybridSearchRequest, queryInfo ai.QueryInfo) ([]ai.SearchResult, error) {
	// Use enhanced search if options are specified
	if req.Options.ExpandQuery || len(req.Filters) > 0 {
		enhancedReq := EnhancedSearchRequest{
			Query:       queryInfo.Processed,
			Limit:       req.Limit * 2,
			MinScore:    req.Options.MinScore,
			ExpandQuery: req.Options.ExpandQuery,
			EnableFuzzy: false, // Can be enabled based on options
		}

		return hsm.textIndex.SearchWithOptions(ctx, enhancedReq)
	}

	// Regular BM25 search
	return hsm.textIndex.Search(ctx, queryInfo.Processed, req.Limit*2)
}

// processQuery analyzes and processes the search query
func (hsm *HybridSearchManager) processQuery(query string, options ai.SearchOptions) ai.QueryInfo {
	// Basic query processing
	processed := query

	// Parse query for special syntax
	parsedQuery := hsm.textIndex.parseQuery(query)

	// Expand query if requested
	if options.ExpandQuery {
		parsedQuery = hsm.textIndex.expandQuery(parsedQuery)
		processed = parsedQuery.ProcessedQuery
	}

	// Extract entities (simplified - in production use NER)
	entities := make([]ai.Entity, 0)
	// This would use the content analyzer for entity extraction

	// Determine query intent
	intent := ai.QueryIntent{
		Type:       "general",
		Confidence: 0.8,
	}

	// Query expansion info
	expansion := ai.QueryExpansion{
		Synonyms:      []string{},
		Concepts:      []string{},
		RelatedTerms:  []string{},
		WeightedTerms: make(map[string]float64),
	}

	return ai.QueryInfo{
		Original:  query,
		Processed: processed,
		Intent:    intent,
		Entities:  entities,
		Expansion: expansion,
	}
}

// linearFusion performs simple linear combination of scores
func (hsm *HybridSearchManager) linearFusion(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Create score map
	docScores := make(map[string]*ai.SearchResult)

	// Add vector results
	for _, result := range vectorResults {
		r := result // Copy
		r.Score = result.Score * weights.Vector
		docScores[result.ID] = &r
	}

	// Add text results
	for _, result := range textResults {
		if existing, exists := docScores[result.ID]; exists {
			existing.Score += result.Score * weights.Text
			// Merge explanation
			if existing.Explanation != nil && result.Explanation != nil {
				existing.Explanation.TextScore = result.Explanation.TextScore
				existing.Explanation.FusionScore = existing.Score
				existing.Explanation.MatchedTerms = result.Explanation.MatchedTerms
			}
		} else {
			r := result // Copy
			r.Score = result.Score * weights.Text
			docScores[result.ID] = &r
		}
	}

	// Convert to slice and sort
	results := make([]ai.SearchResult, 0, len(docScores))
	for _, result := range docScores {
		results = append(results, *result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// rerankResults applies additional ranking logic
func (hsm *HybridSearchManager) rerankResults(results []ai.SearchResult, queryInfo ai.QueryInfo) []ai.SearchResult {
	// Simple re-ranking based on metadata and query intent
	// In production, this could use a learned ranking model

	reranked := make([]ai.SearchResult, len(results))
	copy(reranked, results)

	// Apply boosts based on metadata
	for i := range reranked {
		boost := 1.0

		// Boost recent documents
		if timestamp, ok := reranked[i].Metadata["timestamp"].(int64); ok {
			age := time.Since(time.Unix(timestamp, 0))
			if age < 24*time.Hour {
				boost *= 1.2
			} else if age < 7*24*time.Hour {
				boost *= 1.1
			}
		}

		// Boost based on authority/source
		if source, ok := reranked[i].Metadata["source"].(string); ok {
			if source == "official" || source == "verified" {
				boost *= 1.3
			}
		}

		reranked[i].Score *= boost
	}

	// Re-sort after re-ranking
	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].Score > reranked[j].Score
	})

	return reranked
}

// ReciprocalRankFusion implements the RRF algorithm
type ReciprocalRankFusion struct {
	k float64 // Constant for RRF (default: 60)
}

// NewReciprocalRankFusion creates a new RRF fusion algorithm
func NewReciprocalRankFusion() *ReciprocalRankFusion {
	return &ReciprocalRankFusion{k: 60}
}

// Fuse combines results using Reciprocal Rank Fusion
func (rrf *ReciprocalRankFusion) Fuse(vectorResults, textResults []ai.SearchResult, weights ai.SearchWeights) []ai.SearchResult {
	// Calculate RRF scores
	docScores := make(map[string]float64)
	docContent := make(map[string]*ai.SearchResult)

	// Process vector results
	for rank, result := range vectorResults {
		score := weights.Vector * (1.0 / (float64(rank+1) + rrf.k))
		docScores[result.ID] = score
		r := result // Copy
		docContent[result.ID] = &r
	}

	// Process text results
	for rank, result := range textResults {
		score := weights.Text * (1.0 / (float64(rank+1) + rrf.k))
		if existing, exists := docScores[result.ID]; exists {
			docScores[result.ID] = existing + score
			// Update content if not present
			if docContent[result.ID].Content == "" {
				docContent[result.ID].Content = result.Content
			}
			// Merge explanations
			if docContent[result.ID].Explanation != nil && result.Explanation != nil {
				docContent[result.ID].Explanation.TextScore = result.Explanation.TextScore
				docContent[result.ID].Explanation.MatchedTerms = result.Explanation.MatchedTerms
			}
		} else {
			docScores[result.ID] = score
			r := result // Copy
			docContent[result.ID] = &r
		}
	}

	// Convert to slice and sort
	results := make([]ai.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		result := *docContent[docID]
		result.Score = score
		if result.Explanation != nil {
			result.Explanation.FusionScore = score
		}
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// GetName returns the algorithm name
func (rrf *ReciprocalRankFusion) GetName() string {
	return "reciprocal_rank_fusion"
}

// GetParameters returns algorithm parameters
func (rrf *ReciprocalRankFusion) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"k": rrf.k,
	}
}

// QueryCache implements a simple LRU cache for search results
type QueryCache struct {
	mu       sync.RWMutex
	cache    map[string]cacheEntry
	maxSize  int
	eviction []string // Simple FIFO for now
}

type cacheEntry struct {
	result     ai.HybridSearchResult
	expiration time.Time
}

// NewQueryCache creates a new query cache
func NewQueryCache(maxSize int) *QueryCache {
	return &QueryCache{
		cache:    make(map[string]cacheEntry),
		maxSize:  maxSize,
		eviction: make([]string, 0, maxSize),
	}
}

// Get retrieves a cached result
func (qc *QueryCache) Get(req ai.HybridSearchRequest) (ai.HybridSearchResult, bool) {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	key := qc.makeKey(req)
	entry, exists := qc.cache[key]
	if !exists || time.Now().After(entry.expiration) {
		return ai.HybridSearchResult{}, false
	}

	return entry.result, true
}

// Set stores a result in cache
func (qc *QueryCache) Set(req ai.HybridSearchRequest, result ai.HybridSearchResult, ttl time.Duration) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	key := qc.makeKey(req)

	// Evict if at capacity
	if len(qc.cache) >= qc.maxSize && qc.cache[key].expiration.IsZero() {
		// Remove oldest entry
		if len(qc.eviction) > 0 {
			oldest := qc.eviction[0]
			delete(qc.cache, oldest)
			qc.eviction = qc.eviction[1:]
		}
	}

	qc.cache[key] = cacheEntry{
		result:     result,
		expiration: time.Now().Add(ttl),
	}
	qc.eviction = append(qc.eviction, key)
}

// Clear empties the cache
func (qc *QueryCache) Clear() {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	qc.cache = make(map[string]cacheEntry)
	qc.eviction = make([]string, 0, qc.maxSize)
}

// makeKey creates a cache key from request
func (qc *QueryCache) makeKey(req ai.HybridSearchRequest) string {
	// Simple key generation - in production use a proper hash
	return fmt.Sprintf("%s:%s:%d:%v:%v",
		req.Query,
		req.Collection,
		req.Limit,
		req.Weights,
		req.Options,
	)
}

// SearchStatistics tracks search performance metrics
type SearchStatistics struct {
	mu             sync.RWMutex
	totalQueries   int64
	successQueries int64
	totalLatency   time.Duration
	latencies      []time.Duration
	cacheHits      int64
	lastReset      time.Time
}

// NewSearchStatistics creates new search statistics
func NewSearchStatistics() *SearchStatistics {
	return &SearchStatistics{
		latencies: make([]time.Duration, 0, 1000),
		lastReset: time.Now(),
	}
}

// RecordSearch records a search operation
func (ss *SearchStatistics) RecordSearch(latency time.Duration, success bool) {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	ss.totalQueries++
	if success {
		ss.successQueries++
	}

	ss.totalLatency += latency
	ss.latencies = append(ss.latencies, latency)

	// Keep only last 1000 for percentile calculation
	if len(ss.latencies) > 1000 {
		ss.latencies = ss.latencies[len(ss.latencies)-1000:]
	}
}

// RecordCacheHit records a cache hit
func (ss *SearchStatistics) RecordCacheHit() {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	ss.cacheHits++
}

// GetStats returns current statistics
func (ss *SearchStatistics) GetStats() ai.SearchStats {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	var avgLatency time.Duration
	if ss.totalQueries > 0 {
		avgLatency = ss.totalLatency / time.Duration(ss.totalQueries)
	}

	var p95Latency time.Duration
	if len(ss.latencies) > 0 {
		sorted := make([]time.Duration, len(ss.latencies))
		copy(sorted, ss.latencies)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i] < sorted[j]
		})
		p95Index := int(float64(len(sorted)) * 0.95)
		if p95Index < len(sorted) {
			p95Latency = sorted[p95Index]
		}
	}

	var successRate float64
	if ss.totalQueries > 0 {
		successRate = float64(ss.successQueries) / float64(ss.totalQueries)
	}

	var cacheHitRate float64
	if ss.totalQueries > 0 {
		cacheHitRate = float64(ss.cacheHits) / float64(ss.totalQueries)
	}

	return ai.SearchStats{
		TotalQueries:   ss.totalQueries,
		AverageLatency: avgLatency,
		P95Latency:     p95Latency,
		SuccessRate:    successRate,
		CacheHitRate:   cacheHitRate,
	}
}
