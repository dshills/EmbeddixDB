package index

import (
	"encoding/json"
	"fmt"
	"sort"
	
	"github.com/dshills/EmbeddixDB/core"
)

// HNSWIndex implements the HNSW (Hierarchical Navigable Small World) algorithm
type HNSWIndex struct {
	graph *HNSWGraph
}

// NewHNSWIndex creates a new HNSW index
func NewHNSWIndex(dimension int, distanceMetric core.DistanceMetric, config HNSWConfig) *HNSWIndex {
	return &HNSWIndex{
		graph: NewHNSWGraph(dimension, distanceMetric, config),
	}
}

// Add adds a vector to the HNSW index
func (h *HNSWIndex) Add(vector core.Vector) error {
	if err := core.ValidateVector(vector); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}
	
	if err := core.ValidateVectorDimension(vector, h.graph.dimension); err != nil {
		return fmt.Errorf("dimension mismatch: %w", err)
	}
	
	// Assign level to new node
	level := h.graph.assignLevel()
	newNode := NewHNSWNode(vector, level)
	
	h.graph.mu.Lock()
	defer h.graph.mu.Unlock()
	
	// If this is the first node, make it the entry point
	if h.graph.size == 0 {
		h.graph.entryPoint = newNode
		h.graph.nodes[vector.ID] = newNode
		h.graph.size++
		return nil
	}
	
	// Find closest points and connect
	if err := h.insertNode(newNode); err != nil {
		return fmt.Errorf("failed to insert node: %w", err)
	}
	
	h.graph.nodes[vector.ID] = newNode
	h.graph.size++
	
	// Update entry point if new node has higher level
	if newNode.Level > h.graph.entryPoint.Level {
		h.graph.entryPoint = newNode
	}
	
	return nil
}// Search performs k-nearest neighbor search
func (h *HNSWIndex) Search(query []float32, k int, filter map[string]string) ([]core.SearchResult, error) {
	if len(query) != h.graph.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", 
			len(query), h.graph.dimension)
	}
	
	h.graph.mu.RLock()
	defer h.graph.mu.RUnlock()
	
	if h.graph.size == 0 {
		return []core.SearchResult{}, nil
	}
	
	// Set ef parameter for search
	ef := h.graph.config.EfSearch
	if ef < k {
		ef = k
	}
	
	// Phase 1: Navigate from entry point to layer 1
	entryPoints := []*HNSWNode{h.graph.entryPoint}
	
	// Search from top level down to level 1
	for level := h.graph.entryPoint.Level; level > 0; level-- {
		entryPoints = h.searchLayer(query, entryPoints, 1, level)
	}
	
	// Phase 2: Search layer 0 with ef candidates
	candidates := h.searchLayer(query, entryPoints, ef, 0)
	
	// Apply metadata filter and convert to results
	var results []core.SearchResult
	for _, node := range candidates {
		if !matchesFilter(node.Vector.Metadata, filter) {
			continue
		}
		
		distance, err := core.CalculateDistance(query, node.Vector.Values, h.graph.distanceMetric)
		if err != nil {
			continue
		}
		
		result := core.SearchResult{
			ID:       node.ID,
			Score:    distance,
			Metadata: node.Vector.Metadata,
		}
		results = append(results, result)
	}
	
	// Sort by distance and return top k
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})
	
	if k > len(results) {
		k = len(results)
	}
	
	return results[:k], nil
}// RangeSearch finds all vectors within a distance threshold
func (h *HNSWIndex) RangeSearch(query []float32, radius float32, filter map[string]string, limit int) ([]core.SearchResult, error) {
	if len(query) != h.graph.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", 
			len(query), h.graph.dimension)
	}
	
	h.graph.mu.RLock()
	defer h.graph.mu.RUnlock()
	
	if h.graph.size == 0 {
		return []core.SearchResult{}, nil
	}
	
	// Use a larger ef for range search to ensure we explore enough of the graph
	ef := h.graph.config.EfSearch * 2
	if ef < 100 {
		ef = 100
	}
	
	// Navigate from entry point to layer 0
	entryPoints := []*HNSWNode{h.graph.entryPoint}
	
	// Search from top level down to level 1
	for level := h.graph.entryPoint.Level; level > 0; level-- {
		entryPoints = h.searchLayer(query, entryPoints, 1, level)
	}
	
	// Search layer 0 with expanded ef
	candidates := h.searchLayer(query, entryPoints, ef, 0)
	
	// Collect all results within radius
	var results []core.SearchResult
	visited := make(map[string]bool)
	
	// Process initial candidates
	for _, node := range candidates {
		if visited[node.ID] {
			continue
		}
		visited[node.ID] = true
		
		distance, err := core.CalculateDistance(query, node.Vector.Values, h.graph.distanceMetric)
		if err != nil {
			continue
		}
		
		if distance <= radius && matchesFilter(node.Vector.Metadata, filter) {
			results = append(results, core.SearchResult{
				ID:       node.ID,
				Score:    distance,
				Vector:   &node.Vector,
				Metadata: node.Vector.Metadata,
			})
			
			if limit > 0 && len(results) >= limit {
				break
			}
		}
	}
	
	// Sort results by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})
	
	return results, nil
}

// Delete removes a vector from the HNSW index
func (h *HNSWIndex) Delete(id string) error {
	h.graph.mu.Lock()
	defer h.graph.mu.Unlock()
	
	node, exists := h.graph.nodes[id]
	if !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}
	
	// Remove all connections to this node
	for level, connections := range node.Connections {
		for connID := range connections {
			if connNode, exists := h.graph.nodes[connID]; exists {
				connNode.RemoveConnection(id, level)
			}
		}
	}
	
	// Remove node from graph
	delete(h.graph.nodes, id)
	h.graph.size--
	
	// Update entry point if this was the entry point
	if h.graph.entryPoint != nil && h.graph.entryPoint.ID == id {
		h.findNewEntryPoint()
	}
	
	return nil
}

// findNewEntryPoint finds a new entry point after the current one is deleted
func (h *HNSWIndex) findNewEntryPoint() {
	h.graph.entryPoint = nil
	maxLevel := -1
	
	for _, node := range h.graph.nodes {
		if node.Level > maxLevel {
			maxLevel = node.Level
			h.graph.entryPoint = node
		}
	}
}

// Rebuild rebuilds the entire index (placeholder - could implement optimized version)
func (h *HNSWIndex) Rebuild() error {
	// For now, just return nil - a full rebuild would require 
	// collecting all vectors and rebuilding the graph from scratch
	return nil
}

// Size returns the number of vectors in the index
func (h *HNSWIndex) Size() int {
	return h.graph.Size()
}

// Type returns the index type
func (h *HNSWIndex) Type() string {
	return "hnsw"
}

// hnswNodeState represents the serializable state of an HNSW node
type hnswNodeState struct {
	ID          string                    `json:"id"`
	Vector      core.Vector               `json:"vector"`
	Level       int                       `json:"level"`
	Connections map[int]map[string]bool   `json:"connections"`
}

// hnswIndexState represents the serializable state of an HNSW index
type hnswIndexState struct {
	Nodes          map[string]hnswNodeState `json:"nodes"`
	EntryPointID   string                   `json:"entry_point_id"`
	Config         HNSWConfig               `json:"config"`
	Dimension      int                      `json:"dimension"`
	DistanceMetric core.DistanceMetric      `json:"distance_metric"`
	Size           int                      `json:"size"`
}

// Serialize converts the HNSW index state to bytes
func (h *HNSWIndex) Serialize() ([]byte, error) {
	h.graph.mu.RLock()
	defer h.graph.mu.RUnlock()
	
	// Convert nodes to serializable format
	nodes := make(map[string]hnswNodeState)
	for id, node := range h.graph.nodes {
		nodes[id] = hnswNodeState{
			ID:          node.ID,
			Vector:      node.Vector,
			Level:       node.Level,
			Connections: node.Connections,
		}
	}
	
	// Get entry point ID
	entryPointID := ""
	if h.graph.entryPoint != nil {
		entryPointID = h.graph.entryPoint.ID
	}
	
	state := hnswIndexState{
		Nodes:          nodes,
		EntryPointID:   entryPointID,
		Config:         h.graph.config,
		Dimension:      h.graph.dimension,
		DistanceMetric: h.graph.distanceMetric,
		Size:           h.graph.size,
	}
	
	return json.Marshal(state)
}

// Deserialize restores the HNSW index state from bytes
func (h *HNSWIndex) Deserialize(data []byte) error {
	var state hnswIndexState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to unmarshal HNSW index state: %w", err)
	}
	
	h.graph.mu.Lock()
	defer h.graph.mu.Unlock()
	
	// Rebuild graph structure
	h.graph.nodes = make(map[string]*HNSWNode)
	h.graph.config = state.Config
	h.graph.dimension = state.Dimension
	h.graph.distanceMetric = state.DistanceMetric
	h.graph.size = state.Size
	
	// Recreate nodes
	for id, nodeState := range state.Nodes {
		node := &HNSWNode{
			ID:          nodeState.ID,
			Vector:      nodeState.Vector,
			Level:       nodeState.Level,
			Connections: nodeState.Connections,
		}
		h.graph.nodes[id] = node
	}
	
	// Restore entry point
	if state.EntryPointID != "" {
		if node, exists := h.graph.nodes[state.EntryPointID]; exists {
			h.graph.entryPoint = node
		}
	}
	
	return nil
}