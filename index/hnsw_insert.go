package index

import (
	"sort"
	
	"github.com/dshills/EmbeddixDB/core"
)

// insertNode inserts a new node into the HNSW graph
func (h *HNSWIndex) insertNode(newNode *HNSWNode) error {
	entryPoints := []*HNSWNode{h.graph.entryPoint}
	
	// Phase 1: Search from top level down to newNode.Level + 1
	for level := h.graph.entryPoint.Level; level > newNode.Level; level-- {
		entryPoints = h.searchLayer(newNode.Vector.Values, entryPoints, 1, level)
	}
	
	// Phase 2: Search and connect from newNode.Level down to 0
	for level := min(newNode.Level, h.graph.entryPoint.Level); level >= 0; level-- {
		candidates := h.searchLayer(newNode.Vector.Values, entryPoints, h.graph.config.EfConstruction, level)
		
		// Select M closest candidates
		maxConn := h.graph.getMaxConnections(level)
		if level == 0 {
			maxConn = h.graph.config.M
		}
		
		selected := h.selectNeighbors(newNode, candidates, maxConn, level)
		
		// Add bidirectional connections
		for _, neighbor := range selected {
			h.addConnection(newNode, neighbor, level)
			h.addConnection(neighbor, newNode, level)
			
			// Prune connections of neighbor if needed
			h.pruneConnections(neighbor, level)
		}
		
		entryPoints = candidates
	}
	
	return nil
}

// selectNeighbors selects the best neighbors using a simple heuristic
func (h *HNSWIndex) selectNeighbors(node *HNSWNode, candidates []*HNSWNode, maxCount int, level int) []*HNSWNode {
	if len(candidates) <= maxCount {
		return candidates
	}
	
	// Calculate distances to all candidates
	distances := make([]DistanceNode, len(candidates))
	for i, candidate := range candidates {
		distance, err := core.CalculateDistance(node.Vector.Values, candidate.Vector.Values, h.graph.distanceMetric)
		if err != nil {
			distance = float32(1e9) // Large distance for errors
		}
		distances[i] = DistanceNode{Node: candidate, Distance: distance}
	}
	
	// Sort by distance and take closest
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Distance < distances[j].Distance
	})
	
	result := make([]*HNSWNode, maxCount)
	for i := 0; i < maxCount; i++ {
		result[i] = distances[i].Node
	}
	
	return result
}// addConnection adds a connection between two nodes at a specific level
func (h *HNSWIndex) addConnection(from, to *HNSWNode, level int) {
	from.AddConnection(to.ID, level)
}

// pruneConnections removes excess connections if a node has too many
func (h *HNSWIndex) pruneConnections(node *HNSWNode, level int) {
	maxConn := h.graph.getMaxConnections(level)
	connections := node.GetConnections(level)
	
	if len(connections) <= maxConn {
		return
	}
	
	// Calculate distances to all connected nodes
	distances := make([]DistanceNode, len(connections))
	for i, connID := range connections {
		connNode, exists := h.graph.nodes[connID]
		if !exists {
			continue
		}
		
		distance, err := core.CalculateDistance(node.Vector.Values, connNode.Vector.Values, h.graph.distanceMetric)
		if err != nil {
			distance = float32(1e9)
		}
		distances[i] = DistanceNode{Node: connNode, Distance: distance}
	}
	
	// Sort by distance and keep only the closest maxConn
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Distance < distances[j].Distance
	})
	
	// Remove excess connections
	for i := maxConn; i < len(distances); i++ {
		if distances[i].Node != nil {
			node.RemoveConnection(distances[i].Node.ID, level)
			distances[i].Node.RemoveConnection(node.ID, level)
		}
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}