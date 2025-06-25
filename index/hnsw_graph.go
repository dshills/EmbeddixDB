package index

import (
	"math/rand"
	"sync"
	
	"github.com/dshills/EmbeddixDB/core"
)

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	ID       string
	Vector   core.Vector
	Level    int
	// Connections at each level: level -> set of connected node IDs
	Connections map[int]map[string]bool
}

// NewHNSWNode creates a new HNSW node
func NewHNSWNode(vector core.Vector, level int) *HNSWNode {
	return &HNSWNode{
		ID:          vector.ID,
		Vector:      vector,
		Level:       level,
		Connections: make(map[int]map[string]bool),
	}
}

// AddConnection adds a bidirectional connection between two nodes at a specific level
func (n *HNSWNode) AddConnection(nodeID string, level int) {
	if n.Connections[level] == nil {
		n.Connections[level] = make(map[string]bool)
	}
	n.Connections[level][nodeID] = true
}

// RemoveConnection removes a connection at a specific level
func (n *HNSWNode) RemoveConnection(nodeID string, level int) {
	if connections, exists := n.Connections[level]; exists {
		delete(connections, nodeID)
		if len(connections) == 0 {
			delete(n.Connections, level)
		}
	}
}

// GetConnections returns all connected node IDs at a specific level
func (n *HNSWNode) GetConnections(level int) []string {
	connections, exists := n.Connections[level]
	if !exists {
		return nil
	}
	
	result := make([]string, 0, len(connections))
	for nodeID := range connections {
		result = append(result, nodeID)
	}
	return result
}// HNSWGraph represents the HNSW graph structure
type HNSWGraph struct {
	mu           sync.RWMutex
	nodes        map[string]*HNSWNode
	entryPoint   *HNSWNode
	config       HNSWConfig
	rng          *rand.Rand
	dimension    int
	distanceMetric core.DistanceMetric
	size         int
}

// NewHNSWGraph creates a new HNSW graph
func NewHNSWGraph(dimension int, distanceMetric core.DistanceMetric, config HNSWConfig) *HNSWGraph {
	return &HNSWGraph{
		nodes:          make(map[string]*HNSWNode),
		config:         config,
		rng:            rand.New(rand.NewSource(config.Seed)),
		dimension:      dimension,
		distanceMetric: distanceMetric,
		size:           0,
	}
}

// Size returns the number of nodes in the graph
func (g *HNSWGraph) Size() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.size
}

// GetNode returns a node by ID
func (g *HNSWGraph) GetNode(id string) (*HNSWNode, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	node, exists := g.nodes[id]
	return node, exists
}

// assignLevel assigns a level to a new node based on the ML parameter
func (g *HNSWGraph) assignLevel() int {
	level := int(-g.config.ML * g.rng.ExpFloat64())
	if level > g.config.MaxLevels-1 {
		level = g.config.MaxLevels - 1
	}
	return level
}

// getMaxConnections returns the maximum number of connections for a given level
func (g *HNSWGraph) getMaxConnections(level int) int {
	if level == 0 {
		return g.config.M * 2 // Level 0 can have more connections
	}
	return g.config.MMax
}