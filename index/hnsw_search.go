package index

import (
	"container/heap"
	
	"github.com/dshills/EmbeddixDB/core"
)

// DistanceNode represents a node with its distance for priority queue operations
type DistanceNode struct {
	Node     *HNSWNode
	Distance float32
}

// MinHeap for closest candidates (min-heap: smallest distances first)
type MinHeap []*DistanceNode

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(*DistanceNode))
}

func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// MaxHeap for dynamic candidate list (max-heap: largest distances first)
type MaxHeap []*DistanceNode

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(*DistanceNode))
}

func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}// searchLayer performs search in a specific layer of the HNSW graph
func (h *HNSWIndex) searchLayer(query []float32, entryPoints []*HNSWNode, ef int, level int) []*HNSWNode {
	visited := make(map[string]bool)
	candidates := &MaxHeap{}
	dynamic := &MaxHeap{}
	
	// Initialize with entry points
	for _, ep := range entryPoints {
		if visited[ep.ID] {
			continue
		}
		
		distance, err := core.CalculateDistance(query, ep.Vector.Values, h.graph.distanceMetric)
		if err != nil {
			continue
		}
		
		distNode := &DistanceNode{Node: ep, Distance: distance}
		heap.Push(candidates, distNode)
		heap.Push(dynamic, distNode)
		visited[ep.ID] = true
	}
	
	for candidates.Len() > 0 {
		// Get closest unvisited candidate
		current := heap.Pop(candidates).(*DistanceNode)
		
		// If current is farther than farthest in dynamic list, stop
		if dynamic.Len() >= ef && current.Distance > (*dynamic)[0].Distance {
			break
		}
		
		// Check all neighbors of current node at this level
		neighbors := current.Node.GetConnections(level)
		for _, neighborID := range neighbors {
			if visited[neighborID] {
				continue
			}
			
			neighbor, exists := h.graph.nodes[neighborID]
			if !exists {
				continue
			}
			
			distance, err := core.CalculateDistance(query, neighbor.Vector.Values, h.graph.distanceMetric)
			if err != nil {
				continue
			}
			
			visited[neighborID] = true
			
			// Add to dynamic list if better than worst or if we have space
			if dynamic.Len() < ef {
				distNode := &DistanceNode{Node: neighbor, Distance: distance}
				heap.Push(candidates, distNode)
				heap.Push(dynamic, distNode)
			} else if distance < (*dynamic)[0].Distance {
				// Replace worst candidate
				heap.Pop(dynamic)
				distNode := &DistanceNode{Node: neighbor, Distance: distance}
				heap.Push(candidates, distNode)
				heap.Push(dynamic, distNode)
			}
		}
	}
	
	// Extract results from dynamic list
	results := make([]*HNSWNode, dynamic.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(dynamic).(*DistanceNode).Node
	}
	
	return results
}