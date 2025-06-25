package index

import "github.com/dshills/EmbeddixDB/core"



// IndexConfig holds configuration for index creation
type IndexConfig struct {
	Type           string
	Dimension      int
	DistanceMetric core.DistanceMetric
}