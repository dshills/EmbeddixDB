package index

import (
	"fmt"

	"github.com/dshills/EmbeddixDB/core"
)

// DefaultFactory implements core.IndexFactory
type DefaultFactory struct{}

// NewDefaultFactory creates a new default index factory
func NewDefaultFactory() *DefaultFactory {
	return &DefaultFactory{}
}

// CreateIndex creates an index instance based on type and configuration
func (f *DefaultFactory) CreateIndex(indexType string, dimension int, distanceMetric core.DistanceMetric) (core.Index, error) {
	switch indexType {
	case "flat":
		return NewFlatIndex(dimension, distanceMetric), nil
	case "hnsw":
		config := DefaultHNSWConfig()
		return NewHNSWIndex(dimension, distanceMetric, config), nil
	default:
		return nil, fmt.Errorf("unsupported index type: %s", indexType)
	}
}
