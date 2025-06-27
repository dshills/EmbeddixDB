package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/dshills/EmbeddixDB/core"
)

// SearchVectorsHandler handles the search_vectors tool
type SearchVectorsHandler struct {
	store core.VectorStore
}

// Execute performs vector search
func (h *SearchVectorsHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// Parse arguments
	jsonData, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal args: %w", err)
	}
	
	var searchArgs SearchVectorsArgs
	if err := json.Unmarshal(jsonData, &searchArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	
	// Validate required fields
	if searchArgs.Collection == "" {
		return nil, fmt.Errorf("collection is required")
	}
	
	if searchArgs.Query == "" && len(searchArgs.Vector) == 0 {
		return nil, fmt.Errorf("either query or vector must be provided")
	}
	
	// Set defaults
	if searchArgs.Limit == 0 {
		searchArgs.Limit = 10
	}
	
	var queryVector []float32
	
	// Handle text query
	if searchArgs.Query != "" {
		// Check if the store supports auto-embedding
		if embeddingStore, ok := h.store.(interface {
			EmbedText(ctx context.Context, text string) ([]float32, error)
		}); ok {
			vec, err := embeddingStore.EmbedText(ctx, searchArgs.Query)
			if err != nil {
				return nil, fmt.Errorf("failed to embed query: %w", err)
			}
			queryVector = vec
		} else {
			return nil, fmt.Errorf("text queries require embedding support")
		}
	} else {
		queryVector = searchArgs.Vector
	}
	
	// Convert filters from map[string]interface{} to map[string]string
	filters := make(map[string]string)
	if searchArgs.Filters != nil {
		for k, v := range searchArgs.Filters {
			if strVal, ok := v.(string); ok {
				filters[k] = strVal
			} else {
				filters[k] = fmt.Sprintf("%v", v)
			}
		}
	}
	
	// Create search request
	searchReq := core.SearchRequest{
		Query:          queryVector,
		TopK:           searchArgs.Limit,
		Filter:         filters,
		IncludeVectors: false,
		UserID:         searchArgs.UserID,
	}
	
	// Perform the search
	results, err := h.store.Search(ctx, searchArgs.Collection, searchReq)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	
	// Format results
	searchResults := make([]SearchResult, len(results))
	for i, result := range results {
		sr := SearchResult{
			ID:    result.ID,
			Score: result.Score,
		}
		
		if searchArgs.IncludeMetadata && result.Metadata != nil {
			// Convert map[string]string to map[string]interface{}
			metadata := make(map[string]interface{})
			for k, v := range result.Metadata {
				metadata[k] = v
			}
			sr.Metadata = metadata
			
			// Extract content if available
			if content, ok := result.Metadata["content"]; ok {
				sr.Content = content
			}
		}
		
		searchResults[i] = sr
	}
	
	// Create response
	resultData, err := json.Marshal(searchResults)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal results: %w", err)
	}
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Found %d matching vectors", len(searchResults)),
			},
			{
				Type: "text",
				Text: string(resultData),
			},
		},
	}, nil
}

// AddVectorsHandler handles the add_vectors tool
type AddVectorsHandler struct {
	store core.VectorStore
}

// Execute adds vectors to a collection
func (h *AddVectorsHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// Parse arguments
	jsonData, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal args: %w", err)
	}
	
	var addArgs AddVectorsArgs
	if err := json.Unmarshal(jsonData, &addArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	
	// Validate required fields
	if addArgs.Collection == "" {
		return nil, fmt.Errorf("collection is required")
	}
	
	if len(addArgs.Vectors) == 0 {
		return nil, fmt.Errorf("vectors array cannot be empty")
	}
	
	// Process vectors
	addedIDs := make([]string, 0, len(addArgs.Vectors))
	errors := make([]string, 0)
	
	for i, vecInput := range addArgs.Vectors {
		// Generate ID if not provided
		if vecInput.ID == "" {
			vecInput.ID = uuid.New().String()
		}
		
		var vector []float32
		
		// Handle content vs raw vector
		if vecInput.Content != "" {
			// Check if the store supports auto-embedding
			if embeddingStore, ok := h.store.(interface {
				EmbedText(ctx context.Context, text string) ([]float32, error)
			}); ok {
				vec, err := embeddingStore.EmbedText(ctx, vecInput.Content)
				if err != nil {
					errors = append(errors, fmt.Sprintf("vector %d: failed to embed content: %v", i, err))
					continue
				}
				vector = vec
			} else {
				errors = append(errors, fmt.Sprintf("vector %d: content provided but embedding not supported", i))
				continue
			}
		} else if len(vecInput.Vector) > 0 {
			vector = vecInput.Vector
		} else {
			errors = append(errors, fmt.Sprintf("vector %d: either content or vector must be provided", i))
			continue
		}
		
		// Prepare metadata - convert to map[string]string
		metadata := make(map[string]string)
		if vecInput.Metadata != nil {
			for k, v := range vecInput.Metadata {
				if strVal, ok := v.(string); ok {
					metadata[k] = strVal
				} else {
					metadata[k] = fmt.Sprintf("%v", v)
				}
			}
		}
		
		// Store content in metadata if provided
		if vecInput.Content != "" {
			metadata["content"] = vecInput.Content
		}
		
		// Add timestamp
		metadata["created_at"] = time.Now().UTC().Format(time.RFC3339)
		
		// Create vector object
		vec := core.Vector{
			ID:       vecInput.ID,
			Values:   vector,
			Metadata: metadata,
		}
		
		// Add to store
		if err := h.store.AddVector(ctx, addArgs.Collection, vec); err != nil {
			errors = append(errors, fmt.Sprintf("vector %d (ID: %s): %v", i, vecInput.ID, err))
			continue
		}
		
		addedIDs = append(addedIDs, vecInput.ID)
	}
	
	// Create result
	result := AddVectorsResult{
		Added:  len(addedIDs),
		IDs:    addedIDs,
		Errors: errors,
	}
	
	resultData, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal result: %w", err)
	}
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Successfully added %d vectors", len(addedIDs)),
			},
			{
				Type: "text",
				Text: string(resultData),
			},
		},
	}, nil
}

// GetVectorHandler handles the get_vector tool
type GetVectorHandler struct {
	store core.VectorStore
}

// Execute retrieves a specific vector
func (h *GetVectorHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// Parse arguments
	jsonData, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal args: %w", err)
	}
	
	var getArgs GetVectorArgs
	if err := json.Unmarshal(jsonData, &getArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}
	
	// Validate required fields
	if getArgs.Collection == "" {
		return nil, fmt.Errorf("collection is required")
	}
	
	if getArgs.ID == "" {
		return nil, fmt.Errorf("id is required")
	}
	
	// Get the vector
	vector, err := h.store.GetVector(ctx, getArgs.Collection, getArgs.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get vector: %w", err)
	}
	
	// Convert metadata from map[string]string to map[string]interface{}
	metadata := make(map[string]interface{})
	for k, v := range vector.Metadata {
		metadata[k] = v
	}
	
	// Format result
	result := VectorData{
		ID:       vector.ID,
		Vector:   vector.Values,
		Metadata: metadata,
	}
	
	// Extract content if available
	if content, ok := vector.Metadata["content"]; ok {
		result.Content = content
	}
	
	// Extract timestamps if available
	if createdAt, ok := vector.Metadata["created_at"]; ok {
		if t, err := time.Parse(time.RFC3339, createdAt); err == nil {
			result.CreatedAt = t
		}
	}
	
	if updatedAt, ok := vector.Metadata["updated_at"]; ok {
		if t, err := time.Parse(time.RFC3339, updatedAt); err == nil {
			result.UpdatedAt = t
		}
	}
	
	resultData, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal result: %w", err)
	}
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Retrieved vector %s", getArgs.ID),
			},
			{
				Type: "text",
				Text: string(resultData),
			},
		},
	}, nil
}

// DeleteVectorHandler handles the delete_vector tool
type DeleteVectorHandler struct {
	store core.VectorStore
}

// Execute deletes a vector
func (h *DeleteVectorHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// Parse arguments
	collection, ok := args["collection"].(string)
	if !ok || collection == "" {
		return nil, fmt.Errorf("collection is required")
	}
	
	id, ok := args["id"].(string)
	if !ok || id == "" {
		return nil, fmt.Errorf("id is required")
	}
	
	// Delete the vector
	if err := h.store.DeleteVector(ctx, collection, id); err != nil {
		return nil, fmt.Errorf("failed to delete vector: %w", err)
	}
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Successfully deleted vector %s from collection %s", id, collection),
			},
		},
	}, nil
}

// CreateCollectionHandler handles the create_collection tool
type CreateCollectionHandler struct {
	store core.VectorStore
}

// Execute creates a new collection
func (h *CreateCollectionHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// Parse arguments
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, fmt.Errorf("name is required")
	}
	
	dimensionFloat, ok := args["dimension"].(float64)
	if !ok {
		return nil, fmt.Errorf("dimension is required")
	}
	dimension := int(dimensionFloat)
	
	// Parse optional fields
	distance := "cosine"
	if d, ok := args["distance"].(string); ok {
		distance = d
	}
	
	indexType := "hnsw"
	if it, ok := args["indexType"].(string); ok {
		indexType = it
	}
	
	// Create collection
	collection := core.Collection{
		Name:      name,
		Dimension: dimension,
		Distance:  distance,
		IndexType: indexType,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	// Create the collection
	if err := h.store.CreateCollection(ctx, collection); err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	
	resultData, _ := json.Marshal(map[string]interface{}{
		"name":      name,
		"dimension": dimension,
		"distance":  distance,
		"indexType": indexType,
		"created":   true,
	})
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Successfully created collection '%s'", name),
			},
			{
				Type: "text",
				Text: string(resultData),
			},
		},
	}, nil
}

// ListCollectionsHandler handles the list_collections tool
type ListCollectionsHandler struct {
	store core.VectorStore
}

// Execute lists all collections
func (h *ListCollectionsHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// List collections
	collections, err := h.store.ListCollections(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list collections: %w", err)
	}
	
	resultData, err := json.Marshal(collections)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal collections: %w", err)
	}
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Found %d collections", len(collections)),
			},
			{
				Type: "text",
				Text: string(resultData),
			},
		},
	}, nil
}

// DeleteCollectionHandler handles the delete_collection tool
type DeleteCollectionHandler struct {
	store core.VectorStore
}

// Execute deletes a collection
func (h *DeleteCollectionHandler) Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error) {
	// Parse arguments
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, fmt.Errorf("name is required")
	}
	
	// Delete the collection
	if err := h.store.DeleteCollection(ctx, name); err != nil {
		return nil, fmt.Errorf("failed to delete collection: %w", err)
	}
	
	return &ToolCallResponse{
		Content: []ToolContent{
			{
				Type: "text",
				Text: fmt.Sprintf("Successfully deleted collection '%s'", name),
			},
		},
	}, nil
}