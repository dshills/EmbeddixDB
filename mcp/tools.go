package mcp

// GetTools returns all available MCP tools
func GetTools() []Tool {
	return []Tool{
		{
			Name:        "search_vectors",
			Description: "Search for similar vectors using semantic similarity",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"collection": {
						Type:        "string",
						Description: "The collection to search in",
					},
					"query": {
						Type:        "string",
						Description: "Text query for semantic search",
					},
					"vector": {
						Type:        "array",
						Description: "Raw vector for similarity search",
						Items: &Property{
							Type: "number",
						},
					},
					"limit": {
						Type:        "integer",
						Description: "Maximum number of results to return",
						Default:     10,
					},
					"filters": {
						Type:        "object",
						Description: "Metadata filters to apply",
					},
					"includeMetadata": {
						Type:        "boolean",
						Description: "Whether to include metadata in results",
						Default:     true,
					},
					"sessionId": {
						Type:        "string",
						Description: "Session ID for personalized search",
					},
					"userId": {
						Type:        "string",
						Description: "User ID for personalized search",
					},
				},
				Required: []string{"collection"},
				OneOf: []map[string][]string{
					{"required": []string{"query"}},
					{"required": []string{"vector"}},
				},
			},
		},
		{
			Name:        "add_vectors",
			Description: "Add vectors to a collection",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"collection": {
						Type:        "string",
						Description: "The collection to add vectors to",
					},
					"vectors": {
						Type:        "array",
						Description: "Array of vectors to add",
						Items: &Property{
							Type: "object",
							Properties: map[string]Property{
								"id": {
									Type:        "string",
									Description: "Optional vector ID",
								},
								"content": {
									Type:        "string",
									Description: "Text content to embed",
								},
								"vector": {
									Type:        "array",
									Description: "Raw vector data",
									Items: &Property{
										Type: "number",
									},
								},
								"metadata": {
									Type:        "object",
									Description: "Additional metadata",
								},
							},
						},
					},
				},
				Required: []string{"collection", "vectors"},
			},
		},
		{
			Name:        "get_vector",
			Description: "Retrieve a specific vector by ID",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"collection": {
						Type:        "string",
						Description: "The collection containing the vector",
					},
					"id": {
						Type:        "string",
						Description: "The vector ID to retrieve",
					},
				},
				Required: []string{"collection", "id"},
			},
		},
		{
			Name:        "delete_vector",
			Description: "Delete a vector from a collection",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"collection": {
						Type:        "string",
						Description: "The collection containing the vector",
					},
					"id": {
						Type:        "string",
						Description: "The vector ID to delete",
					},
				},
				Required: []string{"collection", "id"},
			},
		},
		{
			Name:        "create_collection",
			Description: "Create a new vector collection",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"name": {
						Type:        "string",
						Description: "Name of the collection",
					},
					"dimension": {
						Type:        "integer",
						Description: "Vector dimension",
					},
					"distance": {
						Type:        "string",
						Description: "Distance metric to use",
						Enum:        []string{"cosine", "l2", "dot"},
						Default:     "cosine",
					},
					"indexType": {
						Type:        "string",
						Description: "Index type to use",
						Enum:        []string{"flat", "hnsw"},
						Default:     "hnsw",
					},
				},
				Required: []string{"name", "dimension"},
			},
		},
		{
			Name:        "list_collections",
			Description: "List all available collections",
			InputSchema: InputSchema{
				Type:       "object",
				Properties: map[string]Property{},
			},
		},
		{
			Name:        "delete_collection",
			Description: "Delete a collection",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"name": {
						Type:        "string",
						Description: "Name of the collection to delete",
					},
				},
				Required: []string{"name"},
			},
		},
	}
}

// GetToolByName returns a tool by its name
func GetToolByName(name string) (*Tool, bool) {
	tools := GetTools()
	for _, tool := range tools {
		if tool.Name == name {
			return &tool, true
		}
	}
	return nil, false
}