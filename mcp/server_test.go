package mcp

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/dshills/EmbeddixDB/core"
	"github.com/dshills/EmbeddixDB/index"
	"github.com/dshills/EmbeddixDB/persistence"
)

func TestServerInitialize(t *testing.T) {
	// Create a test server
	persistenceConfig := persistence.DefaultPersistenceConfig(persistence.PersistenceMemory, "")
	persistenceFactory := persistence.NewDefaultFactory()
	backend, err := persistenceFactory.CreatePersistence(persistenceConfig)
	if err != nil {
		t.Fatalf("Failed to create persistence backend: %v", err)
	}
	defer backend.Close()
	
	indexFactory := index.NewDefaultFactory()
	store := core.NewVectorStore(backend, indexFactory)
	
	server := NewServer(store)
	
	// Test initialize request
	initReq := InitializeRequest{
		ProtocolVersion: ProtocolVersion,
		Capabilities:    ClientCapabilities{},
		ClientInfo: ClientInfo{
			Name:    "test-client",
			Version: "1.0.0",
		},
	}
	
	params, _ := json.Marshal(initReq)
	
	resp, err := server.handleInitialize(context.Background(), params)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	if resp.ProtocolVersion != ProtocolVersion {
		t.Errorf("Expected protocol version %s, got %s", ProtocolVersion, resp.ProtocolVersion)
	}
	
	if resp.ServerInfo.Name != "EmbeddixDB MCP Server" {
		t.Errorf("Unexpected server name: %s", resp.ServerInfo.Name)
	}
}

func TestToolsList(t *testing.T) {
	tools := GetTools()
	
	// Check that we have the expected tools
	expectedTools := []string{
		"search_vectors",
		"add_vectors",
		"get_vector",
		"delete_vector",
		"create_collection",
		"list_collections",
		"delete_collection",
	}
	
	if len(tools) != len(expectedTools) {
		t.Errorf("Expected %d tools, got %d", len(expectedTools), len(tools))
	}
	
	// Check each tool exists
	for _, expectedName := range expectedTools {
		found := false
		for _, tool := range tools {
			if tool.Name == expectedName {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected tool %s not found", expectedName)
		}
	}
}

func TestHandleRequest(t *testing.T) {
	// Create a test server
	persistenceConfig := persistence.DefaultPersistenceConfig(persistence.PersistenceMemory, "")
	persistenceFactory := persistence.NewDefaultFactory()
	backend, err := persistenceFactory.CreatePersistence(persistenceConfig)
	if err != nil {
		t.Fatalf("Failed to create persistence backend: %v", err)
	}
	defer backend.Close()
	
	indexFactory := index.NewDefaultFactory()
	store := core.NewVectorStore(backend, indexFactory)
	
	server := NewServer(store)
	
	// Test tools/list request
	req := &Request{
		JSONRPC: JSONRPCVersion,
		Method:  "tools/list",
		ID:      1,
	}
	
	resp := server.handleRequest(context.Background(), req)
	
	if resp.Error != nil {
		t.Fatalf("Request failed: %v", resp.Error)
	}
	
	if resp.Result == nil {
		t.Fatal("Expected result, got nil")
	}
	
	// Check the result type
	result, ok := resp.Result.(ToolsListResponse)
	if !ok {
		t.Fatal("Expected ToolsListResponse")
	}
	
	if len(result.Tools) == 0 {
		t.Error("Expected tools in response")
	}
}

func TestCreateAndSearchVectors(t *testing.T) {
	// Create a test server
	persistenceConfig := persistence.DefaultPersistenceConfig(persistence.PersistenceMemory, "")
	persistenceFactory := persistence.NewDefaultFactory()
	backend, err := persistenceFactory.CreatePersistence(persistenceConfig)
	if err != nil {
		t.Fatalf("Failed to create persistence backend: %v", err)
	}
	defer backend.Close()
	
	indexFactory := index.NewDefaultFactory()
	store := core.NewVectorStore(backend, indexFactory)
	
	ctx := context.Background()
	
	// Create a collection
	createHandler := &CreateCollectionHandler{store: store}
	createArgs := map[string]interface{}{
		"name":      "test-collection",
		"dimension": float64(3),
		"distance":  "cosine",
	}
	
	resp, err := createHandler.Execute(ctx, createArgs)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	
	if resp.IsError {
		t.Fatal("Create collection returned error")
	}
	
	// Add vectors
	addHandler := &AddVectorsHandler{store: store}
	addArgs := map[string]interface{}{
		"collection": "test-collection",
		"vectors": []map[string]interface{}{
			{
				"id":     "vec1",
				"vector": []float32{1.0, 0.0, 0.0},
				"metadata": map[string]interface{}{
					"content": "First vector",
				},
			},
			{
				"id":     "vec2",
				"vector": []float32{0.0, 1.0, 0.0},
				"metadata": map[string]interface{}{
					"content": "Second vector",
				},
			},
		},
	}
	
	resp, err = addHandler.Execute(ctx, addArgs)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	
	if resp.IsError {
		t.Fatal("Add vectors returned error")
	}
	
	// Search vectors
	searchHandler := &SearchVectorsHandler{store: store}
	searchArgs := map[string]interface{}{
		"collection": "test-collection",
		"vector":     []float32{0.9, 0.1, 0.0},
		"limit":      2,
		"includeMetadata": true,
	}
	
	resp, err = searchHandler.Execute(ctx, searchArgs)
	if err != nil {
		t.Fatalf("Failed to search vectors: %v", err)
	}
	
	if resp.IsError {
		t.Fatal("Search vectors returned error")
	}
	
	// Verify we got results
	if len(resp.Content) < 2 {
		t.Fatal("Expected at least 2 content items in search response")
	}
}