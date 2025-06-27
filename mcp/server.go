package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"sync"

	"github.com/dshills/EmbeddixDB/core"
)

// Server represents an MCP server
type Server struct {
	store    core.VectorStore
	handlers map[string]Handler
	mu       sync.RWMutex
	logger   *log.Logger
}

// NewServer creates a new MCP server
func NewServer(store core.VectorStore) *Server {
	logger := log.New(os.Stderr, "[MCP] ", log.LstdFlags|log.Lshortfile)
	
	s := &Server{
		store:    store,
		handlers: make(map[string]Handler),
		logger:   logger,
	}
	
	// Register tool handlers
	s.registerHandlers()
	
	return s
}

// registerHandlers registers all tool handlers
func (s *Server) registerHandlers() {
	s.handlers["search_vectors"] = &SearchVectorsHandler{store: s.store}
	s.handlers["add_vectors"] = &AddVectorsHandler{store: s.store}
	s.handlers["get_vector"] = &GetVectorHandler{store: s.store}
	s.handlers["delete_vector"] = &DeleteVectorHandler{store: s.store}
	s.handlers["create_collection"] = &CreateCollectionHandler{store: s.store}
	s.handlers["list_collections"] = &ListCollectionsHandler{store: s.store}
	s.handlers["delete_collection"] = &DeleteCollectionHandler{store: s.store}
}

// Serve starts the MCP server on stdio
func (s *Server) Serve(ctx context.Context) error {
	reader := bufio.NewReader(os.Stdin)
	encoder := json.NewEncoder(os.Stdout)
	
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Read until we get a complete JSON object
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					return nil
				}
				return fmt.Errorf("read error: %w", err)
			}
			
			// Parse the request
			var req Request
			if err := json.Unmarshal(line, &req); err != nil {
				// Send parse error response
				resp := Response{
					JSONRPC: JSONRPCVersion,
					Error: &Error{
						Code:    ErrorCodeParse,
						Message: "Parse error",
						Data:    err.Error(),
					},
					ID: nil,
				}
				if err := encoder.Encode(resp); err != nil {
					s.logger.Printf("Failed to send parse error response: %v", err)
				}
				continue
			}
			
			// Validate JSON-RPC version
			if req.JSONRPC != JSONRPCVersion {
				resp := Response{
					JSONRPC: JSONRPCVersion,
					Error: &Error{
						Code:    ErrorCodeInvalidRequest,
						Message: "Invalid request",
						Data:    "Unsupported JSON-RPC version",
					},
					ID: req.ID,
				}
				if err := encoder.Encode(resp); err != nil {
					s.logger.Printf("Failed to send error response: %v", err)
				}
				continue
			}
			
			// Handle the request
			resp := s.handleRequest(ctx, &req)
			
			// Send the response
			if err := encoder.Encode(resp); err != nil {
				s.logger.Printf("Failed to send response: %v", err)
			}
		}
	}
}

// handleRequest handles a single JSON-RPC request
func (s *Server) handleRequest(ctx context.Context, req *Request) Response {
	resp := Response{
		JSONRPC: JSONRPCVersion,
		ID:      req.ID,
	}
	
	switch req.Method {
	case "initialize":
		result, err := s.handleInitialize(ctx, req.Params)
		if err != nil {
			resp.Error = &Error{
				Code:    ErrorCodeInvalidParams,
				Message: err.Error(),
			}
		} else {
			resp.Result = result
		}
		
	case "tools/list":
		resp.Result = ToolsListResponse{
			Tools: GetTools(),
		}
		
	case "tools/call":
		result, err := s.handleToolCall(ctx, req.Params)
		if err != nil {
			resp.Error = &Error{
				Code:    ErrorCodeInternal,
				Message: err.Error(),
			}
		} else {
			resp.Result = result
		}
		
	default:
		resp.Error = &Error{
			Code:    ErrorCodeMethodNotFound,
			Message: "Method not found",
			Data:    req.Method,
		}
	}
	
	return resp
}

// handleInitialize handles the initialize request
func (s *Server) handleInitialize(ctx context.Context, params json.RawMessage) (*InitializeResponse, error) {
	var req InitializeRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("invalid initialize params: %w", err)
	}
	
	return &InitializeResponse{
		ProtocolVersion: ProtocolVersion,
		Capabilities: ServerCapabilities{
			Tools: &ToolsCapability{
				ListChanged: false,
			},
		},
		ServerInfo: ServerInfo{
			Name:    "EmbeddixDB MCP Server",
			Version: "0.1.0",
		},
	}, nil
}

// handleToolCall handles a tool call request
func (s *Server) handleToolCall(ctx context.Context, params json.RawMessage) (*ToolCallResponse, error) {
	var req ToolCallRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("invalid tool call params: %w", err)
	}
	
	// Get the handler for this tool
	s.mu.RLock()
	handler, exists := s.handlers[req.Name]
	s.mu.RUnlock()
	
	if !exists {
		return &ToolCallResponse{
			Content: []ToolContent{
				{
					Type: "text",
					Text: fmt.Sprintf("Unknown tool: %s", req.Name),
				},
			},
			IsError: true,
		}, nil
	}
	
	// Execute the tool
	result, err := handler.Execute(ctx, req.Arguments)
	if err != nil {
		return &ToolCallResponse{
			Content: []ToolContent{
				{
					Type: "text",
					Text: fmt.Sprintf("Tool execution error: %v", err),
				},
			},
			IsError: true,
		}, nil
	}
	
	return result, nil
}

// Close gracefully shuts down the server
func (s *Server) Close() error {
	s.logger.Println("MCP server shutting down")
	return nil
}