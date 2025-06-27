package mcp

import (
	"context"
	"encoding/json"
	"time"
)

// Protocol constants
const (
	ProtocolVersion = "2024-11-05"
	JSONRPCVersion  = "2.0"
)

// Request represents an MCP JSON-RPC request
type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	ID      interface{}     `json:"id"`
}

// Response represents an MCP JSON-RPC response
type Response struct {
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result,omitempty"`
	Error   *Error      `json:"error,omitempty"`
	ID      interface{} `json:"id"`
}

// Error represents an MCP error
type Error struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Standard error codes
const (
	ErrorCodeParse          = -32700
	ErrorCodeInvalidRequest = -32600
	ErrorCodeMethodNotFound = -32601
	ErrorCodeInvalidParams  = -32602
	ErrorCodeInternal       = -32603
)

// Tool represents an MCP tool definition
type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema InputSchema `json:"inputSchema"`
}

// InputSchema defines the schema for tool inputs
type InputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]Property    `json:"properties"`
	Required   []string               `json:"required,omitempty"`
	OneOf      []map[string][]string  `json:"oneOf,omitempty"`
}

// Property defines a property in the input schema
type Property struct {
	Type        string      `json:"type"`
	Description string      `json:"description,omitempty"`
	Default     interface{} `json:"default,omitempty"`
	Enum        []string    `json:"enum,omitempty"`
	Items       *Property   `json:"items,omitempty"`
	Properties  map[string]Property `json:"properties,omitempty"`
	Format      string      `json:"format,omitempty"`
	Minimum     *float64    `json:"minimum,omitempty"`
	Maximum     *float64    `json:"maximum,omitempty"`
}

// InitializeRequest for initialize method
type InitializeRequest struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    ClientCapabilities     `json:"capabilities"`
	ClientInfo      ClientInfo             `json:"clientInfo"`
}

// InitializeResponse for initialize method response
type InitializeResponse struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    ServerCapabilities     `json:"capabilities"`
	ServerInfo      ServerInfo             `json:"serverInfo"`
}

// ClientCapabilities defines what the client supports
type ClientCapabilities struct {
	Experimental map[string]interface{} `json:"experimental,omitempty"`
}

// ServerCapabilities defines what the server supports
type ServerCapabilities struct {
	Tools        *ToolsCapability       `json:"tools,omitempty"`
	Experimental map[string]interface{} `json:"experimental,omitempty"`
}

// ToolsCapability indicates tool support
type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// ClientInfo provides client information
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ServerInfo provides server information
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ToolsListRequest for tools/list method
type ToolsListRequest struct{}

// ToolsListResponse for tools/list method response
type ToolsListResponse struct {
	Tools []Tool `json:"tools"`
}

// ToolCallRequest for tools/call method
type ToolCallRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// ToolCallResponse for tools/call method response
type ToolCallResponse struct {
	Content []ToolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ToolContent represents content returned by a tool
type ToolContent struct {
	Type string      `json:"type"`
	Text string      `json:"text,omitempty"`
	Data interface{} `json:"data,omitempty"`
}

// Handler defines the interface for tool handlers
type Handler interface {
	Execute(ctx context.Context, args map[string]interface{}) (*ToolCallResponse, error)
}

// SearchVectorsArgs for search_vectors tool
type SearchVectorsArgs struct {
	Collection      string                 `json:"collection"`
	Query           string                 `json:"query,omitempty"`
	Vector          []float32              `json:"vector,omitempty"`
	Limit           int                    `json:"limit,omitempty"`
	Filters         map[string]interface{} `json:"filters,omitempty"`
	IncludeMetadata bool                   `json:"includeMetadata,omitempty"`
	SessionID       string                 `json:"sessionId,omitempty"`
	UserID          string                 `json:"userId,omitempty"`
}

// AddVectorsArgs for add_vectors tool
type AddVectorsArgs struct {
	Collection string        `json:"collection"`
	Vectors    []VectorInput `json:"vectors"`
}

// VectorInput represents input for adding a vector
type VectorInput struct {
	ID       string                 `json:"id,omitempty"`
	Content  string                 `json:"content,omitempty"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// GetVectorArgs for get_vector tool
type GetVectorArgs struct {
	Collection string `json:"collection"`
	ID         string `json:"id"`
}

// SearchResult represents a search result
type SearchResult struct {
	ID       string                 `json:"id"`
	Score    float32                `json:"score"`
	Content  string                 `json:"content,omitempty"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// VectorData represents stored vector data
type VectorData struct {
	ID        string                 `json:"id"`
	Content   string                 `json:"content,omitempty"`
	Vector    []float32              `json:"vector"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"createdAt"`
	UpdatedAt time.Time              `json:"updatedAt"`
}

// AddVectorsResult represents the result of adding vectors
type AddVectorsResult struct {
	Added   int      `json:"added"`
	IDs     []string `json:"ids"`
	Errors  []string `json:"errors,omitempty"`
}