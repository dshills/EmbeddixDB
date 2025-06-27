#!/bin/bash
# Test script for MCP server

# Ensure build directory exists
mkdir -p build

# Build the server
echo "Building MCP server..."
go build -o build/embeddix-mcp ./cmd/embeddix-mcp/

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "Build successful!"

# Test initialize request
echo "Testing initialize request..."
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}' | ./build/embeddix-mcp -persistence memory 2>/dev/null | jq .

# Test tools/list request
echo -e "\nTesting tools/list request..."
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}' | ./build/embeddix-mcp -persistence memory 2>/dev/null | jq .

echo -e "\nMCP server test complete!"

# Note: Not cleaning up the binary as it's in the build directory
# which can be cleaned with 'make clean'