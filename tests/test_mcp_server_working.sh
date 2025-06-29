#!/bin/bash
# Working test script for MCP server

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}EmbeddixDB MCP Server Test Suite${NC}"
echo "=================================="

# Ensure build directory exists
mkdir -p build

# Build the server
echo -e "\n${YELLOW}Building MCP server...${NC}"
go build -o build/embeddix-mcp ./cmd/embeddix-mcp/

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"

# Test 1: Initialize request
echo -e "\n${YELLOW}Test 1: Initialize${NC}"
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}' | \
    ./build/embeddix-mcp -persistence memory 2>&1 | grep '^{' | jq .

# Test 2: Tools list request
echo -e "\n${YELLOW}Test 2: Tools List${NC}"
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}' | \
    ./build/embeddix-mcp -persistence memory 2>&1 | grep '^{' | jq '.result.tools | length' | \
    xargs -I {} echo "Found {} tools"

# Test 3: Create and use collection (using bolt for persistence)
echo -e "\n${YELLOW}Test 3: Collection Operations${NC}"

# Create temporary database
TEMP_DB=$(mktemp /tmp/embeddix_test.XXXXXX.db)
echo "Using temporary database: $TEMP_DB"

# Create collection
echo -e "\n${GREEN}Creating collection...${NC}"
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_collection","arguments":{"name":"test","dimension":128,"distance":"cosine","indexType":"hnsw"}},"id":3}' | \
    ./build/embeddix-mcp -persistence bolt -data "$TEMP_DB" 2>&1 | grep '^{' | jq -r '.result.content[0].text'

# List collections
echo -e "\n${GREEN}Listing collections...${NC}"
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_collections","arguments":{}},"id":4}' | \
    ./build/embeddix-mcp -persistence bolt -data "$TEMP_DB" 2>&1 | grep '^{' | jq -r '.result.content[0].text'

# Add vector with raw data
echo -e "\n${GREEN}Adding vector...${NC}"
VECTOR=$(python3 -c "print(','.join([str(i/128.0) for i in range(128)]))")
echo "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"add_vectors\",\"arguments\":{\"collection\":\"test\",\"vectors\":[{\"id\":\"v1\",\"vector\":[$VECTOR],\"metadata\":{\"type\":\"test\"}}]}},\"id\":5}" | \
    ./build/embeddix-mcp -persistence bolt -data "$TEMP_DB" 2>&1 | grep '^{' | jq -r '.result.content[0].text'

# Get vector
echo -e "\n${GREEN}Getting vector...${NC}"
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_vector","arguments":{"collection":"test","id":"v1"}},"id":6}' | \
    ./build/embeddix-mcp -persistence bolt -data "$TEMP_DB" 2>&1 | grep '^{' | jq -r '.result.content[0].text'

# Delete vector
echo -e "\n${GREEN}Deleting vector...${NC}"
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"delete_vector","arguments":{"collection":"test","id":"v1"}},"id":7}' | \
    ./build/embeddix-mcp -persistence bolt -data "$TEMP_DB" 2>&1 | grep '^{' | jq -r '.result.content[0].text'

# Delete collection
echo -e "\n${GREEN}Deleting collection...${NC}"
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"delete_collection","arguments":{"name":"test"}},"id":8}' | \
    ./build/embeddix-mcp -persistence bolt -data "$TEMP_DB" 2>&1 | grep '^{' | jq -r '.result.content[0].text'

# Cleanup
rm -f "$TEMP_DB"

echo -e "\n${GREEN}All tests completed!${NC}"