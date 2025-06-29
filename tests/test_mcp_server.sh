#!/bin/bash
# Simple test script for MCP server using file persistence

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Create test data directory and file
TEST_DATA_DIR="./test_data_mcp"
rm -rf "$TEST_DATA_DIR"
mkdir -p "$TEST_DATA_DIR"
TEST_DB_FILE="${TEST_DATA_DIR}/embeddix.db"

# Helper function to run a test
run_test() {
    local test_name="$1"
    local request="$2"
    local expected_field="$3"
    local expect_error="${4:-false}"
    
    echo -e "\n${YELLOW}Testing: ${test_name}${NC}"
    echo "Request: ${request}"
    
    # Run the test - using configuration file for Ollama support
    # Filter out non-JSON lines (stderr output) and get the JSON response
    response=$(echo "${request}" | ./build/embeddix-mcp -config ~/.embeddixdb.yml 2>&1 | grep '^{' | head -n 1)
    
    # Check if response is valid JSON
    if ! echo "${response}" | jq . >/dev/null 2>&1; then
        echo -e "${RED}✗ Failed: Invalid JSON response${NC}"
        echo "Response: ${response}"
        ((TESTS_FAILED++))
        return
    fi
    
    # Check for JSON-RPC level error
    if echo "${response}" | jq -e '.error' >/dev/null 2>&1; then
        error_msg=$(echo "${response}" | jq -r '.error.message')
        if [ "${expect_error}" = "true" ]; then
            echo -e "${GREEN}✓ Passed (Expected JSON-RPC error: ${error_msg})${NC}"
            echo "Full response:"
            echo "${response}" | jq .
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ Failed: JSON-RPC error: ${error_msg}${NC}"
            echo "Full response:"
            echo "${response}" | jq .
            ((TESTS_FAILED++))
        fi
        return
    fi
    
    # Check for tool-level error (isError field in result)
    if echo "${response}" | jq -e '.result.isError == true' >/dev/null 2>&1; then
        error_text=$(echo "${response}" | jq -r '.result.content[0].text // "Unknown error"')
        if [ "${expect_error}" = "true" ]; then
            echo -e "${GREEN}✓ Passed (Expected tool error: ${error_text})${NC}"
            echo "Full response:"
            echo "${response}" | jq .
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ Failed: Tool error: ${error_text}${NC}"
            echo "Full response:"
            echo "${response}" | jq .
            ((TESTS_FAILED++))
        fi
        return
    fi
    
    # If we expected an error but didn't get one
    if [ "${expect_error}" = "true" ]; then
        echo -e "${RED}✗ Failed: Expected an error but request succeeded${NC}"
        echo "Response:"
        echo "${response}" | jq .
        ((TESTS_FAILED++))
        return
    fi
    
    # Check for expected field if specified
    if [ -n "${expected_field}" ]; then
        if echo "${response}" | jq -e "${expected_field}" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Passed${NC}"
            echo "Response:"
            echo "${response}" | jq .
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ Failed: Expected field '${expected_field}' not found${NC}"
            echo "Response:"
            echo "${response}" | jq .
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${GREEN}✓ Passed${NC}"
        echo "Response:"
        echo "${response}" | jq .
        ((TESTS_PASSED++))
    fi
}

# Ensure build directory exists
mkdir -p build

# Build the server
echo "Building MCP server..."
go build -o build/embeddix-mcp ./cmd/embeddix-mcp/

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"

# Clean up any existing database
rm -f /Users/dshills/.embeddixdb/data/embeddix.db

# Generate 768-dimensional test vector
TEST_VECTOR=$(python3 -c "import json; print(json.dumps([round((i % 10 + 1) * 0.1, 1) for i in range(768)]))")

# Run tests
echo -e "\n${YELLOW}Running tests with BoltDB persistence...${NC}"

# Test 1: Initialize request
run_test "Initialize" \
    '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}' \
    '.result.protocolVersion'

# Test 2: Tools list request
run_test "Tools List" \
    '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}' \
    '.result.tools'

# Test 3: Create collection (768 dimensions for nomic-embed-text)
run_test "Create Collection" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_collection","arguments":{"name":"test_collection","dimension":768,"distance":"cosine","indexType":"hnsw"}},"id":3}' \
    '.result.content'

# Test 4: List collections (should show our created collection)
run_test "List Collections" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_collections","arguments":{}},"id":4}' \
    '.result.content'

# Test 5: Add vectors with content (will use Ollama embedding)
run_test "Add Vectors with Content" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"add_vectors","arguments":{"collection":"test_collection","vectors":[{"id":"vec1","content":"This is a test document about machine learning","metadata":{"type":"document","category":"ML"}},{"id":"vec2","content":"Neural networks are powerful tools","metadata":{"type":"document","category":"AI"}}]}},"id":5}' \
    '.result.content'

# Test 6: Add vectors with raw vector data (768 dimensions)
run_test "Add Vectors with Raw Data" \
    "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"add_vectors\",\"arguments\":{\"collection\":\"test_collection\",\"vectors\":[{\"id\":\"vec3\",\"vector\":${TEST_VECTOR},\"metadata\":{\"type\":\"synthetic\",\"id\":\"3\"}}]}},\"id\":6}" \
    '.result.content'

# Test 7: Get specific vector
run_test "Get Vector" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_vector","arguments":{"collection":"test_collection","id":"vec1"}},"id":7}' \
    '.result.content'

# Test 8: Search vectors by query
run_test "Search Vectors by Query" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"search_vectors","arguments":{"collection":"test_collection","query":"machine learning neural networks","limit":3}},"id":8}' \
    '.result.content'

# Test 9: Search vectors with raw vector (768 dimensions)
run_test "Search Vectors with Raw Vector" \
    "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"search_vectors\",\"arguments\":{\"collection\":\"test_collection\",\"vector\":${TEST_VECTOR},\"limit\":2}},\"id\":9}" \
    '.result.content'

# Test 10: Delete vector
run_test "Delete Vector" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"delete_vector","arguments":{"collection":"test_collection","id":"vec3"}},"id":12}' \
    '.result.content'

# Test 11: Get deleted vector (should fail)
run_test "Get Deleted Vector (Expected to fail)" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_vector","arguments":{"collection":"test_collection","id":"vec3"}},"id":13}' \
    '' \
    'true'

# Test 12: Create collection with different parameters
run_test "Create Collection with L2 Distance" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_collection","arguments":{"name":"l2_collection","dimension":128,"distance":"l2","indexType":"flat"}},"id":14}' \
    '.result.content'

# Test 13: Delete collection
run_test "Delete Collection" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"delete_collection","arguments":{"name":"l2_collection"}},"id":17}' \
    '.result.content'

# Test 14: List collections after deletion
run_test "List Collections After Deletion" \
    '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_collections","arguments":{}},"id":18}' \
    '.result.content'

# Cleanup
echo -e "\n${YELLOW}Cleaning up test data...${NC}"
rm -rf "$TEST_DATA_DIR"
rm -f generate_test_vector.py

# Test summary
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}Test Summary:${NC}"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi