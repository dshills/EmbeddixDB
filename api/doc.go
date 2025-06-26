package api

// NOTE: This file contains the swagger route documentation.
// The actual implementation is in handlers.go

// swagger:route GET /health health getHealth
//
// # Check server health
//
// Returns the current health status of the server.
//
// Responses:
//
//	200: healthResponse
func swaggerRouteHealth() {}

// swagger:route GET /collections collections listCollections
//
// # List all collections
//
// Returns a list of all vector collections in the database.
//
// Responses:
//
//	200: collectionsResponse
//	500: errorResponse
func swaggerRouteListCollections() {}

// swagger:route POST /collections collections createCollection
//
// # Create a new collection
//
// Creates a new vector collection with the specified configuration.
//
// Parameters:
//   - name: body
//     in: body
//     required: true
//     type: CreateCollectionRequest
//
// Responses:
//
//	201: collectionResponse
//	400: errorResponse
//	500: errorResponse
func swaggerRouteCreateCollection() {}

// swagger:route GET /collections/{collection} collections getCollection
//
// # Get collection details
//
// Returns detailed information about a specific collection.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//
// Responses:
//
//	200: collectionResponse
//	404: errorResponse
//	500: errorResponse
func swaggerRouteGetCollection() {}

// swagger:route DELETE /collections/{collection} collections deleteCollection
//
// # Delete a collection
//
// Permanently deletes a collection and all its vectors.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//
// Responses:
//
//	200: messageResponse
//	500: errorResponse
func swaggerRouteDeleteCollection() {}

// swagger:route POST /collections/{collection}/vectors vectors addVector
//
// # Add a vector
//
// Adds a single vector to the specified collection.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: body
//     in: body
//     required: true
//     type: AddVectorRequest
//
// Responses:
//
//	201: vectorResponse
//	400: errorResponse
//	500: errorResponse
func swaggerRouteAddVector() {}

// swagger:route POST /collections/{collection}/vectors/batch vectors addVectorsBatch
//
// # Add vectors in batch
//
// Adds multiple vectors to the collection in a single operation.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: body
//     in: body
//     required: true
//     type: []AddVectorRequest
//
// Responses:
//
//	201: batchAddResponse
//	400: errorResponse
//	500: errorResponse
func swaggerRouteAddVectorsBatch() {}

// swagger:route GET /collections/{collection}/vectors/{id} vectors getVector
//
// # Get a vector
//
// Retrieves a specific vector by its ID.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: id
//     in: path
//     required: true
//     type: string
//
// Responses:
//
//	200: vectorResponse
//	404: errorResponse
func swaggerRouteGetVector() {}

// swagger:route PUT /collections/{collection}/vectors/{id} vectors updateVector
//
// # Update a vector
//
// Updates an existing vector's values and/or metadata.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: id
//     in: path
//     required: true
//     type: string
//   - name: body
//     in: body
//     required: true
//     type: AddVectorRequest
//
// Responses:
//
//	200: vectorResponse
//	400: errorResponse
//	500: errorResponse
func swaggerRouteUpdateVector() {}

// swagger:route DELETE /collections/{collection}/vectors/{id} vectors deleteVector
//
// # Delete a vector
//
// Removes a vector from the collection.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: id
//     in: path
//     required: true
//     type: string
//
// Responses:
//
//	200: messageResponse
//	500: errorResponse
func swaggerRouteDeleteVector() {}

// swagger:route POST /collections/{collection}/search search searchVectors
//
// # Search for nearest neighbors
//
// Performs k-nearest neighbor search to find the most similar vectors.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: body
//     in: body
//     required: true
//     type: SearchRequest
//
// Responses:
//
//	200: searchResultsResponse
//	400: errorResponse
//	500: errorResponse
func swaggerRouteSearch() {}

// swagger:route POST /collections/{collection}/search/batch search batchSearch
//
// # Batch search
//
// Performs multiple k-nearest neighbor searches in a single request.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: body
//     in: body
//     required: true
//     type: []SearchRequest
//
// Responses:
//
//	200: body:[][]SearchResult
//	400: errorResponse
//	500: errorResponse
func swaggerRouteBatchSearch() {}

// swagger:route POST /collections/{collection}/search/range search rangeSearch
//
// # Range search
//
// Finds all vectors within a specified distance threshold from the query vector.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//   - name: body
//     in: body
//     required: true
//     type: RangeSearchRequest
//
// Responses:
//
//	200: rangeSearchResponse
//	400: errorResponse
//	500: errorResponse
func swaggerRouteRangeSearch() {}

// swagger:route GET /stats stats getStats
//
// # Get overall statistics
//
// Returns statistics about all collections and vectors in the database.
//
// Responses:
//
//	200: statsResponse
//	500: errorResponse
func swaggerRouteStats() {}

// swagger:route GET /collections/{collection}/stats stats getCollectionStats
//
// # Get collection statistics
//
// Returns statistics about a specific collection.
//
// Parameters:
//   - name: collection
//     in: path
//     required: true
//     type: string
//
// Responses:
//
//	200: collectionStatsResponse
//	404: errorResponse
//	500: errorResponse
func swaggerRouteCollectionStats() {}
