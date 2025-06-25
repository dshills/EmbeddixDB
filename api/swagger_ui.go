package api

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"path/filepath"
)

// swaggerUIHTML contains the Swagger UI HTML template
const swaggerUIHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EmbeddixDB API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        body {
            margin: 0;
            background: #fafafa;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/swagger.yaml",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'put', 'post', 'delete', 'options', 'head', 'patch', 'trace']
            });
        };
    </script>
</body>
</html>`

// setupSwaggerUI adds Swagger UI endpoints to the server
func (s *Server) setupSwaggerUI() {
	// Serve the Swagger UI HTML
	s.router.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, swaggerUIHTML)
	}).Methods("GET")

	// Serve the Swagger YAML file
	s.router.HandleFunc("/swagger.yaml", func(w http.ResponseWriter, r *http.Request) {
		// Read the swagger.yaml file
		yamlContent, err := swaggerYAML()
		if err != nil {
			http.Error(w, "Failed to load API documentation", http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/x-yaml")
		w.Write(yamlContent)
	}).Methods("GET")

	// Also serve at /docs/ with trailing slash
	s.router.HandleFunc("/docs/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/docs", http.StatusMovedPermanently)
	}).Methods("GET")

	// Serve Swagger JSON (converted from YAML)
	s.router.HandleFunc("/swagger.json", s.handleSwaggerJSON).Methods("GET")
}

// handleSwaggerJSON serves the Swagger spec as JSON
func (s *Server) handleSwaggerJSON(w http.ResponseWriter, r *http.Request) {
	// For now, we'll just redirect to the YAML version
	// In production, you might want to convert YAML to JSON
	http.Error(w, "JSON format not yet implemented. Please use /swagger.yaml", http.StatusNotImplemented)
}

// swaggerYAML returns the swagger.yaml content from file
func swaggerYAML() ([]byte, error) {
	// Try to read from the api directory first
	paths := []string{
		"api/swagger.yaml",
		"./swagger.yaml",
		filepath.Join(".", "api", "swagger.yaml"),
	}
	
	for _, path := range paths {
		data, err := ioutil.ReadFile(path)
		if err == nil {
			return data, nil
		}
	}
	
	// If not found, return embedded content as fallback
	return []byte(embeddedSwaggerYAML), nil
}

// ReDoc UI as an alternative to Swagger UI
const redocHTML = `<!DOCTYPE html>
<html>
  <head>
    <title>EmbeddixDB API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
  </head>
  <body>
    <redoc spec-url='/swagger.yaml'></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"></script>
  </body>
</html>`

// setupReDoc adds ReDoc documentation endpoint
func (s *Server) setupReDoc() {
	s.router.HandleFunc("/redoc", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, redocHTML)
	}).Methods("GET")
}

// embeddedSwaggerYAML is a minimal embedded version of the API spec
// The full version should be loaded from the swagger.yaml file
const embeddedSwaggerYAML = `swagger: "2.0"
info:
  title: EmbeddixDB API
  description: High-performance vector database for embeddings
  version: 1.0.0
basePath: /
host: localhost:8080
schemes:
  - http
  - https
paths:
  /health:
    get:
      summary: Check server health
      responses:
        200:
          description: Server is healthy`