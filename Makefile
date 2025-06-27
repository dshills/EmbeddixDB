.PHONY: all build test clean run benchmark docker-build docker-run help

# Variables
BINARY_NAME=embeddixdb
DOCKER_IMAGE=embeddixdb:latest
GO=go
GOFLAGS=-v

# Default target
all: test build

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Quick Start:"
	@echo "  make test        - Run fast tests (recommended for development)"
	@echo "  make test-all    - Run all tests including long-running integration tests"
	@echo ""
	@echo "All Targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' |  sed -e 's/^/ /'

## build: Build the server binary
build:
	$(GO) build $(GOFLAGS) -o ./build/$(BINARY_NAME) ./cmd/server

## build-all: Build all binaries
build-all:
	$(GO) build $(GOFLAGS) -o ./build/$(BINARY_NAME) ./cmd/server
	$(GO) build $(GOFLAGS) -o ./build/$(BINARY_NAME)-benchmark ./cmd/benchmark

install: build
	cp ./build/$(BINARY_NAME) /Users/dshills/Development/Go/bin/.

## test: Run fast tests only (excludes integration tests and benchmarks)
test:
	$(GO) test -short -timeout=2m -race -coverprofile=coverage.out -covermode=atomic ./... || true
	@echo "\n=== Test Coverage ==="
	$(GO) tool cover -func=coverage.out | tail -1

## test-all: Run all tests including long-running integration tests
test-all:
	$(GO) test -timeout=30m -tags=integration -race -coverprofile=coverage-all.out -covermode=atomic ./... || true
	@echo "\n=== Full Test Coverage ==="
	$(GO) tool cover -func=coverage-all.out | tail -1

## test-verbose: Run fast tests with verbose output
test-verbose:
	$(GO) test -short -timeout=2m $(GOFLAGS) -v ./...

## test-verbose-all: Run all tests with verbose output
test-verbose-all:
	$(GO) test -timeout=30m -tags=integration $(GOFLAGS) -v ./...

## test-quiet: Run fast tests quietly, only show summary
test-quiet:
	@$(GO) test -short -timeout=2m -race -coverprofile=coverage.out -covermode=atomic ./... > /dev/null 2>&1 && echo "✅ All tests passed" || echo "❌ Some tests failed"
	@$(GO) tool cover -func=coverage.out | tail -1

## test-coverage: Run fast tests with coverage and generate HTML report
test-coverage:
	$(GO) test -short -timeout=2m -race -coverprofile=coverage.out -covermode=atomic ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"
	$(GO) tool cover -func=coverage.out | tail -1

## test-coverage-all: Run all tests with coverage and generate HTML report
test-coverage-all:
	$(GO) test -timeout=30m -tags=integration -race -coverprofile=coverage-all.out -covermode=atomic ./...
	$(GO) tool cover -html=coverage-all.out -o coverage-all.html
	@echo "Full coverage report generated: coverage-all.html"
	$(GO) tool cover -func=coverage-all.out | tail -1

## benchmark: Run benchmarks
benchmark:
	$(GO) run ./cmd/benchmark -vectors 1000 -queries 100

## benchmark-go: Run Go benchmark tests
benchmark-go:
	$(GO) test -bench=. -benchmem -timeout=30m ./...

## run: Run the server
run:
	$(GO) run ./cmd/server

## run-dev: Run with hot reload (requires air)
run-dev:
	air -c .air.toml

## clean: Clean build artifacts
clean:
	rm -f $(BINARY_NAME) $(BINARY_NAME)-benchmark
	rm -f coverage.out coverage.html coverage-all.out coverage-all.html
	rm -rf tmp/
	rm -rf data/

## docker-build: Build Docker image
docker-build:
	docker build -t $(DOCKER_IMAGE) .

## docker-run: Run Docker container
docker-run:
	docker run -d --name embeddixdb -p 8080:8080 $(DOCKER_IMAGE)

## docker-compose-up: Start services with docker-compose
docker-compose-up:
	docker-compose up -d

## docker-compose-down: Stop services
docker-compose-down:
	docker-compose down

## docker-compose-logs: View logs
docker-compose-logs:
	docker-compose logs -f

## docker-dev: Run development environment
docker-dev:
	docker-compose -f docker-compose.dev.yml up

## lint: Run linters
lint:
	golangci-lint run ./...

## fmt: Format code
fmt:
	$(GO) fmt ./...

## vet: Run go vet
vet:
	$(GO) vet ./...

## mod-tidy: Tidy go modules
mod-tidy:
	$(GO) mod tidy

## mod-download: Download dependencies
mod-download:
	$(GO) mod download

## install-tools: Install development tools
install-tools:
	go install github.com/air-verse/air@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

## api-test: Test the API with example client
api-test:
	$(GO) run ./examples/api_client/main.go

## generate: Generate code (if needed)
generate:
	$(GO) generate ./...

# Performance testing targets
## perf-memory: Run memory performance test
perf-memory:
	$(GO) run ./cmd/benchmark -db memory -vectors 10000 -queries 1000

## perf-bolt: Run BoltDB performance test
perf-bolt:
	$(GO) run ./cmd/benchmark -db bolt -vectors 10000 -queries 1000

## perf-badger: Run BadgerDB performance test
perf-badger:
	$(GO) run ./cmd/benchmark -db badger -vectors 10000 -queries 1000

## perf-compare: Compare index performance
perf-compare:
	$(GO) run ./cmd/benchmark -db memory -index flat -vectors 5000 -compare
