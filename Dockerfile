# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the server binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o embeddix-api ./cmd/embeddix-api

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1000 -S embeddix && \
    adduser -u 1000 -S embeddix -G embeddix

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/embeddix-api .

# Create data directory
RUN mkdir -p /app/data && chown -R embeddix:embeddix /app

# Switch to non-root user
USER embeddix

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Default command
ENTRYPOINT ["./embeddix-api"]

# Default arguments (can be overridden)
CMD ["-host", "0.0.0.0", "-port", "8080", "-db", "bolt", "-path", "/app/data/embeddix.db"]