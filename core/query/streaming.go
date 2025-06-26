package query

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"
)

// StreamingSearchResult represents a search result that can be streamed
type StreamingSearchResult struct {
	SearchResult
	Timestamp time.Time
	Sequence  int64
}

// ResultStream provides streaming access to search results
type ResultStream interface {
	// Next returns the next result or io.EOF when done
	Next() (StreamingSearchResult, error)

	// Peek returns the next result without consuming it
	Peek() (StreamingSearchResult, error)

	// Skip skips n results
	Skip(n int) error

	// Close closes the stream
	Close() error

	// Context returns the stream's context
	Context() context.Context
}

// BufferedResultStream implements ResultStream with buffering
type BufferedResultStream struct {
	ctx        context.Context
	resultChan <-chan StreamingSearchResult
	errorChan  <-chan error
	buffer     []StreamingSearchResult
	bufferPos  int
	closed     bool
	mu         sync.Mutex
	sequence   int64
}

// NewBufferedResultStream creates a new buffered result stream
func NewBufferedResultStream(ctx context.Context, bufferSize int) (*BufferedResultStream, chan<- StreamingSearchResult, chan<- error) {
	resultChan := make(chan StreamingSearchResult, bufferSize)
	errorChan := make(chan error, 1)

	return &BufferedResultStream{
		ctx:        ctx,
		resultChan: resultChan,
		errorChan:  errorChan,
		buffer:     make([]StreamingSearchResult, 0, bufferSize),
		bufferPos:  0,
	}, resultChan, errorChan
}

// Next returns the next result from the stream
func (s *BufferedResultStream) Next() (StreamingSearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return StreamingSearchResult{}, io.EOF
	}

	// Check buffer first
	if s.bufferPos < len(s.buffer) {
		result := s.buffer[s.bufferPos]
		s.bufferPos++
		return result, nil
	}

	// Try to read from channel
	select {
	case result, ok := <-s.resultChan:
		if !ok {
			s.closed = true
			return StreamingSearchResult{}, io.EOF
		}
		s.sequence++
		result.Sequence = s.sequence
		return result, nil

	case err := <-s.errorChan:
		s.closed = true
		return StreamingSearchResult{}, err

	case <-s.ctx.Done():
		s.closed = true
		return StreamingSearchResult{}, s.ctx.Err()
	}
}

// Peek returns the next result without consuming it
func (s *BufferedResultStream) Peek() (StreamingSearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return StreamingSearchResult{}, io.EOF
	}

	// Check buffer first
	if s.bufferPos < len(s.buffer) {
		return s.buffer[s.bufferPos], nil
	}

	// Try to read and buffer
	select {
	case result, ok := <-s.resultChan:
		if !ok {
			s.closed = true
			return StreamingSearchResult{}, io.EOF
		}
		s.sequence++
		result.Sequence = s.sequence
		s.buffer = append(s.buffer, result)
		return result, nil

	case err := <-s.errorChan:
		s.closed = true
		return StreamingSearchResult{}, err

	case <-s.ctx.Done():
		s.closed = true
		return StreamingSearchResult{}, s.ctx.Err()

	default:
		// Non-blocking check failed
		return StreamingSearchResult{}, fmt.Errorf("no result available")
	}
}

// Skip skips n results
func (s *BufferedResultStream) Skip(n int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return io.EOF
	}

	// Skip buffered results first
	bufferedSkip := min(n, len(s.buffer)-s.bufferPos)
	s.bufferPos += bufferedSkip
	n -= bufferedSkip

	// Skip remaining from channel
	for i := 0; i < n; i++ {
		select {
		case _, ok := <-s.resultChan:
			if !ok {
				s.closed = true
				return io.EOF
			}
			s.sequence++

		case err := <-s.errorChan:
			s.closed = true
			return err

		case <-s.ctx.Done():
			s.closed = true
			return s.ctx.Err()
		}
	}

	return nil
}

// Close closes the stream
func (s *BufferedResultStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.closed = true
	// Drain channels to prevent goroutine leaks
	go func() {
		for range s.resultChan {
		}
		for range s.errorChan {
		}
	}()

	return nil
}

// Context returns the stream's context
func (s *BufferedResultStream) Context() context.Context {
	return s.ctx
}

// StreamingQuery represents a streaming query configuration
type StreamingQuery struct {
	Request       SearchRequest
	BatchSize     int
	FlushInterval time.Duration
	MaxBufferSize int
}

// StreamingExecutor handles streaming query execution
type StreamingExecutor struct {
	executor *ParallelExecutor
	config   StreamingConfig
	metrics  *StreamingMetrics
}

// StreamingConfig configures streaming behavior
type StreamingConfig struct {
	DefaultBatchSize     int
	DefaultFlushInterval time.Duration
	MaxStreamDuration    time.Duration
	MaxConcurrentStreams int
}

// DefaultStreamingConfig returns default streaming configuration
func DefaultStreamingConfig() StreamingConfig {
	return StreamingConfig{
		DefaultBatchSize:     100,
		DefaultFlushInterval: 50 * time.Millisecond,
		MaxStreamDuration:    5 * time.Minute,
		MaxConcurrentStreams: 1000,
	}
}

// StreamingMetrics tracks streaming performance
type StreamingMetrics struct {
	mu                sync.RWMutex
	ActiveStreams     int64
	TotalStreams      int64
	StreamedResults   int64
	AverageStreamTime time.Duration
}

// NewStreamingExecutor creates a new streaming executor
func NewStreamingExecutor(executor *ParallelExecutor, config StreamingConfig) *StreamingExecutor {
	return &StreamingExecutor{
		executor: executor,
		config:   config,
		metrics:  &StreamingMetrics{},
	}
}

// ExecuteStreaming executes a query with streaming results
func (se *StreamingExecutor) ExecuteStreaming(ctx context.Context, collection string, query StreamingQuery) (ResultStream, error) {
	// Apply stream duration limit
	streamCtx, cancel := context.WithTimeout(ctx, se.config.MaxStreamDuration)

	// Create buffered stream
	stream, resultChan, errorChan := NewBufferedResultStream(streamCtx, query.MaxBufferSize)

	// Track stream
	se.metrics.mu.Lock()
	se.metrics.ActiveStreams++
	se.metrics.TotalStreams++
	se.metrics.mu.Unlock()

	// Start async execution
	go func() {
		defer cancel()
		defer close(resultChan)
		defer close(errorChan)
		defer func() {
			se.metrics.mu.Lock()
			se.metrics.ActiveStreams--
			se.metrics.mu.Unlock()
		}()

		// Execute query and stream results
		se.streamResults(streamCtx, collection, query, resultChan, errorChan)
	}()

	return stream, nil
}

// streamResults performs the actual streaming execution
func (se *StreamingExecutor) streamResults(ctx context.Context, collection string, query StreamingQuery, resultChan chan<- StreamingSearchResult, errorChan chan<- error) {
	batchBuffer := make([]SearchResult, 0, query.BatchSize)
	flushTimer := time.NewTicker(query.FlushInterval)
	defer flushTimer.Stop()

	// This would be the actual implementation
	// For now, simulating streaming results
	for i := 0; i < 1000; i++ {
		select {
		case <-ctx.Done():
			return

		case <-flushTimer.C:
			// Flush current batch
			se.flushBatch(batchBuffer, resultChan)
			batchBuffer = batchBuffer[:0]

		default:
			// Add to batch
			result := SearchResult{
				ID:    fmt.Sprintf("stream_%d", i),
				Score: float32(0.95 - float64(i)*0.0001),
			}

			batchBuffer = append(batchBuffer, result)

			// Flush if batch is full
			if len(batchBuffer) >= query.BatchSize {
				se.flushBatch(batchBuffer, resultChan)
				batchBuffer = batchBuffer[:0]
			}
		}

		// Simulate some processing time
		time.Sleep(time.Microsecond * 100)
	}

	// Flush remaining results
	if len(batchBuffer) > 0 {
		se.flushBatch(batchBuffer, resultChan)
	}
}

// flushBatch sends a batch of results to the stream
func (se *StreamingExecutor) flushBatch(batch []SearchResult, resultChan chan<- StreamingSearchResult) {
	timestamp := time.Now()

	for _, result := range batch {
		streamResult := StreamingSearchResult{
			SearchResult: result,
			Timestamp:    timestamp,
		}

		select {
		case resultChan <- streamResult:
			se.metrics.mu.Lock()
			se.metrics.StreamedResults++
			se.metrics.mu.Unlock()
		default:
			// Channel full, drop result
			// In production, we might want to block or use backpressure
		}
	}
}

// ResultIterator provides a convenient iterator interface over streaming results
type ResultIterator struct {
	stream ResultStream
	err    error
}

// NewResultIterator creates a new result iterator
func NewResultIterator(stream ResultStream) *ResultIterator {
	return &ResultIterator{
		stream: stream,
	}
}

// HasNext checks if there are more results
func (it *ResultIterator) HasNext() bool {
	if it.err != nil {
		return false
	}

	_, err := it.stream.Peek()
	if err == io.EOF {
		return false
	}
	if err != nil {
		it.err = err
		return false
	}

	return true
}

// Next returns the next result
func (it *ResultIterator) Next() (StreamingSearchResult, error) {
	if it.err != nil {
		return StreamingSearchResult{}, it.err
	}

	result, err := it.stream.Next()
	if err != nil {
		it.err = err
		return StreamingSearchResult{}, err
	}

	return result, nil
}

// Error returns any error encountered during iteration
func (it *ResultIterator) Error() error {
	return it.err
}

// Close closes the underlying stream
func (it *ResultIterator) Close() error {
	return it.stream.Close()
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
