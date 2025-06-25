package core

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// VectorFingerprint represents a compact representation of a vector
type VectorFingerprint struct {
	Hash        uint64    // LSH hash for similarity detection
	ExactHash   [32]byte  // SHA256 for exact duplicate detection
	Dimension   int       // Vector dimension
	Magnitude   float32   // Vector magnitude for normalization
	CreatedAt   time.Time // When fingerprint was created
}

// DuplicateInfo contains information about duplicate vectors
type DuplicateInfo struct {
	OriginalID    string
	DuplicateIDs  []string
	Similarity    float32
	RefCount      int32
	SharedVector  []float32
}

// DeduplicationManager handles vector deduplication and reference counting
type DeduplicationManager struct {
	// Exact duplicates (hash -> vector info)
	exactDuplicates map[[32]byte]*DuplicateInfo
	
	// LSH buckets for near-duplicate detection
	lshBuckets map[uint64][]*VectorFingerprint
	
	// Vector ID to fingerprint mapping
	vectorFingerprints map[string]*VectorFingerprint
	
	// LSH configuration
	lshHashFunctions []LSHHashFunction
	numHashFunctions int
	similarityThreshold float32
	
	// Statistics
	stats DeduplicationStats
	mutex sync.RWMutex
}

// LSHHashFunction represents a locality-sensitive hash function
type LSHHashFunction struct {
	RandomVector []float32
	Offset       float32
}

// DeduplicationStats tracks deduplication performance
type DeduplicationStats struct {
	TotalVectors      int64
	ExactDuplicates   int64
	NearDuplicates    int64
	StorageSaved      int64 // Bytes saved through deduplication
	ComparisionsMade  int64
	AverageSimilarity float64
	mutex             sync.RWMutex
}

// DeduplicationConfig configures the deduplication system
type DeduplicationConfig struct {
	NumHashFunctions      int     // Number of LSH hash functions
	SimilarityThreshold   float32 // Threshold for near-duplicate detection
	EnableNearDuplicates  bool    // Whether to detect near-duplicates
	MaxCandidatesPerHash  int     // Max candidates to check per LSH bucket
}

// DefaultDeduplicationConfig returns sensible defaults
func DefaultDeduplicationConfig() DeduplicationConfig {
	return DeduplicationConfig{
		NumHashFunctions:     32,   // Good balance of accuracy vs performance
		SimilarityThreshold:  0.95, // 95% similarity threshold
		EnableNearDuplicates: true,
		MaxCandidatesPerHash: 100, // Limit search in crowded buckets
	}
}

// NewDeduplicationManager creates a new deduplication manager
func NewDeduplicationManager(dimension int, config DeduplicationConfig) *DeduplicationManager {
	dm := &DeduplicationManager{
		exactDuplicates:     make(map[[32]byte]*DuplicateInfo),
		lshBuckets:          make(map[uint64][]*VectorFingerprint),
		vectorFingerprints:  make(map[string]*VectorFingerprint),
		numHashFunctions:    config.NumHashFunctions,
		similarityThreshold: config.SimilarityThreshold,
	}
	
	// Initialize LSH hash functions
	dm.initializeLSHFunctions(dimension)
	
	return dm
}

// AddVector processes a new vector for deduplication
func (dm *DeduplicationManager) AddVector(vector Vector) (*DuplicateInfo, error) {
	fingerprint := dm.createFingerprint(vector)
	
	dm.mutex.Lock()
	defer dm.mutex.Unlock()
	
	// Check for exact duplicates first
	if duplicate, exists := dm.exactDuplicates[fingerprint.ExactHash]; exists {
		// Exact duplicate found
		duplicate.DuplicateIDs = append(duplicate.DuplicateIDs, vector.ID)
		duplicate.RefCount++
		dm.stats.ExactDuplicates++
		dm.stats.StorageSaved += int64(len(vector.Values) * 4) // 4 bytes per float32
		
		// Store fingerprint for this vector ID
		dm.vectorFingerprints[vector.ID] = fingerprint
		
		return duplicate, nil
	}
	
	// Check for near duplicates using LSH
	nearDuplicate := dm.findNearDuplicate(fingerprint, vector.Values)
	if nearDuplicate != nil {
		nearDuplicate.DuplicateIDs = append(nearDuplicate.DuplicateIDs, vector.ID)
		nearDuplicate.RefCount++
		dm.stats.NearDuplicates++
		dm.stats.StorageSaved += int64(len(vector.Values) * 4 / 2) // Partial savings
		
		dm.vectorFingerprints[vector.ID] = fingerprint
		return nearDuplicate, nil
	}
	
	// No duplicate found, create new entry
	duplicateInfo := &DuplicateInfo{
		OriginalID:   vector.ID,
		DuplicateIDs: []string{},
		Similarity:   1.0,
		RefCount:     1,
		SharedVector: make([]float32, len(vector.Values)),
	}
	copy(duplicateInfo.SharedVector, vector.Values)
	
	// Store in exact duplicates map
	dm.exactDuplicates[fingerprint.ExactHash] = duplicateInfo
	
	// Add to LSH buckets
	dm.addToLSHBuckets(fingerprint)
	
	// Store fingerprint
	dm.vectorFingerprints[vector.ID] = fingerprint
	
	dm.stats.TotalVectors++
	
	return duplicateInfo, nil
}

// RemoveVector removes a vector from deduplication tracking
func (dm *DeduplicationManager) RemoveVector(vectorID string) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()
	
	fingerprint, exists := dm.vectorFingerprints[vectorID]
	if !exists {
		return fmt.Errorf("vector ID %s not found in deduplication tracking", vectorID)
	}
	
	duplicate, exists := dm.exactDuplicates[fingerprint.ExactHash]
	if !exists {
		return fmt.Errorf("duplicate info not found for vector %s", vectorID)
	}
	
	// Decrease reference count
	duplicate.RefCount--
	
	// Remove from duplicate IDs list
	for i, id := range duplicate.DuplicateIDs {
		if id == vectorID {
			duplicate.DuplicateIDs = append(duplicate.DuplicateIDs[:i], duplicate.DuplicateIDs[i+1:]...)
			break
		}
	}
	
	// If this was the original, promote the first duplicate
	if duplicate.OriginalID == vectorID && len(duplicate.DuplicateIDs) > 0 {
		duplicate.OriginalID = duplicate.DuplicateIDs[0]
		duplicate.DuplicateIDs = duplicate.DuplicateIDs[1:]
	}
	
	// If no references left, remove completely
	if duplicate.RefCount <= 0 {
		delete(dm.exactDuplicates, fingerprint.ExactHash)
		dm.removeFromLSHBuckets(fingerprint)
	}
	
	delete(dm.vectorFingerprints, vectorID)
	
	return nil
}

// GetDuplicateInfo returns duplicate information for a vector
func (dm *DeduplicationManager) GetDuplicateInfo(vectorID string) (*DuplicateInfo, bool) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	
	fingerprint, exists := dm.vectorFingerprints[vectorID]
	if !exists {
		return nil, false
	}
	
	duplicate, exists := dm.exactDuplicates[fingerprint.ExactHash]
	return duplicate, exists
}

// FindSimilarVectors finds vectors similar to the given vector
func (dm *DeduplicationManager) FindSimilarVectors(vector []float32, threshold float32) []string {
	fingerprint := dm.createFingerprintFromValues(vector)
	
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	
	similarVectors := make([]string, 0)
	
	// Check LSH buckets
	candidates := dm.getLSHCandidates(fingerprint)
	
	for _, candidate := range candidates {
		// Get the actual vector from exact duplicates
		if duplicate, exists := dm.exactDuplicates[candidate.ExactHash]; exists {
			similarity, err := CosineSimilarity(vector, duplicate.SharedVector)
			if err == nil && similarity >= threshold {
				similarVectors = append(similarVectors, duplicate.OriginalID)
				similarVectors = append(similarVectors, duplicate.DuplicateIDs...)
			}
		}
	}
	
	return similarVectors
}

// GetStats returns deduplication statistics
func (dm *DeduplicationManager) GetStats() DeduplicationStats {
	dm.stats.mutex.RLock()
	defer dm.stats.mutex.RUnlock()
	
	return DeduplicationStats{
		TotalVectors:      dm.stats.TotalVectors,
		ExactDuplicates:   dm.stats.ExactDuplicates,
		NearDuplicates:    dm.stats.NearDuplicates,
		StorageSaved:      dm.stats.StorageSaved,
		ComparisionsMade:  dm.stats.ComparisionsMade,
		AverageSimilarity: dm.stats.AverageSimilarity,
	}
}

// Private methods

func (dm *DeduplicationManager) createFingerprint(vector Vector) *VectorFingerprint {
	return dm.createFingerprintFromValues(vector.Values)
}

func (dm *DeduplicationManager) createFingerprintFromValues(values []float32) *VectorFingerprint {
	// Create exact hash using SHA256
	hasher := sha256.New()
	for _, val := range values {
		bytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(bytes, math.Float32bits(val))
		hasher.Write(bytes)
	}
	var exactHash [32]byte
	copy(exactHash[:], hasher.Sum(nil))
	
	// Create LSH hash
	lshHash := dm.computeLSHHash(values)
	
	// Calculate magnitude
	var magnitude float32
	for _, val := range values {
		magnitude += val * val
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))
	
	return &VectorFingerprint{
		Hash:      lshHash,
		ExactHash: exactHash,
		Dimension: len(values),
		Magnitude: magnitude,
		CreatedAt: time.Now(),
	}
}

func (dm *DeduplicationManager) initializeLSHFunctions(dimension int) {
	rand.Seed(time.Now().UnixNano())
	
	dm.lshHashFunctions = make([]LSHHashFunction, dm.numHashFunctions)
	
	for i := 0; i < dm.numHashFunctions; i++ {
		// Create random hyperplane
		randomVector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			randomVector[j] = float32(rand.NormFloat64()) // Gaussian random
		}
		
		dm.lshHashFunctions[i] = LSHHashFunction{
			RandomVector: randomVector,
			Offset:       rand.Float32(), // Random offset for better distribution
		}
	}
}

func (dm *DeduplicationManager) computeLSHHash(vector []float32) uint64 {
	var hash uint64
	
	for i, hashFunc := range dm.lshHashFunctions {
		// Compute dot product with random hyperplane
		var dotProduct float32
		for j, val := range vector {
			if j < len(hashFunc.RandomVector) {
				dotProduct += val * hashFunc.RandomVector[j]
			}
		}
		
		// Add offset and determine which side of hyperplane
		if dotProduct+hashFunc.Offset > 0 {
			hash |= (1 << uint(i))
		}
	}
	
	return hash
}

func (dm *DeduplicationManager) findNearDuplicate(fingerprint *VectorFingerprint, vector []float32) *DuplicateInfo {
	candidates := dm.getLSHCandidates(fingerprint)
	
	bestMatch := (*DuplicateInfo)(nil)
	bestSimilarity := float32(0)
	
	for _, candidate := range candidates {
		if duplicate, exists := dm.exactDuplicates[candidate.ExactHash]; exists {
			similarity, err := CosineSimilarity(vector, duplicate.SharedVector)
			if err == nil && similarity >= dm.similarityThreshold && similarity > bestSimilarity {
				bestMatch = duplicate
				bestSimilarity = similarity
			}
			dm.stats.ComparisionsMade++
		}
	}
	
	if bestMatch != nil {
		bestMatch.Similarity = bestSimilarity
	}
	
	return bestMatch
}

func (dm *DeduplicationManager) getLSHCandidates(fingerprint *VectorFingerprint) []*VectorFingerprint {
	candidates := make([]*VectorFingerprint, 0)
	
	// Check exact hash bucket
	if bucket, exists := dm.lshBuckets[fingerprint.Hash]; exists {
		candidates = append(candidates, bucket...)
	}
	
	// Check similar hash buckets (Hamming distance 1-3)
	for hammingDistance := 1; hammingDistance <= 3; hammingDistance++ {
		similarHashes := dm.generateSimilarHashes(fingerprint.Hash, hammingDistance)
		for _, hash := range similarHashes {
			if bucket, exists := dm.lshBuckets[hash]; exists {
				candidates = append(candidates, bucket...)
			}
		}
	}
	
	// Limit candidates to avoid performance issues
	maxCandidates := 1000
	if len(candidates) > maxCandidates {
		candidates = candidates[:maxCandidates]
	}
	
	return candidates
}

func (dm *DeduplicationManager) generateSimilarHashes(originalHash uint64, hammingDistance int) []uint64 {
	if hammingDistance == 0 {
		return []uint64{originalHash}
	}
	
	var hashes []uint64
	
	// Generate all combinations of bit flips for the given Hamming distance
	dm.generateBitFlips(originalHash, 0, 0, hammingDistance, &hashes)
	
	return hashes
}

func (dm *DeduplicationManager) generateBitFlips(hash uint64, position, flips, maxFlips int, results *[]uint64) {
	if flips == maxFlips {
		*results = append(*results, hash)
		return
	}
	
	if position >= 64 {
		return
	}
	
	// Try flipping current bit
	flippedHash := hash ^ (1 << uint(position))
	dm.generateBitFlips(flippedHash, position+1, flips+1, maxFlips, results)
	
	// Try not flipping current bit
	dm.generateBitFlips(hash, position+1, flips, maxFlips, results)
}

func (dm *DeduplicationManager) addToLSHBuckets(fingerprint *VectorFingerprint) {
	bucket := dm.lshBuckets[fingerprint.Hash]
	bucket = append(bucket, fingerprint)
	dm.lshBuckets[fingerprint.Hash] = bucket
}

func (dm *DeduplicationManager) removeFromLSHBuckets(fingerprint *VectorFingerprint) {
	bucket := dm.lshBuckets[fingerprint.Hash]
	
	for i, fp := range bucket {
		if fp.ExactHash == fingerprint.ExactHash {
			// Remove from bucket
			bucket = append(bucket[:i], bucket[i+1:]...)
			break
		}
	}
	
	if len(bucket) == 0 {
		delete(dm.lshBuckets, fingerprint.Hash)
	} else {
		dm.lshBuckets[fingerprint.Hash] = bucket
	}
}

// Cleanup removes old fingerprints and optimizes bucket structure
func (dm *DeduplicationManager) Cleanup(maxAge time.Duration) {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()
	
	cutoff := time.Now().Add(-maxAge)
	
	// Remove old fingerprints
	for vectorID, fingerprint := range dm.vectorFingerprints {
		if fingerprint.CreatedAt.Before(cutoff) {
			dm.RemoveVector(vectorID)
		}
	}
	
	// Optimize LSH buckets by removing empty ones
	for hash, bucket := range dm.lshBuckets {
		if len(bucket) == 0 {
			delete(dm.lshBuckets, hash)
		}
	}
}

// GetBucketDistribution returns distribution of vectors across LSH buckets
func (dm *DeduplicationManager) GetBucketDistribution() map[string]int {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	
	distribution := make(map[string]int)
	
	for _, bucket := range dm.lshBuckets {
		size := len(bucket)
		sizeRange := fmt.Sprintf("%d-%d", (size/10)*10, (size/10)*10+9)
		distribution[sizeRange]++
	}
	
	return distribution
}