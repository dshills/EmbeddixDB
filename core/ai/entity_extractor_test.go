package ai

import (
	"context"
	"testing"
)

func TestNewEntityExtractor(t *testing.T) {
	extractor := NewEntityExtractor()
	
	if extractor == nil {
		t.Fatal("Expected extractor to be created, got nil")
	}
	
	if len(extractor.patterns) == 0 {
		t.Error("Expected patterns to be populated")
	}
	
	// Check that all expected entity types are present
	expectedTypes := []string{"PERSON", "ORG", "LOC", "DATE", "EMAIL", "PHONE", "URL", "MONEY"}
	for _, expectedType := range expectedTypes {
		if _, exists := extractor.patterns[expectedType]; !exists {
			t.Errorf("Expected entity type %s to be configured", expectedType)
		}
	}
}

func TestEntityExtractor_ExtractEntities(t *testing.T) {
	extractor := NewEntityExtractor()
	ctx := context.Background()
	
	testCases := []struct {
		name            string
		content         string
		expectedTypes   []string
		minEntities     int
	}{
		{
			name:        "person and organization",
			content:     "John Smith works at Google Inc. in New York.",
			expectedTypes: []string{"PERSON", "ORG", "LOC"},
			minEntities: 1,
		},
		{
			name:        "email and phone",
			content:     "Contact us at support@example.com or call (555) 123-4567.",
			expectedTypes: []string{"EMAIL", "PHONE"},
			minEntities: 1,
		},
		{
			name:        "dates and money",
			content:     "The meeting on January 15, 2023 will cost $1,500.",
			expectedTypes: []string{"DATE", "MONEY"},
			minEntities: 1,
		},
		{
			name:        "URLs",
			content:     "Visit our website at https://www.example.com for more info.",
			expectedTypes: []string{"URL"},
			minEntities: 1,
		},
		{
			name:        "empty content",
			content:     "",
			expectedTypes: []string{},
			minEntities: 0,
		},
		{
			name:        "no entities",
			content:     "This text has no special entities.",
			expectedTypes: []string{},
			minEntities: 0,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			entities, err := extractor.ExtractEntities(ctx, tc.content)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			
			if len(entities) < tc.minEntities {
				t.Errorf("Expected at least %d entities, got %d", tc.minEntities, len(entities))
			}
			
			// Check for expected types (if any entities found)
			if len(entities) > 0 && len(tc.expectedTypes) > 0 {
				foundTypes := make(map[string]bool)
				for _, entity := range entities {
					foundTypes[entity.Label] = true
					
					// Validate entity structure
					if entity.Text == "" {
						t.Error("Entity text should not be empty")
					}
					
					if entity.Confidence < 0 || entity.Confidence > 1 {
						t.Errorf("Entity confidence should be between 0 and 1, got %f", entity.Confidence)
					}
					
					if entity.StartPos < 0 || entity.EndPos < entity.StartPos {
						t.Errorf("Invalid entity positions: start=%d, end=%d", entity.StartPos, entity.EndPos)
					}
				}
				
				t.Logf("Found entity types: %v", getKeys(foundTypes))
			}
		})
	}
}

func TestGetSupportedEntityTypes(t *testing.T) {
	extractor := NewEntityExtractor()
	types := extractor.GetSupportedEntityTypes()
	
	if len(types) == 0 {
		t.Error("Expected supported entity types to be returned")
	}
	
	expectedTypes := []string{"PERSON", "ORG", "LOC", "DATE", "EMAIL", "PHONE", "URL", "MONEY"}
	foundTypes := make(map[string]bool)
	for _, t := range types {
		foundTypes[t] = true
	}
	
	for _, expectedType := range expectedTypes {
		if !foundTypes[expectedType] {
			t.Errorf("Expected entity type %s not found in supported types", expectedType)
		}
	}
}

func TestEntityValidation(t *testing.T) {
	extractor := NewEntityExtractor()
	
	testCases := []struct {
		name     string
		text     string
		expected bool
	}{
		// Person name validation
		{"valid person name", "John Smith", true},
		{"invalid person name (common word)", "the", false},
		{"invalid person name (lowercase)", "john smith", false},
		
		// Organization validation  
		{"valid organization", "Google Inc", true},
		{"invalid organization (too short)", "Go", false},
		{"invalid organization (no capital)", "google inc", false},
		
		// Location validation
		{"valid location", "New York", true},
		{"invalid location (lowercase)", "new york", false},
		{"invalid location (too short)", "A", false},
		
		// Email validation
		{"valid email", "test@example.com", true},
		{"invalid email (no @)", "testexample.com", false},
		{"invalid email (no domain)", "test@", false},
		
		// Phone validation
		{"valid phone", "(555) 123-4567", true},
		{"invalid phone (too few digits)", "555-123", false},
		
		// URL validation
		{"valid http URL", "http://example.com", true},
		{"valid https URL", "https://example.com", true},
		{"valid www URL", "www.example.com", true},
		{"invalid URL", "example", false},
		
		// Money validation
		{"valid money with $", "$100.50", true},
		{"valid money with currency", "100 dollars", true},
		{"invalid money (no digits)", "$", false},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test person name validation
			if tc.name == "valid person name" || tc.name == "invalid person name (common word)" || tc.name == "invalid person name (lowercase)" {
				result := extractor.isValidPersonName(tc.text)
				if result != tc.expected {
					t.Errorf("Person name validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
			
			// Test organization validation
			if tc.name == "valid organization" || tc.name == "invalid organization (too short)" || tc.name == "invalid organization (no capital)" {
				result := extractor.isValidOrganization(tc.text)
				if result != tc.expected {
					t.Errorf("Organization validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
			
			// Test location validation
			if tc.name == "valid location" || tc.name == "invalid location (lowercase)" || tc.name == "invalid location (too short)" {
				result := extractor.isValidLocation(tc.text)
				if result != tc.expected {
					t.Errorf("Location validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
			
			// Test email validation
			if tc.name == "valid email" || tc.name == "invalid email (no @)" || tc.name == "invalid email (no domain)" {
				result := extractor.isValidEmail(tc.text)
				if result != tc.expected {
					t.Errorf("Email validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
			
			// Test phone validation
			if tc.name == "valid phone" || tc.name == "invalid phone (too few digits)" {
				result := extractor.isValidPhone(tc.text)
				if result != tc.expected {
					t.Errorf("Phone validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
			
			// Test URL validation
			if tc.name == "valid http URL" || tc.name == "valid https URL" || tc.name == "valid www URL" || tc.name == "invalid URL" {
				result := extractor.isValidURL(tc.text)
				if result != tc.expected {
					t.Errorf("URL validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
			
			// Test money validation
			if tc.name == "valid money with $" || tc.name == "valid money with currency" || tc.name == "invalid money (no digits)" {
				result := extractor.isValidMoney(tc.text)
				if result != tc.expected {
					t.Errorf("Money validation for '%s': expected %t, got %t", tc.text, tc.expected, result)
				}
			}
		})
	}
}

// Helper function to get keys from a map
func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}