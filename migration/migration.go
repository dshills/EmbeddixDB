package migration

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/dshills/EmbeddixDB/core"
)

// MigrationVersion represents a database schema version
type MigrationVersion int64

// Migration represents a single database migration
type Migration struct {
	Version     MigrationVersion `json:"version"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	UpFunc      func(ctx context.Context, db core.Persistence) error
	DownFunc    func(ctx context.Context, db core.Persistence) error
	CreatedAt   time.Time `json:"created_at"`
}

// MigrationRecord tracks applied migrations
type MigrationRecord struct {
	Version   MigrationVersion `json:"version"`
	Name      string           `json:"name"`
	AppliedAt time.Time        `json:"applied_at"`
	Checksum  string           `json:"checksum"`
}

// Migrator manages database migrations
type Migrator struct {
	persistence core.Persistence
	migrations  map[MigrationVersion]*Migration
	versions    []MigrationVersion
}

// NewMigrator creates a new database migrator
func NewMigrator(persistence core.Persistence) *Migrator {
	return &Migrator{
		persistence: persistence,
		migrations:  make(map[MigrationVersion]*Migration),
		versions:    make([]MigrationVersion, 0),
	}
}

// AddMigration adds a migration to the migrator
func (m *Migrator) AddMigration(migration *Migration) {
	m.migrations[migration.Version] = migration
	m.versions = append(m.versions, migration.Version)
	sort.Slice(m.versions, func(i, j int) bool {
		return m.versions[i] < m.versions[j]
	})
}

// GetCurrentVersion returns the current database version
func (m *Migrator) GetCurrentVersion(ctx context.Context) (MigrationVersion, error) {
	records, err := m.getAppliedMigrations(ctx)
	if err != nil {
		return 0, err
	}

	if len(records) == 0 {
		return 0, nil
	}

	// Return the highest version
	var maxVersion MigrationVersion
	for _, record := range records {
		if record.Version > maxVersion {
			maxVersion = record.Version
		}
	}

	return maxVersion, nil
}

// GetTargetVersion returns the latest available migration version
func (m *Migrator) GetTargetVersion() MigrationVersion {
	if len(m.versions) == 0 {
		return 0
	}
	return m.versions[len(m.versions)-1]
}

// Migrate migrates the database to the target version
func (m *Migrator) Migrate(ctx context.Context, targetVersion MigrationVersion) error {
	currentVersion, err := m.GetCurrentVersion(ctx)
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	if currentVersion == targetVersion {
		return nil // Already at target version
	}

	if currentVersion < targetVersion {
		return m.migrateUp(ctx, currentVersion, targetVersion)
	} else {
		return m.migrateDown(ctx, currentVersion, targetVersion)
	}
}

// MigrateUp migrates the database to the latest version
func (m *Migrator) MigrateUp(ctx context.Context) error {
	targetVersion := m.GetTargetVersion()
	return m.Migrate(ctx, targetVersion)
}

// migrateUp applies migrations from current to target version
func (m *Migrator) migrateUp(ctx context.Context, currentVersion, targetVersion MigrationVersion) error {
	for _, version := range m.versions {
		if version <= currentVersion {
			continue
		}
		if version > targetVersion {
			break
		}

		migration, exists := m.migrations[version]
		if !exists {
			return fmt.Errorf("migration %d not found", version)
		}

		if migration.UpFunc == nil {
			return fmt.Errorf("migration %d has no up function", version)
		}

		fmt.Printf("Applying migration %d: %s\n", version, migration.Name)

		if err := migration.UpFunc(ctx, m.persistence); err != nil {
			return fmt.Errorf("failed to apply migration %d: %w", version, err)
		}

		if err := m.recordMigration(ctx, migration); err != nil {
			return fmt.Errorf("failed to record migration %d: %w", version, err)
		}

		fmt.Printf("Migration %d applied successfully\n", version)
	}

	return nil
}

// migrateDown rolls back migrations from current to target version
func (m *Migrator) migrateDown(ctx context.Context, currentVersion, targetVersion MigrationVersion) error {
	// Sort versions in descending order for rollback
	versions := make([]MigrationVersion, len(m.versions))
	copy(versions, m.versions)
	sort.Slice(versions, func(i, j int) bool {
		return versions[i] > versions[j]
	})

	for _, version := range versions {
		if version > currentVersion {
			continue
		}
		if version <= targetVersion {
			break
		}

		migration, exists := m.migrations[version]
		if !exists {
			return fmt.Errorf("migration %d not found", version)
		}

		if migration.DownFunc == nil {
			return fmt.Errorf("migration %d has no down function", version)
		}

		fmt.Printf("Rolling back migration %d: %s\n", version, migration.Name)

		if err := migration.DownFunc(ctx, m.persistence); err != nil {
			return fmt.Errorf("failed to rollback migration %d: %w", version, err)
		}

		if err := m.removeMigrationRecord(ctx, version); err != nil {
			return fmt.Errorf("failed to remove migration record %d: %w", version, err)
		}

		fmt.Printf("Migration %d rolled back successfully\n", version)
	}

	return nil
}

// getAppliedMigrations retrieves the list of applied migrations
func (m *Migrator) getAppliedMigrations(ctx context.Context) ([]MigrationRecord, error) {
	// Try to load migration records from a special collection
	data, err := m.persistence.LoadIndexState(ctx, "__migrations__")
	if err != nil {
		// If no migration records exist, return empty slice
		return []MigrationRecord{}, nil
	}

	var records []MigrationRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("failed to unmarshal migration records: %w", err)
	}

	return records, nil
}

// recordMigration records that a migration has been applied
func (m *Migrator) recordMigration(ctx context.Context, migration *Migration) error {
	records, err := m.getAppliedMigrations(ctx)
	if err != nil {
		return err
	}

	// Check if migration is already recorded
	for _, record := range records {
		if record.Version == migration.Version {
			return nil // Already recorded
		}
	}

	// Add new record
	newRecord := MigrationRecord{
		Version:   migration.Version,
		Name:      migration.Name,
		AppliedAt: time.Now(),
		Checksum:  m.calculateChecksum(migration),
	}

	records = append(records, newRecord)

	// Save updated records
	data, err := json.Marshal(records)
	if err != nil {
		return fmt.Errorf("failed to marshal migration records: %w", err)
	}

	return m.persistence.SaveIndexState(ctx, "__migrations__", data)
}

// removeMigrationRecord removes a migration record
func (m *Migrator) removeMigrationRecord(ctx context.Context, version MigrationVersion) error {
	records, err := m.getAppliedMigrations(ctx)
	if err != nil {
		return err
	}

	// Filter out the version to remove
	filteredRecords := make([]MigrationRecord, 0)
	for _, record := range records {
		if record.Version != version {
			filteredRecords = append(filteredRecords, record)
		}
	}

	// Save updated records
	data, err := json.Marshal(filteredRecords)
	if err != nil {
		return fmt.Errorf("failed to marshal migration records: %w", err)
	}

	return m.persistence.SaveIndexState(ctx, "__migrations__", data)
}

// calculateChecksum calculates a checksum for migration integrity
func (m *Migrator) calculateChecksum(migration *Migration) string {
	data := fmt.Sprintf("%d:%s:%s", migration.Version, migration.Name, migration.Description)
	hash := uint32(0)
	for _, b := range []byte(data) {
		hash = hash*31 + uint32(b)
	}
	return fmt.Sprintf("%x", hash)
}

// ListMigrations returns information about all available migrations
func (m *Migrator) ListMigrations(ctx context.Context) ([]MigrationInfo, error) {
	appliedRecords, err := m.getAppliedMigrations(ctx)
	if err != nil {
		return nil, err
	}

	appliedMap := make(map[MigrationVersion]MigrationRecord)
	for _, record := range appliedRecords {
		appliedMap[record.Version] = record
	}

	var migrations []MigrationInfo
	for _, version := range m.versions {
		migration := m.migrations[version]
		info := MigrationInfo{
			Version:     migration.Version,
			Name:        migration.Name,
			Description: migration.Description,
			CreatedAt:   migration.CreatedAt,
		}

		if record, applied := appliedMap[version]; applied {
			info.Applied = true
			info.AppliedAt = &record.AppliedAt
		}

		migrations = append(migrations, info)
	}

	return migrations, nil
}

// MigrationInfo provides information about a migration
type MigrationInfo struct {
	Version     MigrationVersion `json:"version"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Applied     bool             `json:"applied"`
	CreatedAt   time.Time        `json:"created_at"`
	AppliedAt   *time.Time       `json:"applied_at,omitempty"`
}

// ValidateMigrations checks migration integrity
func (m *Migrator) ValidateMigrations(ctx context.Context) error {
	records, err := m.getAppliedMigrations(ctx)
	if err != nil {
		return err
	}

	for _, record := range records {
		migration, exists := m.migrations[record.Version]
		if !exists {
			return fmt.Errorf("applied migration %d not found in migration list", record.Version)
		}

		expectedChecksum := m.calculateChecksum(migration)
		if record.Checksum != expectedChecksum {
			return fmt.Errorf("migration %d checksum mismatch: expected %s, got %s",
				record.Version, expectedChecksum, record.Checksum)
		}
	}

	return nil
}
