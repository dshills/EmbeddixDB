# EmbeddixDB Global MCP Configuration

## Overview

This directory contains example configurations for setting up EmbeddixDB as a global MCP server with project-specific database isolation.

## Global Configuration (~/.claude.json)

Copy the `claude-config.json` to your home directory as `~/.claude.json`:

```json
{
  "mcpServers": {
    "embeddixdb": {
      "command": "/usr/local/bin/embeddix-mcp",
      "args": []
    }
  }
}
```

**Note**: Update the `command` path to match your actual installation path of the `embeddix-mcp` binary.

## How It Works

The MCP server automatically determines the database location using this priority order:

1. **EMBEDDIXDB_DATA_DIR** environment variable (explicit override)
2. **CLAUDE_WORKING_DIR** + `/.embeddixdb/` (Claude's working directory)
3. Current working directory + `/.embeddixdb/` (fallback)
4. `~/.embeddixdb/default/` (global fallback)

## Project Structure

When you run Claude Code in a project directory, EmbeddixDB automatically creates:

```
your-project/
├── .embeddixdb/                    # Safe, dedicated database directory
│   ├── embeddix.db                # BoltDB database file (if using bolt backend)
│   ├── collections/               # Collection metadata (BadgerDB mode)
│   ├── vectors/                   # Vector data files (BadgerDB mode)
│   ├── indexes/                   # Index files (BadgerDB mode)
│   └── *.vlog, MANIFEST, etc.     # BadgerDB internal files
├── src/
└── ...
```

### Database File Details

**BoltDB Backend** (recommended for most use cases):
- **Single file**: `.embeddixdb/embeddix.db`
- **File permissions**: `0600` (owner read/write only)
- **Self-contained**: All data in one file for easy backup/transfer

**BadgerDB Backend** (high-performance option):
- **Multiple files**: Various files directly in `.embeddixdb/`
- **Directory-based**: Uses entire directory as database
- **Higher performance**: Better for heavy write workloads

**Memory Backend** (testing/temporary):
- **No files created**: Data exists only in RAM
- **Data loss**: All data lost when process terminates
- **Safe for testing**: No file conflicts possible

## Environment Variable Override

For special cases, you can override the database location:

```bash
# Use a specific directory
export EMBEDDIXDB_DATA_DIR="/path/to/custom/database"
claude-code

# Use a shared team database
export EMBEDDIXDB_DATA_DIR="/shared/team/embeddixdb"
claude-code
```

## Safety Features

### File Conflict Prevention
- **Dedicated Directory**: Uses `.embeddixdb/` subdirectory to avoid conflicts
- **Safe Permissions**: 
  - Directories: `0755` (owner full, group/others read+execute)
  - BoltDB files: `0600` (owner read/write only)
- **Automatic Warnings**: Server warns if existing database files are found
- **Non-Destructive**: Won't overwrite files outside `.embeddixdb/` directory

### Directory Creation
- **Automatic**: Server creates `.embeddixdb/` directory if it doesn't exist
- **No User Action Required**: Zero manual setup needed
- **Parent Directory Respect**: Only creates the dedicated subdirectory

### Conflict Detection
The server automatically checks for and warns about:
- Existing `embeddix.db` files
- Other database files that might conflict
- Directory contents when using BadgerDB

### Environment Variable Safety
```bash
# Safe: Uses dedicated custom directory
export EMBEDDIXDB_DATA_DIR="/dedicated/embeddix/data"

# Careful: Could conflict with existing files
export EMBEDDIXDB_DATA_DIR="/path/with/existing/files"
```

## Benefits

- **Zero Configuration**: Works automatically in any project
- **Project Isolation**: Each project gets its own vector database
- **Global Setup**: Configure once in ~/.claude.json
- **Team Friendly**: Projects can share databases when needed
- **Portable**: Database travels with the project
- **Safe by Default**: Minimal risk of file conflicts

## Installation

1. Build the MCP server:
   ```bash
   go build -o embeddix-mcp ./cmd/embeddix-mcp
   ```

2. Install globally:
   ```bash
   sudo cp embeddix-mcp /usr/local/bin/
   ```

3. Copy the configuration:
   ```bash
   cp examples/claude-config.json ~/.claude.json
   ```

4. Start using Claude Code in any project - EmbeddixDB will automatically create project-specific databases!