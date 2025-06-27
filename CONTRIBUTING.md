# Contributing to EmbeddixDB

Thank you for your interest in contributing to EmbeddixDB! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct: be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Issues

- Check if the issue already exists in our [issue tracker](https://github.com/dshills/EmbeddixDB/issues)
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Share relevant logs or error messages
- Mention your environment (OS, Go version, etc.)

### Suggesting Features

- Open a discussion in [GitHub Discussions](https://github.com/dshills/EmbeddixDB/discussions)
- Describe the feature and its use case
- Explain how it benefits other users
- Be open to feedback and alternative approaches

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding standards**:
   - Run `go fmt ./...` before committing
   - Ensure `go vet ./...` passes
   - Add tests for new functionality
   - Update documentation as needed
3. **Write clear commit messages**:
   - Use the present tense ("Add feature" not "Added feature")
   - Reference issues and pull requests when relevant
4. **Submit a pull request**:
   - Provide a clear description of changes
   - Link to any related issues
   - Ensure all tests pass
   - Be responsive to feedback

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/EmbeddixDB.git
cd EmbeddixDB

# Add upstream remote
git remote add upstream https://github.com/dshills/EmbeddixDB.git

# Install dependencies
go mod download

# Run tests
make test

# Run benchmarks
make benchmark
```

## Testing

- Write unit tests for new functionality
- Ensure existing tests pass: `go test ./...`
- Add integration tests for API changes
- Include benchmarks for performance-critical code

## Documentation

- Update README.md for user-facing changes
- Add/update code comments
- Document new configuration options
- Include examples for new features

## Questions?

Feel free to open a discussion or reach out to the maintainers. We're here to help!