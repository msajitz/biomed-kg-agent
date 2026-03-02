# Test Suite Documentation

This document describes the test organization, conventions, and how to run tests for the biomed-kg-agent project.

## Quick Start

```bash
# Run all tests
poetry run pytest -v

# Run only unit tests (fast)
poetry run pytest -v -m unit

# Run with coverage
poetry run pytest --cov=biomed_kg_agent --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_entity_linking.py -v
```

## Test Organization

### Test Markers

Tests are organized using pytest markers:

- **`@pytest.mark.unit`**: Fast unit tests, heavily mocked, no external dependencies
- **`@pytest.mark.slow`**: Slow tests (>5s), typically model loading or large data processing

### Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── mocks/                   # Mock helpers
│   ├── biomedical_ner.py   # Mock NER models (prevents OOM)
│   └── ncbi.py             # Mock PubMed API responses
├── test_*.py               # Test modules (one per source module)
└── README.md               # This file
```

## Mocking

Biomedical NER models are mocked in tests to prevent OOM errors (three models totaling ~1.5GB). See `mocks/biomedical_ner.py` for mock helpers.

**Note:** Mocking patterns vary across tests (decorator, context manager, fixture-based) due to incremental development.

## Running Tests in CI

The test suite is designed to run in GitHub Actions. See `.github/workflows/` for CI configuration.

**CI considerations:**
- Models are mocked to avoid OOM
- Use `pytest -v -m unit` for fast feedback
- Use `pytest -v -m "not slow"` to skip slow tests
