# Tests

For testing, you need to install `pytest` and `pytest-cov` packages.

To run a specific test

```bash
    pytest -v ./tests/test_specific_file.py
```

To run all tests

```bash
    pytest -v tests/
```

To get a test coverage
```bash
    pytest --cov=src/genomic_benchmarks/ tests/ 
```
