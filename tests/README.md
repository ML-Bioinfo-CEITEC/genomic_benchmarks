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

To deselect/select tests by markers use the -m option, it supports the boolean `not`,
f.e. to deselect tests marked as `slow` use

```bash
    pytest -v tests/ -m "not slow"
```

To get a test coverage
```bash
    pytest --cov=src/genomic_benchmarks/ tests/ 
```

## How to write tests

The structure of the test folder mirrors the package structure (note the prefix `test_`). Try to cover both the typical and the corner cases. Avoid slow tests (or at least tag them). Use `pytest-cov` to see which parts of the code are not yet tested. Tests are essential - be a test hero!
