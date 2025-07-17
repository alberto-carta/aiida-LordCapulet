# LordCapulet Tests

This directory contains the test suite for the LordCapulet AiiDA plugin.

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Pytest configuration 
└── utils/                          # Tests for utility functions
    ├── __init__.py                 # Utils test package init
    └── test_rotation_matrices.py   # Tests for rotation matrix utilities
```

## Running Tests

### Simple Test Runner (No Dependencies)

Use the simple test runner that doesn't require pytest:

```bash
python run_tests.py
```

### With Pytest (Recommended)

If you have pytest installed, you can run the full test suite:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/utils/test_rotation_matrices.py -v
```

You can also use the pytest runner script:

```bash
python pytest_runner.py
```

## Test Coverage

### Rotation Matrices (`tests/utils/test_rotation_matrices.py`)

Tests for the `spherical_to_cubic_rotation` function:

- ✅ Function existence and basic functionality
- ✅ Input validation (invalid conventions and dimensions)
- ✅ Matrix mathematical properties (unitarity)
- ✅ Correct matrix elements for QE convention
- ✅ Simple density matrix rotation (Y₂⁰ → r²-3z²)
- ✅ Complex density matrix rotation with multiple orbitals
- ✅ Hermiticity preservation after rotation

## Adding New Tests

### For New Utility Functions

1. Create test files in the appropriate subdirectory under `tests/`
2. Follow the naming convention: `test_<module_name>.py`
3. Use the `TestClassName` pattern for test classes
4. Make tests compatible with both pytest and standalone execution

### Example Test Structure

```python
"""Tests for new_module utilities."""

import numpy as np
from lordcapulet.utils.new_module import new_function

class TestNewFunction:
    """Test suite for new_function."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = new_function()
        assert result is not None
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Add your tests here
        pass

if __name__ == "__main__":
    # Standalone execution code
    pass
```

## Test Guidelines

1. **Test Independence**: Each test should be independent and not rely on the state from other tests
2. **Error Testing**: Always test both success and failure cases
3. **Mathematical Properties**: For numerical functions, test mathematical properties (unitarity, Hermiticity, etc.)
4. **Edge Cases**: Test boundary conditions and edge cases
5. **Documentation**: Add clear docstrings explaining what each test verifies
6. **Reproducibility**: Use fixed random seeds when testing with random data

## Future Test Areas

Areas that could benefit from additional testing:

- **Calculations**: Tests for `ConstrainedPWCalculation`
- **Workflows**: Tests for `AFMScanWorkChain`, `ConstrainedScanWorkChain`, `GlobalConstrainedSearchWorkChain`
- **Functions**: Tests for `aiida_propose_occ_matrices_from_results`
- **Integration**: End-to-end workflow tests
- **Performance**: Benchmarking for large systems
