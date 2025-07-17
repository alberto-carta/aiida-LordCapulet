"""
Pytest configuration and runner for LordCapulet tests.

Run with: python pytest_runner.py
Or if pytest is installed: pytest tests/
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

def run_with_pytest():
    """Run tests using pytest if available."""
    if not PYTEST_AVAILABLE:
        print("❌ pytest not available. Please install it with: pip install pytest")
        print("   Or use the simple test runner: python run_tests.py")
        return False
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    return exit_code == 0

if __name__ == "__main__":
    if run_with_pytest():
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed or pytest unavailable")
