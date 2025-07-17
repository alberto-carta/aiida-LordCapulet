#!/usr/bin/env python3
"""
Simple test runner for LordCapulet utilities.

This script can be used to run tests without requiring pytest installation.
For more advanced testing features, install pytest and run: pytest tests/
"""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_rotation_matrix_tests():
    """Run the rotation matrix tests."""
    print("=" * 60)
    print("Running LordCapulet Rotation Matrix Tests")
    print("=" * 60)
    
    try:
        # Import and run the test class
        from tests.utils.test_rotation_matrices import TestSphericalToCubicRotation
        
        test_suite = TestSphericalToCubicRotation()
        
        print("\n🧪 Testing function existence...")
        test_suite.test_function_exists()
        print("   ✅ PASSED")
        
        print("\n🧪 Testing invalid inputs...")
        test_suite.test_invalid_convention()
        test_suite.test_invalid_dimension()
        print("   ✅ PASSED")
        
        print("\n🧪 Testing matrix properties...")
        test_suite.test_matrix_properties()
        print("   ✅ PASSED")
        
        print("\n🧪 Testing simple density matrix rotation...")
        test_suite.test_simple_density_matrix_rotation()
        print("   ✅ PASSED")
        
        print("\n🧪 Testing complex density matrix rotation...")
        test_suite.test_complex_density_matrix_rotation()
        print("   ✅ PASSED")
        
        print("\n🧪 Testing Hermiticity preservation...")
        test_suite.test_matrix_hermiticity_preservation()
        print("   ✅ PASSED")
        
        print("\n" + "=" * 60)
        print("🎉 All rotation matrix tests PASSED! 🎉")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner function."""
    print("LordCapulet Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run rotation matrix tests
    success &= run_rotation_matrix_tests()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED! 🎉")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
