#!/usr/bin/env python3
"""
Verification script for code review fixes.

This script verifies that the following issues have been fixed:
1. DataProto non_tensor_batch now uses NumPy arrays (not lists)
2. CUDA tensor slicing uses .item() to convert scalars before indexing
"""

import ast
import sys


def check_item_calls_in_peer_review():
    """Check that .item() is called on tensor scalars before using as indices."""
    print("="*60)
    print("Checking CUDA tensor fixes in peer_review.py")
    print("="*60)

    file_path = "/home/user/self-questioning-lm/verl/workers/reward_manager/peer_review.py"

    with open(file_path, 'r') as f:
        content = f.read()

    # Look for the pattern: .sum().item()
    # This indicates we're converting tensor scalars to Python scalars
    item_call_count = content.count('.sum().item()')

    print(f"\nFound {item_call_count} instances of '.sum().item()' pattern")

    # We should have 3 instances (lines 352, 383, 412)
    expected_count = 3

    if item_call_count >= expected_count:
        print(f"✓ PASS: Found {item_call_count} >= {expected_count} .item() calls")
        print("  This fixes the CUDA tensor slicing issue")
        return True
    else:
        print(f"✗ FAIL: Expected at least {expected_count} .item() calls, found {item_call_count}")
        return False


def check_numpy_arrays_in_tests():
    """Check that test file uses NumPy arrays for non_tensor_batch."""
    print("\n" + "="*60)
    print("Checking NumPy array usage in test_peer_review_voting.py")
    print("="*60)

    file_path = "/home/user/self-questioning-lm/tests/test_peer_review_voting.py"

    with open(file_path, 'r') as f:
        content = f.read()

    # Check for numpy import
    has_numpy_import = 'import numpy as np' in content

    # Check for np.array usage in non_tensor_batch
    has_numpy_arrays = 'np.array([item[\'uid\']' in content
    has_dtype_object = 'dtype=object' in content

    print(f"\n✓ NumPy imported: {has_numpy_import}")
    print(f"✓ np.array() used for uid: {has_numpy_arrays}")
    print(f"✓ dtype=object specified: {has_dtype_object}")

    if has_numpy_import and has_numpy_arrays and has_dtype_object:
        print("\n✓ PASS: Test file properly uses NumPy arrays")
        print("  This fixes the DataProto consistency check issue")
        return True
    else:
        print("\n✗ FAIL: Test file doesn't properly use NumPy arrays")
        return False


def check_specific_fixes():
    """Check specific line numbers mentioned in the review."""
    print("\n" + "="*60)
    print("Verifying specific fixes mentioned in code review")
    print("="*60)

    file_path = "/home/user/self-questioning-lm/verl/workers/reward_manager/peer_review.py"

    with open(file_path, 'r') as f:
        lines = f.readlines()

    fixes_verified = []

    # Check line 352 (was 353 before): valid_response_length = ...sum().item()
    line_352 = lines[351].strip()  # 0-indexed
    if '.sum().item()' in line_352:
        print(f"✓ Line 352: {line_352[:60]}...")
        fixes_verified.append(True)
    else:
        print(f"✗ Line 352: Missing .item() call")
        fixes_verified.append(False)

    # Check line 383 (was 384): valid_response_length = ...sum().item()
    line_383 = lines[382].strip()
    if '.sum().item()' in line_383:
        print(f"✓ Line 383: {line_383[:60]}...")
        fixes_verified.append(True)
    else:
        print(f"✗ Line 383: Missing .item() call")
        fixes_verified.append(False)

    # Check line 412 (was 415): valid_response_length = ...sum().item()
    line_412 = lines[411].strip()
    if '.sum().item()' in line_412:
        print(f"✓ Line 412: {line_412[:60]}...")
        fixes_verified.append(True)
    else:
        print(f"✗ Line 412: Missing .item() call")
        fixes_verified.append(False)

    if all(fixes_verified):
        print("\n✓ PASS: All specific fixes verified")
        return True
    else:
        print("\n✗ FAIL: Some fixes missing")
        return False


def main():
    """Run all verification checks."""
    print("CODE REVIEW FIXES - VERIFICATION")
    print("="*60)
    print("\nThis script verifies the following fixes:")
    print("1. Issue: DataProto requires NumPy arrays, not lists")
    print("   Fix: Updated test to use np.array() with dtype=object")
    print("\n2. Issue: CUDA tensor scalars can't be used as slice indices")
    print("   Fix: Call .item() to convert to Python scalar before indexing")
    print()

    results = []

    # Run checks
    results.append(("CUDA tensor fixes", check_item_calls_in_peer_review()))
    results.append(("NumPy array usage", check_numpy_arrays_in_tests()))
    results.append(("Specific line fixes", check_specific_fixes()))

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} checks passed")
    print("="*60)

    if passed_count == total_count:
        print("\n🎉 All code review fixes have been verified!")
        print("\nFixed issues:")
        print("  1. ✓ DataProto non_tensor_batch now uses NumPy arrays")
        print("  2. ✓ CUDA tensor scalars converted to Python ints before slicing")
        print("\nThe peer-review voting system should now work correctly with:")
        print("  - DataProto consistency checks")
        print("  - CUDA tensors in rollout workers")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} verification(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
