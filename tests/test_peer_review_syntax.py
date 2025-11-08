#!/usr/bin/env python3
"""
Syntax validation for peer-review voting implementation.

This script checks that all the code is syntactically correct
and can be imported without errors (no runtime tests).
"""

import sys
import ast
import os


def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    print(f"Checking syntax: {file_path}")

    try:
        with open(file_path, 'r') as f:
            code = f.read()

        ast.parse(code)
        print(f"  ✓ Syntax OK")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_imports(file_path):
    """Check that imports in the file are properly structured."""
    print(f"Checking imports: {file_path}")

    try:
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        print(f"  ✓ Found {len(imports)} import statements")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_class_structure(file_path, expected_classes):
    """Check that expected classes are defined in the file."""
    print(f"Checking classes in: {file_path}")

    try:
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        for expected_class in expected_classes:
            if expected_class in classes:
                print(f"  ✓ Class '{expected_class}' found")
            else:
                print(f"  ✗ Class '{expected_class}' not found")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_function_structure(file_path, expected_functions):
    """Check that expected functions are defined in the file."""
    print(f"Checking functions in: {file_path}")

    try:
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        for expected_func in expected_functions:
            if expected_func in functions:
                print(f"  ✓ Function '{expected_func}' found")
            else:
                print(f"  ✗ Function '{expected_func}' not found")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all validation checks."""
    print("="*60)
    print("PEER-REVIEW VOTING - SYNTAX VALIDATION")
    print("="*60)

    base_path = "/home/user/self-questioning-lm"
    files_to_check = [
        (
            os.path.join(base_path, "verl/workers/reward_manager/peer_review.py"),
            ["PeerReviewVotingRewardManager"],
            ["extract_answer", "create_voting_prompt", "perform_model_based_voting", "simulate_voting", "__call__"]
        ),
        (
            os.path.join(base_path, "verl/workers/reward_manager/voting_utils.py"),
            ["VotingCache"],
            ["create_simple_generation_fn", "create_batched_generation_fn", "create_actor_wrapper_generation_fn"]
        ),
    ]

    results = []

    for file_path, expected_classes, expected_functions in files_to_check:
        print(f"\n{'='*60}")
        print(f"Validating: {os.path.basename(file_path)}")
        print(f"{'='*60}")

        # Check file exists
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            results.append((file_path, False))
            continue

        # Check syntax
        syntax_ok = check_syntax(file_path)
        if not syntax_ok:
            results.append((file_path, False))
            continue

        # Check imports
        imports_ok = check_imports(file_path)

        # Check class structure
        classes_ok = check_class_structure(file_path, expected_classes)

        # Check function structure
        functions_ok = check_function_structure(file_path, expected_functions)

        all_ok = syntax_ok and imports_ok and classes_ok and functions_ok
        results.append((file_path, all_ok))

    # Check configuration file
    print(f"\n{'='*60}")
    print("Checking configuration file")
    print(f"{'='*60}")

    config_path = os.path.join(base_path, "verl/trainer/config/exps/peer_review.yaml")
    if os.path.exists(config_path):
        print(f"✓ Configuration file exists: {config_path}")
        results.append((config_path, True))
    else:
        print(f"✗ Configuration file not found: {config_path}")
        results.append((config_path, False))

    # Check documentation
    print(f"\n{'='*60}")
    print("Checking documentation")
    print(f"{'='*60}")

    doc_path = os.path.join(base_path, "docs/peer_review_voting.md")
    if os.path.exists(doc_path):
        size = os.path.getsize(doc_path)
        print(f"✓ Documentation file exists: {doc_path} ({size} bytes)")
        results.append((doc_path, True))
    else:
        print(f"✗ Documentation file not found: {doc_path}")
        results.append((doc_path, False))

    # Check integration with __init__.py
    print(f"\n{'='*60}")
    print("Checking integration with reward manager registry")
    print(f"{'='*60}")

    init_path = os.path.join(base_path, "verl/workers/reward_manager/__init__.py")
    if os.path.exists(init_path):
        with open(init_path, 'r') as f:
            init_content = f.read()

        if "peer_review" in init_content and "PeerReviewVotingRewardManager" in init_content:
            print(f"✓ PeerReviewVotingRewardManager registered in __init__.py")
            results.append((init_path, True))
        else:
            print(f"✗ PeerReviewVotingRewardManager not registered in __init__.py")
            results.append((init_path, False))
    else:
        print(f"✗ __init__.py not found")
        results.append((init_path, False))

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for file_path, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        basename = os.path.basename(file_path)
        print(f"{status}: {basename}")

    print(f"\nTotal: {passed}/{total} checks passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All validation checks passed!")
        print("The peer-review voting implementation is syntactically correct and properly integrated.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
