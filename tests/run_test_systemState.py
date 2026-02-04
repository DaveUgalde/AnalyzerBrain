#!/usr/bin/env python3
"""
Test runner for SystemState tests.
"""

import pytest
import sys
import os

if __name__ == "__main__":
    # Add the parent directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Run tests
    exit_code = pytest.main(
        [
            "tests/unit/test_system_state.py",
            "tests/test_configuration.py",
            "-v",
            "--tb=short",
            "--cov=system_state",
            "--cov-report=term-missing",
            "--cov-report=html",
        ]
    )

    sys.exit(exit_code)
