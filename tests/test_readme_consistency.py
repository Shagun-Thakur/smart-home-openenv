"""
tests/test_readme_consistency.py

Unit test to verify that all numbers, tables, and claims in README.md
are strictly synchronized with CSV data in results/.
Base-install safe (runs via subprocess without training dependencies).
"""

import os
import sys
import subprocess
import pytest


def test_readme_numbers_match_results():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "check_readme_numbers.py")
    assert os.path.exists(script_path), f"Script {script_path} not found"

    # Execute check_readme_numbers.py using current Python interpreter
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, (
        f"check_readme_numbers.py failed with returncode {result.returncode}:\n"
        f"{result.stdout}\n{result.stderr}"
    )
