import argparse
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "tests"

SUITES = [
    ("Loader", TEST_DIR / "test_loader.py"),
    ("Transformers", TEST_DIR / "test_transformers.py"),
    ("Pipeline", TEST_DIR / "test_pipeline.py"),
    ("Integration", TEST_DIR / "test_integration.py"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run all project test suites with pytest."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip integration tests for a faster test run.",
    )
    args, pytest_args = parser.parse_known_args()

    selected_suites = [
        (label, path)
        for label, path in SUITES
        if not (args.fast and label == "Integration")
    ]

    missing = [str(path) for _, path in selected_suites if not path.is_file()]
    if missing:
        print("The following test files could not be found:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("=" * 60)
    print("  PROJECT TEST SUITES")
    print("=" * 60)

    for label, path in SUITES:
        status = "SKIPPED (--fast)" if args.fast and label == "Integration" else "SELECTED"
        print(f"  {label:<15} {status}")

    print("=" * 60)

    pytest_paths = [str(path) for _, path in selected_suites]

    pytest_options = ["-ra"]
    if not any(arg in ("-v", "--verbose", "-q", "--quiet") for arg in pytest_args):
        pytest_options.append("-q")

    start = time.time()
    exit_code = pytest.main(pytest_options + pytest_args + pytest_paths)
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"  Finished in {elapsed:.1f}s")
    print("=" * 60)

    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
