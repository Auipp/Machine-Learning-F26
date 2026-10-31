import sys
import time
import pathlib
import importlib
import traceback
import argparse

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

SUITES = [
    ("Loader",       "tests.test_loader"),
    ("Transformers", "tests.test_transformers"),
    ("Pipeline",     "tests.test_pipeline"),
    ("Integration",  "tests.test_integration"),
]

FAST_SKIP = {"Integration"}


def run_suite(label, module_path):
    mod = importlib.import_module(module_path)
    tests = sorted(
        (name, fn)
        for name, fn in vars(mod).items()
        if name.startswith("test_") and callable(fn)
    )
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            print(f"  FAIL  {name}")
            print(f"        {tb}")
            failed += 1
    return passed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    total_pass = total_fail = 0
    t_start = time.time()

    for label, module_path in SUITES:
        if args.fast and label in FAST_SKIP:
            print(f"\n{'─'*60}")
            print(f"  [{label}]  SKIPPED (--fast)")
            continue
        print(f"\n{'─'*60}")
        print(f"  [{label}]")
        print(f"{'─'*60}")
        p, f = run_suite(label, module_path)
        total_pass += p
        total_fail += f
        print(f"  → {p} passed, {f} failed")

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  TOTAL  {total_pass} passed, {total_fail} failed  ({elapsed:.1f}s)")
    print(f"{'═'*60}")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
