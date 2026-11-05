import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_compare(args):
    import compare_models

    sys.argv = [sys.argv[0]]
    if args.output_dir:
        sys.argv.extend(["--output-dir", str(args.output_dir)])
    if args.skip_knn:
        sys.argv.append("--skip-knn")
    return compare_models.main()


def run_test(args):
    from tests import run_all

    sys.argv = [sys.argv[0]]
    if args.fast:
        sys.argv.append("--fast")
    return run_all.main()


def run_train(args):
    if args.task == "regression":
        from src.Regression.linear_regression import (
            train_base_Linear_Regression,
        )

        train_base_Linear_Regression(args.name)
    elif args.task == "classification":
        from src.Classification.KNN import train_base_KNN

        train_base_KNN(args.name)
    elif args.task == "clustering":
        from src.Clustering.Kmeans import train_base_KMeans

        train_base_KMeans(args.name)


def run_predict(args):
    if args.task == "regression":
        from src.Regression.linear_regression import (
            predict_with_Linear_Regression,
        )

        predict_with_Linear_Regression(args.name, args.data)
    elif args.task == "classification":
        from src.Classification.KNN import predict_with_KNN

        predict_with_KNN(args.name, args.data)
    elif args.task == "clustering":
        from src.Clustering.Kmeans import predict_with_KMeans

        predict_with_KMeans(args.name, args.data)


def main():
    parser = argparse.ArgumentParser(
        description="Hotel Booking Demand Machine Learning System"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare", help="Run model comparison across all tasks"
    )
    compare_parser.add_argument("--output-dir", type=Path, default=None)
    compare_parser.add_argument("--skip-knn", action="store_true")
    compare_parser.set_defaults(func=run_compare)

    test_parser = subparsers.add_parser("test", help="Run project test suites")
    test_parser.add_argument("--fast", action="store_true")
    test_parser.set_defaults(func=run_test)

    train_parser = subparsers.add_parser("train", help="Train baseline models")
    train_parser.add_argument(
        "--task",
        choices=["regression", "classification", "clustering"],
        required=True,
    )
    train_parser.add_argument("--name", default="baseline")
    train_parser.set_defaults(func=run_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Generate predictions using saved artifacts"
    )
    predict_parser.add_argument(
        "--task",
        choices=["regression", "classification", "clustering"],
        required=True,
    )
    predict_parser.add_argument("--name", default="baseline")
    predict_parser.add_argument("--data", default="test")
    predict_parser.set_defaults(func=run_predict)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
