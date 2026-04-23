import argparse

from src.config import DATASETS
from src.training.trainer import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train MediSense AI models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to train on (default: all)",
    )
    parser.add_argument(
        "--no-dl",
        action="store_true",
        help="Skip deep learning base learner",
    )
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    results = {}
    for name in datasets:
        results[name] = train_pipeline(name, use_dl=not args.no_dl)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        print(
            f"\n{name}: ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}, F1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
