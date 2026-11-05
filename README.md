# Machine-Learning-F26

- **Repository URL:** https://github.com/Auipp/Machine-Learning-F26
- **Milestone 1 Commit SHA:** `5f3560d`

_____

## Team Members

| **Member Name**              | **ID**  | **Github Username** |
| ---------------------------- | ------- | ------------------- |
| Amir Atef Abdullah           | 24P0310 | Auipp               |
| Yahia Sameh Zakaria          | 23P0311 | Yahia-Sameh         |
| Ahmed Mohamed Abdullah       | 24P0129 | Ragab26             |
| Khaled Mohamed Nabil Mostafa | 2400987 | Xnfinity            |
| Ahmed Tamer Mohamed          | 24P0179 | 24P0179             |
| Eyad Wael Swelam             | 24P0137 | EyadUser            |

## CLI Usage

```bash
python main.py compare
python main.py compare --skip-knn
python main.py test --fast
python main.py train --task regression
python main.py train --task classification
python main.py train --task clustering
python main.py predict --task regression
```

## Models and preprocessing

The models use the shared pipeline in `src/preprocessing/`:

- `src/data/loader.py` loads `data/raw/H2.csv`, removes duplicate/empty bookings, and creates stratified train/validation/test splits.
- `src/preprocessing/pipeline.py` applies the task-specific filters, feature engineering, encoding, and scaling.
- `src/model_data.py` connects the models to that pipeline.

Each saved model in `artifacts/` contains both the fitted preprocessor and the estimator, so prediction data is transformed exactly like the training data.

Run a model directly, for example:

```bash
python src/Regression/linear_regression.py
```

The milestone-1 experiments in `AT_models/` use the same loader and preprocessing pipeline. The `extended` experiments additionally opt into the extra judgment columns:

```bash
python AT_models/at_baseline_regression.py
python AT_models/at_extended_regression.py
python AT_models/at_logistic_regression_base.py
python AT_models/at_logistic_regression_extended.py
python AT_models/at_kmeans_base.py 5
bash AT_models/sweep.sh
```

Or from Python:

```python
from src.Regression.linear_regression import (
    train_base_Linear_Regression,
    predict_with_Linear_Regression,
)

train_base_Linear_Regression("A")
predict_with_Linear_Regression("A", "test")
```

## How much data is used for training?

The models read the whole `data/raw/H2.csv` file, but they do **not** train on all of it:

| Stage | Rows |
| --- | ---: |
| Raw `H2.csv` | 79,330 |
| After collapsing exact duplicates and removing zero-guest / zero-night bookings | 53,043 |
| Training split (70%) | 37,129 |
| Validation split (15%) | 7,957 |
| Test split (15%) | 7,957 |

Regression and clustering additionally remove ADR values outside `[1, 510]`, which leaves `36,616 / 7,862 / 7,841` train/validation/test rows. Classification keeps all 53,043 cleaned rows.

Models are fitted on the training split only; validation and test rows are held out. Cross-validation also fits on folds of the training split only. Duplicate rows are collapsed by default (`duplicate_weight=0`), so each unique booking counts once; pass `duplicate_weight=100` to give every duplicate copy its original weight.

## Comparing models

Run the shared comparison harness:

```bash
python compare_models.py
python compare_models.py --skip-knn
```

It evaluates every model on the same splits and writes:

- `outputs/model_comparison.csv`
- `outputs/model_comparison.md`
- `outputs/model_comparison.png`

The comparison covers classification (dummy, logistic regression base/extended, KNN), regression (dummy, linear regression base/extended, SGD), and clustering (KMeans for `k=2..8`).
