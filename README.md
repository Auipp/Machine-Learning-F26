# Machine-Learning-F26
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

## Models and preprocessing

The models use the shared pipeline in `src/preprocessing/`:

- `src/data/loader.py` loads `data/raw/H2.csv`, removes duplicate/empty bookings, and creates stratified train/validation/test splits.
- `src/preprocessing/pipeline.py` applies the task-specific filters, feature engineering, encoding, and scaling.
- `src/model_data.py` connects the models to that pipeline.

Each saved model in `artifacts/` contains both the fitted preprocessor and the estimator, so prediction data is transformed exactly like the training data. The old CSVs in `data/*.csv` were produced by the previous preprocessing and are no longer used by the models. Artifacts saved before this change must be retrained.

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
