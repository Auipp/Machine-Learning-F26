# Model comparison

All models are trained on the same training split and evaluated on
the same validation/test splits created by `src.data.loader`.
The preprocessing pipeline is fitted on the training split only.

## Classification

### Validation split

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | 0.6979 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3021 |
| Logistic Regression (base) | 0.7255 | 0.5774 | 0.3415 | 0.4292 | 0.7810 | 0.5582 |
| Logistic Regression (extended) | 0.7706 | 0.6667 | 0.4817 | 0.5593 | 0.8183 | 0.6475 |
| KNN (k=5) | 0.7324 | 0.5655 | 0.4938 | 0.5272 | 0.7602 | 0.5380 |

### Test split

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | 0.6980 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3020 |
| Logistic Regression (base) | 0.7219 | 0.5664 | 0.3371 | 0.4226 | 0.7777 | 0.5543 |
| Logistic Regression (extended) | 0.7691 | 0.6638 | 0.4773 | 0.5553 | 0.8191 | 0.6573 |
| KNN (k=5) | 0.7280 | 0.5570 | 0.4856 | 0.5189 | 0.7498 | 0.5294 |

## Regression

### Validation split

| model | mae | rmse | r2 |
| --- | --- | --- | --- |
| Dummy (mean) | 30.2283 | 39.1502 | -0.0000 |
| Linear Regression (base) | 17.4561 | 23.5551 | 0.6380 |
| Linear Regression (extended) | 17.4543 | 23.5525 | 0.6381 |
| SGD Regressor | 17.4809 | 23.6249 | 0.6359 |

### Test split

| model | mae | rmse | r2 |
| --- | --- | --- | --- |
| Dummy (mean) | 30.4453 | 38.9386 | -0.0007 |
| Linear Regression (base) | 17.9029 | 23.8531 | 0.6245 |
| Linear Regression (extended) | 17.8995 | 23.8498 | 0.6246 |
| SGD Regressor | 17.9120 | 23.8904 | 0.6233 |

## Clustering

### Validation split

| model | silhouette | davies_bouldin | min_cluster_size |
| --- | --- | --- | --- |
| KMeans (k=2) | 0.1997 | 2.0604 | 2942 |
| KMeans (k=3) | 0.1858 | 1.7292 | 1890 |
| KMeans (k=4) | 0.1973 | 1.5754 | 1201 |
| KMeans (k=5) | 0.2011 | 1.4737 | 919 |
| KMeans (k=6) | 0.1761 | 1.5053 | 319 |
| KMeans (k=7) | 0.1736 | 1.4380 | 122 |
| KMeans (k=8) | 0.1587 | 1.4787 | 100 |

### Test split

| model | silhouette | davies_bouldin | min_cluster_size |
| --- | --- | --- | --- |
| KMeans (k=2) | 0.2065 | 2.0300 | 2863 |
| KMeans (k=3) | 0.1957 | 1.6815 | 1812 |
| KMeans (k=4) | 0.2041 | 1.5805 | 1184 |
| KMeans (k=5) | 0.2052 | 1.4702 | 885 |
| KMeans (k=6) | 0.1846 | 1.5083 | 318 |
| KMeans (k=7) | 0.1754 | 1.4422 | 88 |
| KMeans (k=8) | 0.1622 | 1.4656 | 70 |

### Training split

| model | inertia |
| --- | --- |
| KMeans (k=2) | 99942.1089 |
| KMeans (k=3) | 85966.9656 |
| KMeans (k=4) | 76279.9497 |
| KMeans (k=5) | 68958.2473 |
| KMeans (k=6) | 64270.8793 |
| KMeans (k=7) | 60434.5325 |
| KMeans (k=8) | 57192.8727 |
