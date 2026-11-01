import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.resolve().parents[1] / "data"
artifact_dir = current_dir.resolve().parents[1] / "artifacts"

def train_KMeans(name, train_set):
    pass

def show_test_results(y_test, y_pred):
    pass

def show_cross_validation(linear_regress, X_train, y_train, cv):
    pass