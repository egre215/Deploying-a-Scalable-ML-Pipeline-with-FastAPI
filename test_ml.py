import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from ml.model import train_model, inference, compute_model_metrics, train_test_split
from train_model import cat_features, process_data

@pytest.fixture(scope="module")
def data():
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train

def test_model(data):
    X_train, y_train = data
    model = train_model(X_train, y_train)
    
    assert model.coef_ is not None
    assert isinstance(model, LogisticRegression)

def test_inference(data):
    X_train, y_train = data
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_train)
    
    assert isinstance(y_pred, np.ndarray)

def test_compute_model_metrics(data):
    X_train, y_train = data
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y=y_train, preds=y_pred)
    
    assert precision > 0
    assert recall > 0
    assert fbeta > 0
