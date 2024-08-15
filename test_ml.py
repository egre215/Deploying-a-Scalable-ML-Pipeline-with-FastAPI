import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from ml.model import load_model

model = load_model(r"/home/miricow/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/model/model.pkl")

# TODO: implement the first test. Change the function name and input as needed
def test_fit():
    """
    Test to see if the model can fit the data without errors.
    """
    assert model.coef_ is not None
    


# TODO: implement the second test. Change the function name and input as needed
def test_2():
    """
    Testing to see if the model's coefs length is a positive value
    """
    model_size = len(model.coef_)
    assert model_size > 0
    


# TODO: implement the third test. Change the function name and input as needed
def test_LogisticRegression():
    """
    If the ML model uses the expected algorithm, LogisticRegression.
    """
    assert isinstance(model, LogisticRegression)
