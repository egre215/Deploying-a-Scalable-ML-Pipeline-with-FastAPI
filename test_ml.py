import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# TODO: implement the first test. Change the function name and input as needed
def test_fit():
    """
    Test to see if the model can fit the data without errors.
    """
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)

    assert model.coef_ is not None
    


# TODO: implement the second test. Change the function name and input as needed
def test_predict():
    """
    If ML function "predict" returns the expected type numpy array.
    """
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    assert isinstance(y_pred, np.ndarray)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    If the ML model uses the expected algorithm, LogisticRegression.
    """
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
