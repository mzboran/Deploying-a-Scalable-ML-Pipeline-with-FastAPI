import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# load sample data
@pytest.fixture
def sample_data():
    """
    Returns a small pandas DataFrame mimicking the structure of the cleaned census dataset.
    This fixture is used to test model training and data processing functions.

    Returns
    -------
     pd.DataFrame
        A 2-row sample dataset with features and labels.

    """
    data = {
        'age': [25, 45],
        'workclass': ['Private', 'Self-emp-not-inc'],
        'education': ['Bachelors', 'HS-grad'],
        'marital_status': ['Never-married', 'Married-civ-spouse'],
        'occupation': ['Tech-support', 'Exec-managerial'],
        'relationship': ['Not-in-family', 'Husband'],
        'race': ['White', 'Black'],
        'sex': ['Male', 'Female'],
        'native_country': ['United-States', 'United-States'],
        'salary': ['<=50K', '>50K']
    }
    df = pd.DataFrame(data)
    return df


def test_train_model_type(sample_data):
    """
    Tests that the train_model function returns a RandomForestClassifier instance.
    Parameters
    ----------
    sample_data : pd.DataFrame
        Sample dataset used to train the model.

    Returns
    -------
    None
    """
    X, y, _, _ = process_data(sample_data, label='salary', training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_range():
    """
    Tests that compute_model_metrics returns precision, recall, and fbeta values within [0, 1].

    Returns
    -------
    None
    """
    # Simulate values
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_data_shape(sample_data):
    """
    Tests that process_data maintains row alignment between features and labels.

    Parameters
    ----------
    sample_data : pd.DataFrame
        Sample dataset used to verify preprocessing output shape.

    Returns
    -------
    None
    """
    X, y, _, _ = process_data(sample_data, label='salary', training=True)
    assert X.shape[0] == 2
    assert y.shape[0] == 2
