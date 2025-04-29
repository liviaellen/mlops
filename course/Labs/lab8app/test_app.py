import pytest
from fastapi.testclient import TestClient
from app import app, WineFeatures
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch
import warnings

# Suppress Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

# Create test client
client = TestClient(app)

# Sample wine features for testing
X, y = load_wine(return_X_y=True)
sample_features = X[0].tolist()

# Create a mock model for testing
mock_model = RandomForestClassifier(n_estimators=100, random_state=42)
mock_model.fit(X, y)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Wine Classification API - Livia Ellen"}

def test_predict():
    # Test with valid input
    with patch('app.model', mock_model):
        response = client.post(
            "/predict",
            json={"features": sample_features}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probabilities"], list)
        assert len(data["probabilities"]) == 3  # 3 classes in wine dataset

def test_predict_invalid_input():
    # Test with invalid input (wrong number of features)
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]}  # Too few features
    )
    assert response.status_code == 422  # Changed from 400 to 422 for Pydantic validation error

def test_model_prediction():
    # Test model prediction directly
    prediction = mock_model.predict(np.array([sample_features]))
    assert prediction.shape == (1,)
    probabilities = mock_model.predict_proba(np.array([sample_features]))
    assert probabilities.shape == (1, 3)  # 3 classes in wine dataset

def test_wine_features_validation():
    # Test Pydantic model validation
    # Valid case
    valid_features = WineFeatures(features=sample_features)
    assert len(valid_features.features) == 13

    # Invalid case - too few features
    with pytest.raises(ValueError):
        WineFeatures(features=[1.0, 2.0])

    # Invalid case - too many features
    with pytest.raises(ValueError):
        WineFeatures(features=sample_features + [1.0])
