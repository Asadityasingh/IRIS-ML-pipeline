# tests/test_model_evaluation.py
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class TestModelEvaluation:
    
    @pytest.fixture
    def model(self):
        """Load trained model"""
        model_path = 'models/iris_model.pkl'
        if not os.path.exists(model_path):
            pytest.skip("Model file not found. Run training first.")
        return joblib.load(model_path)
    
    @pytest.fixture
    def test_data(self):
        """Load test data"""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_test, y_test
    
    def test_model_accuracy(self, model, test_data):
        """Test if model achieves minimum accuracy"""
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy >= 0.85, f"Model accuracy {accuracy:.4f} is below threshold 0.85"
    
    def test_model_predictions_shape(self, model, test_data):
        """Test if predictions have correct shape"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test), "Predictions length mismatch"
    
    def test_model_predictions_range(self, model, test_data):
        """Test if predictions are within valid range"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        assert set(predictions).issubset({0, 1, 2}), "Predictions should be in {0, 1, 2}"
    
    def test_model_proba_output(self, model, test_data):
        """Test if probability predictions are valid"""
        X_test, y_test = test_data
        probas = model.predict_proba(X_test)
        assert probas.shape == (len(X_test), 3), "Probability shape mismatch"
        assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities should sum to 1"
        assert (probas >= 0).all() and (probas <= 1).all(), "Probabilities should be in [0, 1]"
    
    def test_model_deterministic(self, model, test_data):
        """Test if model predictions are deterministic"""
        X_test, y_test = test_data
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        assert np.array_equal(pred1, pred2), "Model should be deterministic"
    
    def test_single_sample_prediction(self, model):
        """Test prediction on single sample"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(sample)
        assert len(prediction) == 1, "Single sample should return single prediction"
        assert prediction[0] in [0, 1, 2], "Prediction should be valid class"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
