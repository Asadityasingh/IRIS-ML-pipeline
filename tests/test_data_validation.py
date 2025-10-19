# tests/test_data_validation.py
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class TestDataValidation:
    
    @pytest.fixture
    def iris_data(self):
        """Load iris dataset for testing"""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        return X, y
    
    def test_data_shape(self, iris_data):
        """Test if data has correct shape"""
        X, y = iris_data
        assert X.shape[0] == 150, "Dataset should have 150 samples"
        assert X.shape[1] == 4, "Dataset should have 4 features"
        assert len(y) == 150, "Target should have 150 labels"
    
    def test_data_types(self, iris_data):
        """Test if data types are correct"""
        X, y = iris_data
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), \
            "All features should be numeric"
        assert np.issubdtype(y.dtype, np.integer), "Target should be integer type"
    
    def test_no_missing_values(self, iris_data):
        """Test if there are no missing values"""
        X, y = iris_data
        assert not X.isnull().any().any(), "Features should not contain missing values"
        assert not pd.isnull(y).any(), "Target should not contain missing values"
    
    def test_feature_ranges(self, iris_data):
        """Test if feature values are within expected ranges"""
        X, y = iris_data
        assert (X >= 0).all().all(), "All feature values should be non-negative"
        assert (X <= 10).all().all(), "All feature values should be reasonable"
    
    def test_target_classes(self, iris_data):
        """Test if target has correct classes"""
        X, y = iris_data
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3, "Should have 3 classes"
        assert set(unique_classes) == {0, 1, 2}, "Classes should be 0, 1, 2"
    
    def test_class_distribution(self, iris_data):
        """Test if classes are balanced"""
        X, y = iris_data
        unique, counts = np.unique(y, return_counts=True)
        assert all(counts == 50), "Each class should have 50 samples"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
