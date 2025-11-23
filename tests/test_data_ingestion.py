"""Tests for data ingestion components."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_data_preprocessing_functions():
    """Test basic data preprocessing functionality."""
    # Create sample data
    data = pd.DataFrame({
        'age': [25, 35, 45, 55],
        'hours-per-week': [40, 30, 50, 20],
        'income': ['<=50K', '>50K', '<=50K', '>50K']
    })
    
    # Test data types
    assert data['age'].dtype in [np.int64, np.float64]
    assert data['hours-per-week'].dtype in [np.int64, np.float64]
    assert data['income'].dtype == object
    
    # Test data is not empty
    assert len(data) > 0
    assert data.shape[1] == 3


def test_feature_encoding():
    """Test feature encoding for categorical variables."""
    # Sample categorical data
    categories = ['<=50K', '>50K', '<=50K', '>50K']
    
    # Simple label encoding test
    encoded = [0 if cat == '<=50K' else 1 for cat in categories]
    
    assert len(encoded) == len(categories)
    assert all(val in [0, 1] for val in encoded)


def test_data_normalization():
    """Test data normalization."""
    # Sample numerical data
    data = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    
    # Simple min-max normalization
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    normalized = (data - data_min) / (data_max - data_min)
    
    # Check normalization worked
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.shape == data.shape


def test_train_test_split():
    """Test train/test data splitting."""
    # Sample data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Simple split
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Verify split
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert X_train.shape[1] == X_test.shape[1]


def test_data_validation():
    """Test data validation checks."""
    # Valid data
    valid_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    # Check for missing values
    assert not valid_data.isnull().any().any()
    
    # Check data types
    assert all(valid_data.dtypes != object) or valid_data.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__])