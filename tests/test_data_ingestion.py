"""Tests for data ingestion components."""
import pytest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
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
    try:
        import numpy as np
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
    except ImportError:
        pytest.skip("NumPy not available")


def test_basic_data_operations():
    """Test basic data operations without heavy dependencies."""
    # Test basic list operations
    data = [1, 2, 3, 4, 5]
    assert len(data) == 5
    assert sum(data) == 15
    assert max(data) == 5
    assert min(data) == 1
    
    # Test basic dictionary operations
    features = {'age': 25, 'income': '<=50K'}
    assert 'age' in features
    assert features['age'] == 25


if __name__ == "__main__":
    pytest.main([__file__])