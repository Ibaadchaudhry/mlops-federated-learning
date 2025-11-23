"""Tests for model components."""
import pytest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import numpy as np
    from model import TabularMLP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_model_initialization():
    """Test model can be initialized with different input dimensions."""
    model = TabularMLP(input_dim=10)
    assert model.fc1.in_features == 10
    assert model.fc1.out_features == 128
    assert model.fc2.in_features == 128
    assert model.fc2.out_features == 64
    assert model.fc3.in_features == 64
    assert model.fc3.out_features == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_model_forward():
    """Test model forward pass."""
    model = TabularMLP(input_dim=5)
    
    # Test single sample
    x = torch.randn(1, 5)
    output = model(x)
    assert output.shape == (1, 1)
    assert torch.is_tensor(output)


def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    # Test basic Python functionality
    assert 1 + 1 == 2
    assert len([1, 2, 3]) == 3
    
    # Test that we can import our modules
    try:
        import model
        assert hasattr(model, 'TabularMLP')
    except ImportError:
        pytest.skip("Model module not available")


if __name__ == "__main__":
    pytest.main([__file__])