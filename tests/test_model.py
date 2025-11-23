"""Tests for model components."""
import pytest
import torch
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import TabularMLP


def test_model_initialization():
    """Test model can be initialized with different input dimensions."""
    model = TabularMLP(input_dim=10)
    assert model.fc1.in_features == 10
    assert model.fc1.out_features == 128
    assert model.fc2.in_features == 128
    assert model.fc2.out_features == 64
    assert model.fc3.in_features == 64
    assert model.fc3.out_features == 1


def test_model_forward():
    """Test model forward pass."""
    model = TabularMLP(input_dim=5)
    
    # Test single sample
    x = torch.randn(1, 5)
    output = model(x)
    assert output.shape == (1, 1)
    assert torch.is_tensor(output)
    
    # Test batch
    x_batch = torch.randn(10, 5)
    output_batch = model(x_batch)
    assert output_batch.shape == (10, 1)


def test_model_parameters():
    """Test model has trainable parameters."""
    model = TabularMLP(input_dim=8)
    params = list(model.parameters())
    assert len(params) > 0
    
    # Check parameters are trainable
    for param in params:
        assert param.requires_grad


def test_model_train_eval_modes():
    """Test model can switch between train and eval modes."""
    model = TabularMLP(input_dim=5)
    
    # Test training mode
    model.train()
    assert model.training
    
    # Test evaluation mode
    model.eval()
    assert not model.training


def test_model_output_range():
    """Test model output is reasonable for classification."""
    model = TabularMLP(input_dim=5)
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(100, 5)
        output = model(x)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        
        # Check probabilities are in valid range
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)


def test_model_deterministic():
    """Test model produces consistent outputs."""
    model = TabularMLP(input_dim=5)
    model.eval()
    
    x = torch.randn(5, 5)
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)


def test_model_different_inputs():
    """Test model handles different input patterns."""
    model = TabularMLP(input_dim=5)
    
    # Test zeros
    x_zeros = torch.zeros(1, 5)
    output_zeros = model(x_zeros)
    assert torch.is_tensor(output_zeros)
    
    # Test ones
    x_ones = torch.ones(1, 5)
    output_ones = model(x_ones)
    assert torch.is_tensor(output_ones)
    
    # Outputs should be different
    assert not torch.allclose(output_zeros, output_ones, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])