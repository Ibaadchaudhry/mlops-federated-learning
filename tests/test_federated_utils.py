"""Test utilities for federated learning components."""
import pytest
import numpy as np
import torch
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_basic_federated_simulation():
    """Test basic federated learning simulation setup."""
    # Simulate multiple clients with different data
    num_clients = 3
    client_data = {}
    
    for client_id in range(num_clients):
        # Create synthetic data for each client
        X = np.random.rand(50, 5).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        
        client_data[client_id] = {
            'X_train_norm': X[:40],
            'X_test_norm': X[40:],
            'y_train': y[:40],
            'y_test': y[40:]
        }
    
    # Verify client data structure
    assert len(client_data) == num_clients
    
    for client_id, data in client_data.items():
        assert 'X_train_norm' in data
        assert 'X_test_norm' in data
        assert 'y_train' in data
        assert 'y_test' in data
        
        # Check data shapes
        assert data['X_train_norm'].shape[1] == 5  # 5 features
        assert data['X_test_norm'].shape[1] == 5
        assert len(data['y_train']) == len(data['X_train_norm'])
        assert len(data['y_test']) == len(data['X_test_norm'])


def test_model_aggregation_simulation():
    """Test simple model parameter aggregation."""
    # Create mock model parameters
    num_clients = 3
    model_params = []
    
    for i in range(num_clients):
        # Simulate model parameters as simple tensors
        params = {
            'weight1': torch.randn(10, 5),
            'bias1': torch.randn(10),
            'weight2': torch.randn(1, 10),
            'bias2': torch.randn(1)
        }
        model_params.append(params)
    
    # Simple parameter averaging (FedAvg simulation)
    aggregated_params = {}
    
    for param_name in model_params[0].keys():
        param_list = [client_params[param_name] for client_params in model_params]
        aggregated_params[param_name] = torch.stack(param_list).mean(dim=0)
    
    # Verify aggregation
    assert len(aggregated_params) == len(model_params[0])
    
    for param_name, aggregated_param in aggregated_params.items():
        original_shape = model_params[0][param_name].shape
        assert aggregated_param.shape == original_shape


def test_data_distribution_simulation():
    """Test non-IID data distribution simulation."""
    # Simulate heterogeneous data distribution
    total_samples = 300
    num_clients = 3
    
    # Create biased data for each client
    client_distributions = []
    
    for i in range(num_clients):
        # Each client has bias toward certain classes
        if i == 0:
            # Client 0: more class 0 samples
            class_0_samples = int(0.7 * (total_samples // num_clients))
            class_1_samples = (total_samples // num_clients) - class_0_samples
        elif i == 1:
            # Client 1: more class 1 samples
            class_0_samples = int(0.3 * (total_samples // num_clients))
            class_1_samples = (total_samples // num_clients) - class_0_samples
        else:
            # Client 2: balanced
            class_0_samples = (total_samples // num_clients) // 2
            class_1_samples = (total_samples // num_clients) - class_0_samples
        
        distribution = {
            'class_0': class_0_samples,
            'class_1': class_1_samples,
            'total': class_0_samples + class_1_samples
        }
        client_distributions.append(distribution)
    
    # Verify non-IID distribution
    assert len(client_distributions) == num_clients
    
    total_class_0 = sum(dist['class_0'] for dist in client_distributions)
    total_class_1 = sum(dist['class_1'] for dist in client_distributions)
    
    assert total_class_0 + total_class_1 == total_samples
    
    # Check that distributions are different (non-IID)
    class_0_ratios = [dist['class_0'] / dist['total'] for dist in client_distributions]
    assert not all(abs(ratio - 0.5) < 0.1 for ratio in class_0_ratios)  # Not all balanced


def test_communication_round_simulation():
    """Test federated learning communication round simulation."""
    num_clients = 3
    num_rounds = 2
    
    # Simulate training rounds
    round_results = []
    
    for round_num in range(num_rounds):
        round_data = {
            'round': round_num + 1,
            'participating_clients': list(range(num_clients)),
            'aggregated_loss': np.random.uniform(0.5, 1.0),
            'aggregated_accuracy': np.random.uniform(0.6, 0.9)
        }
        round_results.append(round_data)
    
    # Verify simulation results
    assert len(round_results) == num_rounds
    
    for i, result in enumerate(round_results):
        assert result['round'] == i + 1
        assert len(result['participating_clients']) == num_clients
        assert 0 <= result['aggregated_loss'] <= 1.5
        assert 0 <= result['aggregated_accuracy'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])