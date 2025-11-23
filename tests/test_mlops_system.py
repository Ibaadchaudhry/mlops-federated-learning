"""
Unit tests for the federated learning MLOps system
"""
import pytest
import numpy as np
import pandas as pd
import torch
import pickle
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

# Test imports
from model import TabularMLP
from train_utils import train_local, evaluate_model
from drift_detector import detect_drift_featurewise, psi


class TestTabularMLP:
    """Test the neural network model"""
    
    def test_model_creation(self):
        """Test model can be created with correct architecture"""
        model = TabularMLP(input_dim=10)
        assert isinstance(model, TabularMLP)
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5,)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid output
    
    def test_model_parameters(self):
        """Test model has expected parameters"""
        model = TabularMLP(input_dim=20, hidden_dims=(64, 32))
        params = list(model.parameters())
        
        # Should have weights and biases for each layer
        assert len(params) > 0
        
        # Check input dimension matches
        first_weight = params[0]
        assert first_weight.shape[1] == 20


class TestTrainingUtils:
    """Test training and evaluation utilities"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5).astype(np.float32)
        self.y = (np.random.randn(100) > 0).astype(np.float32)
        self.model = TabularMLP(input_dim=5)
    
    def test_train_local(self):
        """Test local training function"""
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Train for one epoch
        train_local(self.model, self.X, self.y, epochs=1, lr=1e-2)
        
        # Check parameters changed
        params_changed = False
        for name, param in self.model.named_parameters():
            if not torch.allclose(initial_params[name], param):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should change after training"
    
    def test_evaluate_model(self):
        """Test model evaluation function"""
        metrics = evaluate_model(self.model, self.X, self.y)
        
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        if metrics['auc'] is not None:
            assert 0 <= metrics['auc'] <= 1


class TestDriftDetection:
    """Test data drift detection functionality"""
    
    def setup_method(self):
        """Setup test data for drift detection"""
        np.random.seed(42)
        
        # Create baseline data
        self.baseline_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.choice([0, 1], 1000)
        })
        
        # Create current data (similar to baseline)
        self.current_no_drift = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000), 
            'feature3': np.random.choice([0, 1], 1000)
        })
        
        # Create current data with drift
        self.current_with_drift = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 1000),  # Mean shift
            'feature2': np.random.normal(5, 4, 1000),  # Variance increase
            'feature3': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # Distribution shift
        })
    
    def test_psi_calculation(self):
        """Test PSI calculation"""
        # No drift case
        baseline = self.baseline_df['feature1'].values
        current = self.current_no_drift['feature1'].values
        
        psi_score = psi(baseline, current)
        assert isinstance(psi_score, float)
        assert psi_score >= 0
        
        # With drift case
        current_drift = self.current_with_drift['feature1'].values
        psi_drift = psi(baseline, current_drift)
        assert psi_drift > psi_score  # Should detect more drift
    
    def test_drift_detection(self):
        """Test feature-wise drift detection"""
        # No drift case
        results_no_drift = detect_drift_featurewise(
            self.baseline_df, 
            self.current_no_drift
        )
        
        assert len(results_no_drift) == 3  # Three features
        for feature, result in results_no_drift.items():
            assert 'psi' in result
            assert 'ks_pvalue' in result
            assert 'drift_flag' in result
            assert isinstance(result['drift_flag'], bool)
        
        # With drift case
        results_with_drift = detect_drift_featurewise(
            self.baseline_df,
            self.current_with_drift
        )
        
        # Should detect more drift flags
        drift_flags_no = sum(r['drift_flag'] for r in results_no_drift.values())
        drift_flags_with = sum(r['drift_flag'] for r in results_with_drift.values())
        
        assert drift_flags_with >= drift_flags_no


class TestDataIngestion:
    """Test data ingestion and preparation"""
    
    @patch('data_ingestion.FederatedDataset')
    def test_load_federated_data(self, mock_dataset):
        """Test federated data loading"""
        # Mock the dataset
        mock_partition = MagicMock()
        mock_df = pd.DataFrame({
            'age': [25, 35, 45],
            'income': ['<=50K', '>50K', '<=50K'],
            'education': ['Bachelors', 'Masters', 'HS-grad']
        })
        mock_partition.to_pandas.return_value = mock_df
        
        mock_fds = MagicMock()
        mock_fds.load_partition.return_value = mock_partition
        mock_dataset.return_value = mock_fds
        
        # Import here to use the mock
        from data_ingestion import load_federated_data
        
        # Test the function
        result = load_federated_data(num_clients=2)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        
        for client_id, data in result.items():
            assert 'X_train_raw' in data
            assert 'X_test_raw' in data
            assert 'X_train_norm' in data
            assert 'X_test_norm' in data
            assert 'y_train' in data
            assert 'y_test' in data


class TestModelAPI:
    """Test FastAPI model service"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_features = {
            'age': 35,
            'hours-per-week': 40,
            'education.num': 13,
            'capital.gain': 0,
            'capital.loss': 0
        }
    
    def test_prediction_input_validation(self):
        """Test prediction input validation"""
        from model_api import PredictionInput
        
        # Valid input
        valid_input = PredictionInput(features=self.test_features)
        assert valid_input.features == self.test_features
        
        # Test with different feature types
        features_with_float = {'feature1': 1.5, 'feature2': 2.0}
        input_float = PredictionInput(features=features_with_float)
        assert input_float.features == features_with_float
    
    def test_model_info_response(self):
        """Test model info response structure"""
        from model_api import ModelInfo
        
        model_info = ModelInfo(
            model_path="test_model.pt",
            model_round=5,
            feature_count=100,
            model_architecture="TabularMLP",
            loaded_at="2023-01-01T00:00:00"
        )
        
        assert model_info.model_path == "test_model.pt"
        assert model_info.model_round == 5
        assert model_info.feature_count == 100


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_client_dataset_creation_and_loading(self):
        """Test end-to-end client dataset creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_data = {
                0: {
                    'X_train_raw': pd.DataFrame({'age': [25, 35], 'hours-per-week': [40, 50]}),
                    'X_test_raw': pd.DataFrame({'age': [30], 'hours-per-week': [45]}),
                    'X_train_norm': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                    'X_test_norm': np.array([[0.25, 0.35]], dtype=np.float32),
                    'y_train': np.array([0, 1], dtype=np.float32),
                    'y_test': np.array([1], dtype=np.float32),
                    'feature_columns': ['age', 'hours-per-week']
                }
            }
            
            # Save to pickle file
            pickle_path = os.path.join(temp_dir, 'test_client_datasets.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(test_data, f)
            
            # Load and verify
            with open(pickle_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            assert len(loaded_data) == 1
            assert 0 in loaded_data
            assert 'X_train_raw' in loaded_data[0]
            assert isinstance(loaded_data[0]['X_train_raw'], pd.DataFrame)
    
    def test_model_save_and_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save model
            model = TabularMLP(input_dim=10)
            model_path = os.path.join(temp_dir, 'test_model.pt')
            torch.save(model.state_dict(), model_path)
            
            # Load model
            loaded_model = TabularMLP(input_dim=10)
            loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Test they produce same output
            x = torch.randn(1, 10)
            original_output = model(x)
            loaded_output = loaded_model(x)
            
            assert torch.allclose(original_output, loaded_output, atol=1e-6)


class TestConfigurationAndEnvironment:
    """Test configuration and environment setup"""
    
    def test_requirements_parsing(self):
        """Test that requirements.txt can be parsed"""
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                requirements = f.read()
            
            # Check key dependencies are present
            assert 'torch' in requirements
            assert 'flwr' in requirements
            assert 'scikit-learn' in requirements
            assert 'pandas' in requirements
            assert 'fastapi' in requirements
            assert 'streamlit' in requirements


# Performance and load tests
class TestPerformance:
    """Performance and load tests"""
    
    def test_model_inference_speed(self):
        """Test model inference performance"""
        model = TabularMLP(input_dim=100)
        model.eval()
        
        # Test batch inference
        batch_sizes = [1, 10, 100, 1000]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 100)
            
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                output = model(x)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    print(f"Batch size {batch_size}: {inference_time:.2f}ms")
                
                assert output.shape[0] == batch_size
    
    def test_drift_detection_performance(self):
        """Test drift detection performance with large datasets"""
        np.random.seed(42)
        
        # Large datasets
        large_baseline = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.normal(5, 2, 10000),
        })
        
        large_current = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1, 10000),
            'feature2': np.random.normal(5, 2, 10000),
        })
        
        import time
        start_time = time.time()
        
        results = detect_drift_featurewise(large_baseline, large_current)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Drift detection for 20k samples took {processing_time:.2f} seconds")
        assert processing_time < 10  # Should complete within 10 seconds
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])