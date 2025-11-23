"""
FastAPI Model Serving Service
Provides REST API endpoints for model predictions
"""
import os
import json
import glob
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
from sklearn.preprocessing import StandardScaler
import uvicorn

from model import TabularMLP

app = FastAPI(
    title="Federated Learning Model API",
    description="REST API for serving federated learning model predictions",
    version="1.0.0"
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
current_model: Optional[TabularMLP] = None
global_scaler: Optional[StandardScaler] = None
feature_columns: Optional[List[str]] = None
model_metadata: Dict[str, Any] = {}

class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values as key-value pairs")
    
class PredictionBatch(BaseModel):
    batch: List[Dict[str, float]] = Field(..., description="List of feature dictionaries for batch prediction")

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Prediction probability (0-1)")
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Prediction confidence")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of predictions")

class ModelInfo(BaseModel):
    model_path: Optional[str]
    model_round: Optional[int]
    feature_count: Optional[int]
    model_architecture: str
    loaded_at: Optional[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_columns_loaded: bool


def load_latest_model():
    """Load the latest trained model from models directory"""
    global current_model, model_metadata
    
    model_files = glob.glob("models/global_model_round_*.pt")
    if not model_files:
        return False
        
    # Get latest model by round number
    latest_file = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    try:
        # Load model state dict
        state_dict = torch.load(latest_file, map_location="cpu")
        
        # Infer input dimension from first layer
        first_weight = None
        for param in state_dict.values():
            if isinstance(param, torch.Tensor) and param.ndim == 2:
                first_weight = param
                break
                
        if first_weight is None:
            return False
            
        input_dim = first_weight.shape[1]
        
        # Create and load model
        current_model = TabularMLP(input_dim=input_dim)
        current_model.load_state_dict(state_dict)
        current_model.eval()
        
        # Extract round number
        round_num = int(latest_file.split('_')[-1].split('.')[0])
        
        model_metadata = {
            "model_path": latest_file,
            "model_round": round_num,
            "input_dim": input_dim,
            "model_architecture": "TabularMLP",
            "loaded_at": pd.Timestamp.now().isoformat()
        }
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def load_client_datasets_and_scaler():
    """Load client datasets to build global scaler and feature columns"""
    global global_scaler, feature_columns
    
    try:
        with open("client_datasets.pkl", "rb") as f:
            client_datasets = pickle.load(f)
            
        # Build global scaler from all clients' raw training data
        all_frames = []
        for cid, ds in client_datasets.items():
            df = ds.get("X_train_raw")
            if isinstance(df, pd.DataFrame):
                all_frames.append(df)
                
        if not all_frames:
            return False
            
        global_df = pd.concat(all_frames, axis=0)
        global_scaler = StandardScaler()
        global_scaler.fit(global_df.values)
        feature_columns = list(global_df.columns)
        
        return True
        
    except Exception as e:
        print(f"Error loading client datasets: {e}")
        return False


def preprocess_features(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to normalized array for model input"""
    if global_scaler is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Scaler or feature columns not loaded")
        
    # Create feature vector with zeros for missing features
    feature_vector = pd.Series(0.0, index=feature_columns)
    
    # Fill in provided features
    for key, value in features.items():
        if key in feature_columns:
            feature_vector[key] = value
            
    # Normalize using global scaler
    normalized = global_scaler.transform([feature_vector.values])
    return normalized.astype(np.float32)


@app.on_event("startup")
async def startup_event():
    """Initialize model and scaler on startup"""
    print("Loading model and scaler...")
    
    model_loaded = load_latest_model()
    scaler_loaded = load_client_datasets_and_scaler()
    
    if model_loaded:
        print(f"✅ Model loaded: {model_metadata.get('model_path')}")
    else:
        print("❌ Failed to load model")
        
    if scaler_loaded:
        print(f"✅ Scaler loaded with {len(feature_columns)} features")
    else:
        print("❌ Failed to load scaler")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=current_model is not None,
        scaler_loaded=global_scaler is not None,
        feature_columns_loaded=feature_columns is not None
    )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model"""
    if current_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    return ModelInfo(
        model_path=model_metadata.get("model_path"),
        model_round=model_metadata.get("model_round"),
        feature_count=len(feature_columns) if feature_columns else None,
        model_architecture=model_metadata.get("model_architecture", "TabularMLP"),
        loaded_at=model_metadata.get("loaded_at")
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make a single prediction"""
    if current_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    try:
        # Preprocess features
        X = preprocess_features(input_data.features)
        
        # Make prediction
        with torch.no_grad():
            tensor_input = torch.from_numpy(X)
            prediction = current_model(tensor_input).cpu().numpy()[0]
            
        # Convert to probability and class
        probability = float(prediction)
        predicted_class = ">50K" if probability >= 0.5 else "<=50K"
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            prediction=probability,
            predicted_class=predicted_class,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_data: PredictionBatch):
    """Make batch predictions"""
    if current_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    try:
        predictions = []
        
        for features in batch_data.batch:
            # Preprocess features
            X = preprocess_features(features)
            
            # Make prediction
            with torch.no_grad():
                tensor_input = torch.from_numpy(X)
                prediction = current_model(tensor_input).cpu().numpy()[0]
                
            # Convert to probability and class
            probability = float(prediction)
            predicted_class = ">50K" if probability >= 0.5 else "<=50K"
            confidence = max(probability, 1 - probability)
            
            predictions.append(PredictionResponse(
                prediction=probability,
                predicted_class=predicted_class,
                confidence=confidence
            ))
            
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the latest model (useful after retraining)"""
    def reload():
        model_loaded = load_latest_model()
        if model_loaded:
            print(f"✅ Model reloaded: {model_metadata.get('model_path')}")
        else:
            print("❌ Failed to reload model")
            
    background_tasks.add_task(reload)
    return {"message": "Model reload initiated"}


@app.get("/features")
async def get_feature_list():
    """Get list of all available features"""
    if feature_columns is None:
        raise HTTPException(status_code=503, detail="Feature columns not loaded")
        
    return {
        "features": feature_columns,
        "feature_count": len(feature_columns)
    }


if __name__ == "__main__":
    uvicorn.run(
        "model_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )