from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from typing import List
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import get_model
from ml.dataset import ACTIVITY_LABELS

# Initialize FastAPI app
app = FastAPI(
    title="Human Activity Recognition API",
    description="REST API for classifying human activities using LSTM model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
scaler = None

# Request/Response models
class SensorData(BaseModel):
    """Sensor data for a single time window (128 timesteps × 9 features)"""
    data: List[List[float]]  # Shape: (128, 9)
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [[0.1] * 9 for _ in range(128)]  # Example 128x9 data
            }
        }


class PredictionResponse(BaseModel):
    """Response containing activity prediction"""
    activity: str
    confidence: float
    probabilities: dict


# Model loading
def load_model():
    """Load the trained LSTM model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Please train the model first using: python ml/train.py"
        )
    
    # Create model
    model = get_model(
        input_size=9,
        hidden_size=128,
        num_layers=2,
        num_classes=6,
        dropout=0.3
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load scaler
    scaler_path = os.path.join(base_dir, 'saved_models', 'scaler.joblib')
    if os.path.exists(scaler_path):
        global scaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
    else:
        print(f"Warning: Scaler not found at {scaler_path}")

    print(f"Model loaded successfully from {model_path}")
    print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")



# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Human Activity Recognition API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "activities": "/activities"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not loaded"
    }


@app.get("/activities")
async def get_activities():
    """Get list of supported activities"""
    return {
        "activities": list(ACTIVITY_LABELS.values()),
        "count": len(ACTIVITY_LABELS)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_activity(sensor_data: SensorData):
    """
    Predict human activity from sensor data
    
    Expected input: 128 timesteps × 9 features
    Features: [body_acc_x, body_acc_y, body_acc_z, 
               body_gyro_x, body_gyro_y, body_gyro_z,
               total_acc_x, total_acc_y, total_acc_z]
    """
    # Load model if not already loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    try:
        # Convert to numpy array and validate shape
        data_array = np.array(sensor_data.data)
        if data_array.shape != (128, 9):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape {data_array.shape}. Expected (128, 9)"
            )
        
        # Apply scaling if available
        if scaler is not None:
            # Reshape for scaling: (timesteps, features) -> (1 * timesteps, features)
            data_scaled = scaler.transform(data_array)
            # data_scaled already has (128, 9)
        else:
            data_scaled = data_array
            
        # Convert to PyTorch tensor and add batch dimension
        input_tensor = torch.FloatTensor(data_scaled).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare response
        activity = ACTIVITY_LABELS[predicted_class]
        probs_dict = {
            ACTIVITY_LABELS[i]: float(probabilities[0][i])
            for i in range(len(ACTIVITY_LABELS))
        }
        
        return PredictionResponse(
            activity=activity,
            confidence=confidence,
            probabilities=probs_dict
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_batch(sensor_data_list: List[SensorData]):
    """Predict activities for multiple sensor data samples"""
    results = []
    
    for sensor_data in sensor_data_list:
        result = await predict_activity(sensor_data)
        results.append(result)
    
    return {"predictions": results, "count": len(results)}


# Image prediction endpoint
from fastapi import UploadFile, File

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict human activity from an image file
    Uses CLIP zero-shot classification
    """
    # Initialize classifier
    try:
        from ml.image_model import get_image_classifier
            classifier = get_image_classifier()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize image classifier: {str(e)}"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Make prediction
        activity, confidence, probabilities = classifier.predict(file.file)
        
        return PredictionResponse(
            activity=activity,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
