"""
FastAPI main application for Tea Yield Prediction API.
Provides endpoints for health check and yield prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import PredictionInput, PredictionOutput, HealthResponse
from app.utils import model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the ML model on startup.
    """
    # Startup: Load the model
    print("Starting up Tea Yield Prediction API...")
    try:
        model_loader.load_model('models/xgboost_model.pkl')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load model: {str(e)}")
        print("API will start but predictions will fail until model is available.")
    
    yield
    
    # Shutdown
    print("Shutting down Tea Yield Prediction API...")


# Initialize FastAPI application
app = FastAPI(
    title="Tea Yield Prediction API",
    description="REST API for predicting tea leaf yield using XGBoost machine learning model",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint providing API health status.
    
    Returns:
        HealthResponse: API status and model loading state
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring API status.
    
    Returns:
        HealthResponse: API status and model loading state
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded()
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_yield(input_data: PredictionInput):
    """
    Predict tea yield based on environmental and agricultural features.
    
    Args:
        input_data (PredictionInput): Feature values for prediction
        
    Returns:
        PredictionOutput: Predicted yield and feature importance
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists and server has started properly."
        )
    
    try:
        # Convert Pydantic model to dictionary
        features = input_data.model_dump()
        
        # Make prediction
        prediction, feature_importance = model_loader.predict(features)
        
        # Return response
        return PredictionOutput(
            prediction=round(prediction, 2),
            feature_importance=feature_importance
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/features")
async def get_features():
    """
    Get list of features required for prediction.
    
    Returns:
        dict: List of feature names and descriptions
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    features = {
        "features": model_loader.get_feature_names(),
        "description": {
            "Rainfall": "Annual rainfall in mm",
            "Temperature": "Average temperature in °C",
            "Fertilizer": "Fertilizer amount in kg"
        }
    }
    
    return features


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
