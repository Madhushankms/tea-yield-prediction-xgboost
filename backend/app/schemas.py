"""
Pydantic schemas for request/response validation in FastAPI.
"""

from pydantic import BaseModel, Field
from typing import Dict


class PredictionInput(BaseModel):
    """
    Input schema for tea yield prediction request.
    All features required for the XGBoost model.
    """
    rainfall: float = Field(..., description="Annual rainfall in mm", ge=0, le=500)
    temperature: float = Field(..., description="Average temperature in °C", ge=0, le=50)
    fertilizer: float = Field(..., description="Fertilizer amount in kg", ge=0, le=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "rainfall": 150.0,
                "temperature": 25.0,
                "fertilizer": 400.0
            }
        }


class PredictionOutput(BaseModel):
    """
    Output schema for tea yield prediction response.
    Contains predicted yield and feature importance scores.
    """
    prediction: float = Field(..., description="Predicted tea yield in kg/hectare")
    feature_importance: Dict[str, float] = Field(..., description="Importance score for each feature")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1250.5,
                "feature_importance": {
                    "Rainfall": 0.35,
                    "Temperature": 0.32,
                    "Fertilizer": 0.33
                }
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }
