"""
Utility functions for model loading and prediction.
"""

import pickle
import numpy as np
from typing import Dict, Tuple
import os


class ModelLoader:
    """
    Singleton class for loading and managing the trained XGBoost model.
    """
    _instance = None
    _model = None
    _scaler = None
    _feature_names = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str = 'models/xgboost_model.pkl'):
        """
        Load the trained XGBoost model and associated artifacts.
        
        Args:
            model_path (str): Path to the saved model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if self._model is not None:
            print("Model already loaded.")
            return
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_artifacts = pickle.load(f)
            
            self._model = model_artifacts['model']
            self._scaler = model_artifacts['scaler']
            self._feature_names = model_artifacts['feature_names']
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Features: {self._feature_names}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self._model is not None
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Make prediction using the loaded model.
        
        Args:
            features (Dict[str, float]): Dictionary of feature values
            
        Returns:
            Tuple[float, Dict[str, float]]: Predicted yield and feature importance
            
        Raises:
            ValueError: If model is not loaded or features are invalid
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Map input features to model feature order
        feature_mapping = {
            'rainfall': 'Rainfall',
            'temperature': 'Temperature',
            'fertilizer': 'Fertilizer'
        }
        
        # Create feature array in correct order
        feature_array = []
        for feature in self._feature_names:
            # Find the corresponding input key
            input_key = None
            for k, v in feature_mapping.items():
                if v == feature:
                    input_key = k
                    break
            
            if input_key is None:
                raise ValueError(f"Feature {feature} not found in input mapping")
            
            if input_key not in features:
                raise ValueError(f"Missing required feature: {input_key}")
            
            feature_array.append(features[input_key])
        
        # Convert to numpy array and reshape
        X = np.array(feature_array).reshape(1, -1)
        
        # Scale features
        X_scaled = self._scaler.transform(X)
        
        # Make prediction
        prediction = self._model.predict(X_scaled)[0]
        
        # Get feature importance
        feature_importance = self._model.feature_importances_
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(self._feature_names, feature_importance)
        }
        
        return float(prediction), importance_dict
    
    def get_feature_names(self):
        """
        Get list of model feature names.
        
        Returns:
            list: Feature names
        """
        return self._feature_names


# Global model loader instance
model_loader = ModelLoader()
