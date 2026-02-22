"""
XGBoost Training Module for Tea Yield Prediction
Implements training pipeline with GridSearchCV for hyperparameter tuning
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap


class TeaYieldModel:
    """
    XGBoost-based model for predicting tea yield with comprehensive evaluation.
    """
    
    def __init__(self, data_path='data/tea_data.csv'):
        """
        Initialize the model with dataset path.
        
        Args:
            data_path (str): Path to the tea yield dataset
        """
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_params = None
        
    def load_and_prepare_data(self):
        """
        Load dataset and split into train/validation/test sets.
        
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Standardize column names
        df = df.rename(columns={
            'Rainfall_mm': 'Rainfall',
            'Temperature_C': 'Temperature',
            'Fertilizer_kg': 'Fertilizer',
            'Yield_kg': 'Yield'
        })
        
        print(f"Dataset shape: {df.shape}")
        
        # Separate features and target
        X = df.drop('Yield', axis=1)
        y = df['Yield']
        
        self.feature_names = X.columns.tolist()
        print(f"Features: {self.feature_names}")
        
        # Split: 70% train, 15% validation, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"\nData split:")
        print(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
        print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_with_grid_search(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model with GridSearchCV for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        print("\n" + "="*60)
        print("Starting GridSearchCV for hyperparameter tuning...")
        print("="*60)
        
        # Define parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        # Initialize base XGBoost model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Configure GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Extract best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print("\n" + "="*60)
        print("GridSearchCV completed!")
        print("="*60)
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score (neg MSE): {grid_search.best_score_:.2f}")
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"\nValidation Set Performance:")
        print(f"  RMSE: {val_rmse:.2f}")
        print(f"  MAE: {val_mae:.2f}")
        print(f"  R² Score: {val_r2:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive evaluation of the trained model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating model on test set...")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {rmse:.2f} kg/hectare")
        print(f"  MAE: {mae:.2f} kg/hectare")
        print(f"  R² Score: {r2:.4f}")
        
        return metrics
    
    def generate_shap_analysis(self, X_test):
        """
        Generate SHAP explainability analysis and visualizations.
        
        Args:
            X_test: Test features for SHAP analysis
        """
        print("\n" + "="*60)
        print("Generating SHAP explainability analysis...")
        print("="*60)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        
        # Create reports directory
        os.makedirs('reports/figures', exist_ok=True)
        
        # Generate SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_test, 
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP summary plot saved to reports/figures/shap_summary_plot.png")
        
        # Generate feature importance bar plot
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
        plt.xticks(range(len(feature_importance)), 
                   [self.feature_names[i] for i in sorted_idx], 
                   rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved to reports/figures/feature_importance.png")
    
    def save_model(self, model_path='models/xgboost_model.pkl'):
        """
        Save trained model and scaler to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs('models', exist_ok=True)
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_artifacts, f)
        
        print(f"\nModel saved to {model_path}")
    
    def train_pipeline(self):
        """
        Complete training pipeline: load data, train, evaluate, save.
        """
        print("\n" + "="*60)
        print("TEA YIELD PREDICTION - XGBOOST TRAINING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
        
        # Train with GridSearchCV
        self.train_with_grid_search(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        metrics = self.evaluate_model(X_test, y_test)
        
        # Generate SHAP analysis
        self.generate_shap_analysis(X_test)
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("Training pipeline completed successfully!")
        print("="*60)
        
        return metrics


if __name__ == "__main__":
    # Initialize and run training pipeline
    model = TeaYieldModel(data_path='data/tea_data.csv')
    model.train_pipeline()
