"""
Data Generation Module for Tea Yield Prediction
Generates synthetic dataset with realistic environmental and agricultural features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_tea_yield_dataset(n_samples=60000, random_state=42):
    """
    Generate synthetic tea yield dataset with environmental and agricultural features.
    
    Args:
        n_samples (int): Number of records to generate (default: 60000)
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Generated dataset with features and target variable (Yield)
    """
    np.random.seed(random_state)
    
    # Generate realistic feature distributions
    # Rainfall: 1500-4000mm per year (tea requires high rainfall)
    rainfall = np.random.normal(2500, 500, n_samples)
    rainfall = np.clip(rainfall, 1500, 4000)
    
    # Temperature: 20-30°C (optimal tea growing temperature)
    temperature = np.random.normal(24, 2.5, n_samples)
    temperature = np.clip(temperature, 18, 32)
    
    # Fertilizer: 200-800 kg/hectare
    fertilizer = np.random.normal(500, 150, n_samples)
    fertilizer = np.clip(fertilizer, 200, 800)
    
    # Generate target variable (Yield) with realistic relationships
    # Yield in kg (typical: 800-2000 kg)
    yield_base = 1200
    
    # Influences from 3 features
    yield_value = (
        yield_base +
        (rainfall - 250) * 1.2 +
        (temperature - 25) * 20 +
        (fertilizer - 500) * 0.8
    )
    
    # Add realistic noise
    noise = np.random.normal(0, 100, n_samples)
    yield_value = yield_value + noise
    
    # Clip to realistic range
    yield_value = np.clip(yield_value, 500, 2500)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Fertilizer': fertilizer,
        'Yield': yield_value
    })
    
    # Round to reasonable precision
    data = data.round({
        'Rainfall': 1,
        'Temperature': 1,
        'Fertilizer': 1,
        'Yield': 1
    })
    
    return data


def save_dataset(data, filepath='data/tea_data.csv'):
    """
    Save generated dataset to CSV file.
    
    Args:
        data (pd.DataFrame): Dataset to save
        filepath (str): Output file path
    """
    data.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nDataset statistics:\n{data.describe()}")


if __name__ == "__main__":
    # Generate dataset
    tea_data = generate_tea_yield_dataset(n_samples=60000)
    
    # Save to CSV
    import os
    os.makedirs('data', exist_ok=True)
    save_dataset(tea_data, 'data/tea_data.csv')
