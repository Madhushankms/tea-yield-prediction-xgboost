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
    
    # Soil pH: 4.5-5.5 (tea prefers acidic soil)
    soil_ph = np.random.normal(5.0, 0.3, n_samples)
    soil_ph = np.clip(soil_ph, 4.0, 6.0)
    
    # Humidity: 70-90%
    humidity = np.random.normal(80, 5, n_samples)
    humidity = np.clip(humidity, 60, 95)
    
    # Altitude: 500-2000 meters (highland tea)
    altitude = np.random.normal(1200, 300, n_samples)
    altitude = np.clip(altitude, 500, 2000)
    
    # Sunlight hours: 4-8 hours per day
    sunlight_hours = np.random.normal(6, 1, n_samples)
    sunlight_hours = np.clip(sunlight_hours, 3, 9)
    
    # Age of tea plants: 5-40 years
    plant_age = np.random.normal(20, 8, n_samples)
    plant_age = np.clip(plant_age, 5, 40)
    
    # Pruning frequency: 1-4 times per year
    pruning_frequency = np.random.randint(1, 5, n_samples)
    
    # Generate target variable (Yield) with realistic relationships
    # Yield in kg/hectare (typical: 1500-3500 kg/hectare)
    yield_base = 2000
    
    # Positive influences
    yield_value = (
        yield_base +
        (rainfall - 2500) * 0.3 +
        (temperature - 24) * 50 +
        (fertilizer - 500) * 1.2 +
        (humidity - 80) * 15 +
        (sunlight_hours - 6) * 80 +
        pruning_frequency * 100 +
        # Optimal pH around 5.0
        -200 * np.abs(soil_ph - 5.0) +
        # Altitude effect (moderate altitude is optimal)
        0.5 * altitude - 0.0002 * (altitude - 1200) ** 2 +
        # Plant age effect (peak production between 15-30 years)
        -2 * np.abs(plant_age - 20)
    )
    
    # Add realistic noise
    noise = np.random.normal(0, 200, n_samples)
    yield_value = yield_value + noise
    
    # Clip to realistic range
    yield_value = np.clip(yield_value, 1000, 4000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Fertilizer': fertilizer,
        'Soil_pH': soil_ph,
        'Humidity': humidity,
        'Altitude': altitude,
        'Sunlight_Hours': sunlight_hours,
        'Plant_Age': plant_age,
        'Pruning_Frequency': pruning_frequency,
        'Yield': yield_value
    })
    
    # Round to reasonable precision
    data = data.round({
        'Rainfall': 1,
        'Temperature': 1,
        'Fertilizer': 1,
        'Soil_pH': 2,
        'Humidity': 1,
        'Altitude': 0,
        'Sunlight_Hours': 1,
        'Plant_Age': 0,
        'Pruning_Frequency': 0,
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
