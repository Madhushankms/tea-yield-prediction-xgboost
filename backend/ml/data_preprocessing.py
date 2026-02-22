"""
Data Preprocessing Module for Tea Yield Prediction
Loads and prepares the custom tea yield dataset
"""

import pandas as pd
import os


def load_tea_dataset(filepath='data/tea_data.csv'):
    """
    Load the tea yield dataset and standardize column names.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset with standardized column names
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df = df.rename(columns={
        'Rainfall_mm': 'Rainfall',
        'Temperature_C': 'Temperature',
        'Fertilizer_kg': 'Fertilizer',
        'Yield_kg': 'Yield'
    })
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset statistics:\n{df.describe()}")
    
    return df


if __name__ == "__main__":
    # Test loading the dataset
    data = load_tea_dataset('data/tea_data.csv')
