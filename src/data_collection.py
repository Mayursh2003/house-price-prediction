import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_california_housing(save_path='data/california_housing.csv'):
    """
    Load California Housing dataset and optionally save to CSV
    
    Args:
        save_path (str): Path to save the dataset
    
    Returns:
        pd.DataFrame: Housing dataset
    """
    try:
        # Fetch dataset
        housing = fetch_california_housing()
        df = pd.DataFrame(
            housing.data, 
            columns=housing.feature_names
        )
        df['PRICE'] = housing.target
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            logging.info(f"Dataset saved to {save_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading or saving dataset: {e}")
        raise

def main():
    # Load and save dataset
    df = load_california_housing()
    logging.info(f"Dataset loaded. Shape: {df.shape}")

if __name__ == "__main__":
    main()