import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class DataPreprocessor:
    def __init__(self, data_path):
        """
        Initialize preprocessor with dataset
        
        Args:
            data_path (str): Path to dataset
        """
        try:
            self.data = pd.read_csv(data_path)
            self.X = None
            self.y = None
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, test_size=0.2, random_state=42):
        """
        Prepare features and split data
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed
        """
        try:
            # Separate features and target
            self.X = self.data.drop('PRICE', axis=1)
            self.y = self.data['PRICE']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                random_state=random_state
            )
            logging.info("Features prepared and data split into train and test sets.")
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            raise
    
    def create_preprocessing_pipeline(self, imputation_strategy='median', scaling_method='standard'):
        """
        Create preprocessing pipeline
        
        Args:
            imputation_strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent')
            scaling_method (str): Method for scaling ('standard', 'minmax')
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        try:
            numeric_features = self.X.columns.tolist()
            
            # Choose scaler
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid scaling method.")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy=imputation_strategy)),
                        ('scaler', scaler)
                    ]), numeric_features)
                ]
            )
            
            logging.info("Preprocessing pipeline created.")
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error creating preprocessing pipeline: {e}")
            raise