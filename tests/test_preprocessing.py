# src/model_training.py
import os
import sys
import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Explicit imports with full path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

# XGBoost import
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'), 
        logging.StreamHandler()
    ]
)

class HousePriceModelTrainer:
    def __init__(self):
        try:
            # Verify import works
            print("Attempting to load California Housing dataset...")
            housing = fetch_california_housing()
            
            self.X = housing.data
            self.y = housing.target
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            # Preprocessor setup
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), list(range(self.X.shape[1])))
                ]
            )
            
            # Prepare transformed data
            self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
            self.X_test_transformed = self.preprocessor.transform(self.X_test)
            
            # Save preprocessor
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
            
            logging.info("Data preprocessing completed successfully")
        
        except ImportError as e:
            logging.error(f"Import Error: {e}")
            print(f"Import Error: {e}")
            print("Scikit-learn import paths:")
            print(sys.path)
            raise
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def train_random_forest(self):
        """
        Train Random Forest with Randomized Search
        """
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        }

        rf_model = RandomForestRegressor(random_state=42)
        
        # Randomized Search
        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        random_search.fit(self.X_train_transformed, self.y_train)
        best_model = random_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(self.X_test_transformed)
        
        # Performance metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred)
        }
        
        # Save model
        joblib.dump(best_model, 'models/house_price_model.pkl')
        
        return best_model, metrics

def main():
    try:
        # Initialize and run model training
        trainer = HousePriceModelTrainer()
        model, metrics = trainer.train_random_forest()
        
        # Print metrics
        print("Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value}")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()