# src/xgboost_model_training.py
import os
import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)

# XGBoost import
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('xgboost_model_training.log'), 
        logging.StreamHandler()
    ]
)

class XGBoostModelTrainer:
    def __init__(self):
        """
        Initialize the XGBoost model trainer
        """
        try:
            # Load California Housing dataset
            housing = fetch_california_housing()
            self.X = housing.data
            self.y = housing.target
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            # Create preprocessor
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
            
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            
            # Save preprocessor
            joblib.dump(self.preprocessor, 'models/xgboost_preprocessor.pkl')
            
            logging.info("Data preprocessing completed successfully")
        
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def train_xgboost(self):
        """
        Train XGBoost Regressor with Randomized Search
        
        Returns:
            tuple: Best model and performance metrics
        """
        try:
            # Comprehensive XGBoost hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 400],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.3, 0.5],
                'min_child_weight': [1, 3, 5, 7]
            }

            # Create XGBoost Regressor
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror', 
                random_state=42
            )
            
            # Randomized Search
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=50,  # Increased iterations for more comprehensive search
                cv=5,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            # Fit the model
            random_search.fit(self.X_train_transformed, self.y_train)
            
            # Best model
            best_xgb_model = random_search.best_estimator_
            
            # Predictions
            y_pred = best_xgb_model.predict(self.X_test_transformed)
            
            # Performance metrics
            metrics = self.evaluate_model(y_pred)
            
            # Save the best model
            joblib.dump(best_xgb_model, 'models/xgboost_house_price_model.pkl')
            
            # Visualize feature importance
            self.plot_feature_importance(best_xgb_model)
            
            return best_xgb_model, metrics
        
        except Exception as e:
            logging.error(f"Error during XGBoost training: {e}")
            raise

    def evaluate_model(self, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_pred (np.array): Predicted values
        
        Returns:
            dict: Performance metrics
        """
        try:
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'MAE': mean_absolute_error(self.y_test, y_pred),
                'MAPE': mean_absolute_percentage_error(self.y_test, y_pred),
                'R²': r2_score(self.y_test, y_pred)
            }
            
            # Log metrics
            logging.info("XGBoost Model Performance:")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value}")
            
            return metrics
        
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

    def plot_feature_importance(self, model):
        """
        Plot feature importance
        
        Args:
            model (XGBRegressor): Trained XGBoost model
        """
        try:
            # Ensure reports directory exists
            os.makedirs('reports', exist_ok=True)
            
            # Get feature importances
            feature_importance = model.feature_importances_
            feature_names = fetch_california_housing().feature_names
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.title("XGBoost Feature Importances")
            plt.bar(range(len(feature_importance)), feature_importance[indices])
            plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('reports/xgboost_feature_importance.png')
            plt.close()
            
            logging.info("Feature importance plot saved successfully")
        
        except Exception as e:
            logging.error(f"Error creating feature importance plot: {e}")

    def learning_curves(self, model):
        """
        Plot learning curves
        
        Args:
            model (XGBRegressor): Trained XGBoost model
        """
        try:
            from sklearn.model_selection import learning_curve
            
            # Compute learning curves
            train_sizes, train_scores, test_scores = learning_curve(
                model, 
                self.X_train_transformed, 
                self.y_train,
                cv=5,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            # Calculate mean and standard deviation
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curves
            plt.figure(figsize=(10, 6))
            plt.title('XGBoost Learning Curves')
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            
            plt.plot(train_sizes, train_mean, label='Training Score')
            plt.plot(train_sizes, test_mean, label='Cross-validation Score')
            
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
            
            plt.legend()
            plt.grid(True)
            plt.savefig('reports/xgboost_learning_curves.png')
            plt.close()
            
            logging.info("Learning curves plot saved successfully")
        
        except Exception as e:
            logging.error(f"Error creating learning curves plot: {e}")

def main():
    try:
        # Initialize XGBoost trainer
        trainer = XGBoostModelTrainer()
        
        # Train XGBoost model
        xgb_model, metrics = trainer.train_xgboost()
        
        # Print final metrics
        print("\nFinal XGBoost Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()