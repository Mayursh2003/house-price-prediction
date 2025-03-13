import os
from src.data_collection import load_california_housing
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import joblib

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and save dataset
    df = load_california_housing()
    
    # Preprocessing
    preprocessor = DataPreprocessor('data/california_housing.csv')
    preprocessor.prepare_features()
    
    # Create preprocessing pipeline
    pipeline = preprocessor.create_preprocessing_pipeline()
    pipeline.fit(preprocessor.X_train)
    
    # Save preprocessor
    joblib.dump(pipeline, 'models/preprocessor.pkl')
    
    # Transform data
    X_train_transformed = pipeline.transform(preprocessor.X_train)
    X_test_transformed = pipeline.transform(preprocessor.X_test)
    
    # Model Training
    trainer = ModelTrainer(
        X_train_transformed, 
        X_test_transformed, 
        preprocessor.y_train, 
        preprocessor.y_test
    )
    
    # Train model
    model, metrics = trainer.train_random_forest()
    
    # Print metrics
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value}")

if __name__ == "__main__":
    main()