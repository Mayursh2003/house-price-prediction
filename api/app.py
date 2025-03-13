from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model and preprocessor
model_path = os.getenv('MODEL_PATH', 'models/house_price_model.pkl')
preprocessor_path = os.getenv('PREPROCESSOR_PATH', 'models/preprocessor.pkl')

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logging.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or preprocessor: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.json
        if not input_data:
            raise ValueError("No input data provided.")
        
        # Validate input data
        required_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        if not all(feature in input_data for feature in required_features):
            raise ValueError("Missing required features.")
        
        # Convert to numpy array
        features = np.array([list(input_data.values())])
        
        # Preprocess and predict
        preprocessed_features = preprocessor.transform(features)
        prediction = model.predict(preprocessed_features)[0]
        
        logging.info(f"Prediction made: {prediction}")
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'House Price Prediction API is running'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))