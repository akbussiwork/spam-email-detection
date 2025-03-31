from flask import Flask, request, jsonify
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')  # Ensure vectorizer is also loaded

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Spam Email Detection App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if 'text' key exists in the request
    if not data or 'text' not in data:
        return jsonify({'error': "Invalid input! Please provide an email text."}), 400

    email_text = data['text']
    
    # Transform input text using the loaded vectorizer
    text_features = vectorizer.transform([email_text])
    
    # Make prediction using the trained model
    prediction = model.predict(text_features)
    
    return jsonify({'prediction': int(prediction[0])})  # Convert NumPy int to regular int

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')  # Ensure vectorizer is also loaded

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Spam Email Detection App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if 'text' key exists in the request
    if not data or 'text' not in data:
        return jsonify({'error': "Invalid input! Please provide an email text."}), 400

    email_text = data['text']
    
    # Transform input text using the loaded vectorizer
    text_features = vectorizer.transform([email_text])
    
    # Make prediction using the trained model
    prediction = model.predict(text_features)
    
    return jsonify({'prediction': int(prediction[0])})  # Convert NumPy int to regular int

if __name__ == '__main__':
    # Get the port from environment variables, default to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
