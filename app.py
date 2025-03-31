
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
    email_text = data['text']
    
    # Transform input text using the loaded vectorizer
    text_features = vectorizer.transform([email_text])
    
    # Make prediction using the trained model
    prediction = model.predict(text_features)
    
    return jsonify({'prediction': int(prediction[0])})  # Convert NumPy int to regular int

if __name__ == '__main__':
    app.run(debug=True)
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
