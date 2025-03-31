from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Spam Email Detection App!"

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib

# Load the model (we'll add this after training the model)
model = joblib.load('models/spam_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Spam Email Detection App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data['text']
    
    # Make predictions (using the trained model)
    prediction = model.predict([email_text])
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
