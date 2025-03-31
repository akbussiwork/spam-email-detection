import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/dataset.csv')

# Preprocess dataset (example: remove null values and features selection)
data = data.dropna()

# Feature selection (adjust according to your dataset)
X = data['text']  # Assuming you have a 'text' column for email content
y = data['label']  # Assuming you have a 'label' column for spam (1) or not spam (0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/spam_model.pkl')
