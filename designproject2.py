import serial
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Read dataset and train models
def train_models():
    # Load dataset
    df = pd.read_csv(r'C:\Users\AMALSIVAN\OneDrive\Desktop\designproject.csv')  
    
    # Print the columns of the DataFrame to ensure the correct columns are present
    print("Columns in the dataset:", df.columns)
    
    X = df[['temp', 'humidity']].values
    y_temp_humidity = df[['temp', 'humidity']].values
    
    # Check if 'conditions' column exists
    if 'conditions' not in df.columns:
        raise KeyError("'conditions' column is not found in the dataset. Please ensure the dataset has 'temp', 'humidity', and 'conditions' columns.")
    
    y_condition = df['conditions'].values

    # Encode condition labels
    label_encoder = LabelEncoder()
    y_condition_encoded = label_encoder.fit_transform(y_condition)

    # Train regression model for temperature and humidity
    regressor = MultiOutputRegressor(LinearRegression())
    regressor.fit(X, y_temp_humidity)

    # Train classification model for condition
    classifier = RandomForestClassifier()
    classifier.fit(X, y_condition_encoded)

    # Save models and label encoder
    joblib.dump(regressor, 'regressor_model.pkl')
    joblib.dump(classifier, 'classifier_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    print("Models trained and saved.")

# Function to load models
def load_models():
    regressor = joblib.load('regressor_model.pkl')
    classifier = joblib.load('classifier_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return regressor, classifier, label_encoder

# Train models initially
train_models()

# Load trained models
regressor, classifier, label_encoder = load_models()

# Establish serial connection
ser = serial.Serial('COM9', 9600)  
time.sleep(2)  

# Real-time prediction loop
while True:
    temp_humidity_str = ser.readline().strip().decode('utf-8')
    print(f"Received data: {temp_humidity_str}")  # Print received data for debugging
    try:
        temp, humidity = map(float, temp_humidity_str.split(','))
        print(f"Temperature: {temp}, Humidity: {humidity}")
    except ValueError:
        print(f"Ignoring non-numeric value: {temp_humidity_str}")
        continue

    # Prepare input for prediction
    X_real_time = np.array([[temp, humidity]])

    # Predict next temperature and humidity
    predicted_temp_humidity = regressor.predict(X_real_time)
    predicted_temp, predicted_humidity = predicted_temp_humidity[0]

    # Predict condition
    predicted_condition_encoded = classifier.predict(X_real_time)
    predicted_condition = label_encoder.inverse_transform(predicted_condition_encoded)

    print(f"Predicted next temperature: {predicted_temp}")
    print(f"Predicted next humidity: {predicted_humidity}")
    print(f"Predicted condition: {predicted_condition[0]}")

    # Send predicted temperature, humidity, and condition to Arduino
    ser.write(f"{predicted_temp},{predicted_humidity},{predicted_condition[0]}\n".encode())

    time.sleep(2)  # Adjust as needed for real-time performance
