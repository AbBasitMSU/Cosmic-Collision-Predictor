import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load or simulate dataset
# Assuming there's a dataset available with relevant features
@st.cache
def load_data():
    data = pd.read_csv('asteroid_data.csv')  # Replace with your dataset file
    return data

data = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ("New Body Input", "Documentation", "Data Processing", "Analysis", "Result Prediction"))

# New Body Input Section
if section == "New Body Input":
    st.title("Add New Celestial Body")
    speed = st.number_input("Speed (km/s)", min_value=0.0, value=0.0)
    angle = st.number_input("Angle (degrees)", min_value=0.0, value=0.0)
    distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=0.0)
    age = st.number_input("Age (millions of years)", min_value=0.0, value=0.0)
    
    if st.button("Predict Collision Risk"):
        # Placeholder for prediction model
        features = np.array([speed, angle, distance, age]).reshape(1, -1)
        # Load model (Assuming it was saved earlier)
        model = pickle.load(open('collision_model.pkl', 'rb'))
        prediction = model.predict(features)
        st.write(f"Collision Probability: {prediction[0]}")

# Documentation Section
elif section == "Documentation":
    st.title("Cosmic Collision Predictor - Documentation")
    st.write("This app predicts the collision probability of asteroids based on features like speed, distance, and angle.")
    st.write("We use advanced Machine Learning models to provide accurate risk analysis.")
    st.write("Each section provides an insight into what the app offers.")

# Data Processing Section
elif section == "Data Processing":
    st.title("Data Processing Overview")
    st.write("### Data Cleaning and Preparation")
    st.write("The data is cleaned by removing NaN values, scaling using StandardScaler, and ensuring valid data entries.")
    st.write("Below is a preview of the cleaned dataset:")
    
    # Data Cleaning Process
    data_cleaned = data.dropna()
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_cleaned), columns=data_cleaned.columns)
    st.write(data_scaled.head())

# Analysis Section
elif section == "Analysis":
    st.title("Asteroid Data Analysis")
    st.write("### Visualizing Key Features")
    st.write("Use the visualizations below to understand relationships between different asteroid features.")
    
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x='speed',
        y='distance',
        color='collision_probability',
        tooltip=['name', 'speed', 'distance', 'collision_probability']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.write("The scatter plot above visualizes the speed vs. distance for asteroids, color-coded by collision probability.")

# Result Prediction Section
elif section == "Result Prediction":
    st.title("Result Prediction for New Bodies")
    st.write("### Predict Collision Probability")
    st.write("Use the prediction model to estimate the collision risk based on input features such as speed, angle, distance, and age.")
    
    # Assume the model is already trained and saved as 'collision_model.pkl'
    model = pickle.load(open('collision_model.pkl', 'rb'))
    
    X = data[['speed', 'angle', 'distance', 'age']]
    y = data['collision_probability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # User Input Form for Prediction
    user_speed = st.number_input("Speed for Prediction", min_value=0.0, value=0.0)
    user_angle = st.number_input("Angle for Prediction", min_value=0.0, value=0.0)
    user_distance = st.number_input("Distance for Prediction", min_value=0.0, value=0.0)
    user_age = st.number_input("Age for Prediction", min_value=0.0, value=0.0)
    
    if st.button("Predict New Body Collision Risk"):
        new_features = np.array([user_speed, user_angle, user_distance, user_age]).reshape(1, -1)
        new_prediction = model.predict(new_features)
        st.write(f"Predicted Collision Probability: {new_prediction[0]}")
