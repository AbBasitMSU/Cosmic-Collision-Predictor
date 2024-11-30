import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf  # For loading .h5 models
import os

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ("Home", "Data Overview", "Predict Impact", "Documentation"),
)

# Function to Load Data
@st.cache
def load_data():
    orbit_data = pd.read_csv("cleaned_Asteroid_orbit.csv")
    impact_data = pd.read_csv("impacts.csv")
    return orbit_data, impact_data

# Function to Load ML Model
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model_path = f"h5_Files/{model_name}"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error(f"Model file '{model_name}' not found!")
        st.stop()

# Section: Home
if section == "Home":
    st.title("Asteroid Impact Prediction")
    st.write(
        """
        Welcome to the Asteroid Impact Prediction Web App! This app uses advanced 
        Machine Learning and Neural Network models to predict potential asteroid impacts.
        
        Navigate through the sections to explore data, make predictions, or view documentation.
        """
    )

# Section: Data Overview
elif section == "Data Overview":
    st.title("Data Overview")
    orbit_data, impact_data = load_data()
    st.subheader("Orbit Data")
    st.write(orbit_data.head())
    st.subheader("Impact Data")
    st.write(impact_data.head())

    st.write("### Dataset Details")
    st.write(
        """
        - **Orbit Data**: Contains cleaned asteroid orbit data with columns like distance, velocity, and angle.
        - **Impact Data**: Contains details of past impacts and predictions for future ones.
        """
    )

# Section: Predict Impact
elif section == "Predict Impact":
    st.title("Predict Asteroid Impact")
    st.write(
        """
        Enter the details of an asteroid to predict the probability of an impact 
        using the pre-trained Neural Network models.
        """
    )

    # Input Fields
    velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=10.0, step=0.1)
    distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=1.0, step=0.1)
    angle = st.number_input("Angle (degrees)", min_value=0.0, value=45.0, step=0.1)
    size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)

    # Select Model
    model_name = st.selectbox(
        "Choose Prediction Model",
        [
            "Asteroid_Impact_Model.h5",
            "Asteroid_Impact_Optimization_Model.h5",
        ],
    )

    if st.button("Predict Impact"):
        model = load_model(model_name)
        input_data = np.array([[velocity, distance, angle, size]])
        prediction = model.predict(input_data)
        impact_probability = prediction[0][0]

        st.subheader("Prediction Result")
        st.write(f"Impact Probability: **{impact_probability:.2%}**")

# Section: Documentation
elif section == "Documentation":
    st.title("Documentation")
    st.write("### About the Project")
    st.write(
        """
        This app leverages datasets related to asteroid orbits and impacts. It uses 
        cleaned data (`cleaned_Asteroid_orbit.csv`) and models trained on Jupyter 
        and Colab notebooks.
        """
    )
    st.write("### Files Overview")
    st.write(
        """
        - **impacts.csv**: Raw impact data.
        - **orbits.csv**: Raw orbit data.
        - **cleaned_Asteroid_orbit.csv**: Cleaned version of orbit data.
        - **Asteroid_Impact_Model.h5**: Pre-trained NN model for impact prediction.
        - **Asteroid_Impact_Optimization_Model.h5**: Optimized NN model for impact prediction.
        - **Asteroid_Predictions.ipynb**: Data cleaning and model training script.
        """
    )
    st.write("### Columns in the Data")
    st.write(
        """
        - **Velocity (km/s)**: Speed of the asteroid.
        - **Distance (AU)**: Distance from Earth in Astronomical Units.
        - **Angle (degrees)**: Trajectory angle.
        - **Size (km)**: Estimated size of the asteroid.
        """
    )
