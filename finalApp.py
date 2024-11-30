import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf  # For loading .h5 models
import random
import os

# Background Image Function
def set_background(image_url):
    """
    Set a light blurry background image in the Streamlit app using custom CSS.
    :param image_url: URL or local path of the image.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.5); /* Light overlay for readability */
            z-index: -1;
            backdrop-filter: blur(8px); /* Blurs background */
            -webkit-backdrop-filter: blur(8px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0221.jpeg")

# Title and Content Styling
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5 {{
        color: #18453b; /* Custom color for headings (RGB 24, 69, 59) */
    }}
    p, label, .stTextInput > label {{
        color: #18453b; /* Custom color for text (RGB 24, 69, 59) */
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.8); /* Light, semi-transparent background for content */
        padding: 20px;
        border-radius: 10px;
        color: #18453b; /* Default text color */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# User selection at the start
st.title("Asteroid Impact Prediction App")
user_type = st.selectbox(
    "Who are you?",
    ["Select User Type", "Public User", "Official User"]
)

# File Paths (Adjust as needed)
CLEANED_ORBIT_FILE = "cleaned_Asteroid_orbit.csv"
IMPACT_FILE = "impacts.csv"
MODELS_DIR = "h5_Files"
MODEL_NAME = "Asteroid_Impact_Model.h5"
OFFICIAL_FILES = {
    "Impact Analysis": "Impact_Analysis.ipynb",
    "Orbits Analysis": "Orbits_Analysis.ipynb",
    "Orbits vs Impacts Analysis": "Orbits_vs_Impacts.ipynb"
}

# Function to Load Data
@st.cache
def load_data():
    orbit_data = pd.read_csv(CLEANED_ORBIT_FILE)
    impact_data = pd.read_csv(IMPACT_FILE)
    return orbit_data, impact_data

# Function to Load Model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error(f"Model file '{MODEL_NAME}' not found in '{MODELS_DIR}'!")
        st.stop()

# Random Location Generator
def generate_random_location():
    latitude
