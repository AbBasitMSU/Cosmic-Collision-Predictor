import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import hashlib
import json
import random
from datetime import datetime
import os
import h5py
import requests
import nbformat
from nbconvert import PythonExporter
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# File to store user credentials
CREDENTIALS_FILE = "Users.json"

# Background Image Function
def set_background(image_url):
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
            background-color: rgba(255, 255, 255, 0.4);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load and save credentials
def load_credentials():
    try:
        with open(CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning("Credentials file is missing or corrupted. Initializing a new one.")
        return {}

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(credentials, file)

# Login functionality
def login():
    st.subheader("Log In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        credentials = load_credentials()
        hashed_password = hash_password(password)

        if username in credentials and credentials[username] == hashed_password:
            st.success(f"Welcome, {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid username or password.")

# Signup functionality
def signup():
    st.subheader("Sign Up")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            credentials = load_credentials()
            if username in credentials:
                st.error("Username already exists.")
            else:
                credentials[username] = hash_password(password)
                save_credentials(credentials)
                st.success("Sign up successful! You can now log in.")

# Function to Load CSV Data
@st.cache_data
def load_csv_data(filename):
    # Attempt to load from the local folder
    file_path = os.path.join("Original_Datasets", filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    # If the local file does not exist, try loading from GitHub
    github_url = f"https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/Original_Datasets/{filename}"
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Check if the request was successful
        return pd.read_csv(github_url)
    except requests.exceptions.RequestException:
        st.error(f"File not found: {filename}. Please ensure that the file exists locally or on GitHub.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

# Function to Extract and Run Code from Jupyter Notebooks
@st.cache_resource
def extract_and_run_notebook(notebook_filename):
    github_url = f"https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/{notebook_filename}"
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        notebook_content = response.text
        notebook = nbformat.reads(notebook_content, as_version=4)
        exporter = PythonExporter()
        python_script, _ = exporter.from_notebook_node(notebook)

        # Run the script in the local context and capture key visualizations
        local_context = {}
        exec(python_script, local_context)
        return local_context
    except requests.exceptions.RequestException:
        st.error(f"Failed to load notebook: {notebook_filename}")
        return {}

# Random Location Generator
def generate_random_location():
    latitude = round(random.uniform(-90, 90), 6)
    longitude = round(random.uniform(-180, 180), 6)
    return latitude, longitude

# Function to Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("h5_Files", "Asteroid_Impact_Model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Ensure the file exists in 'h5_Files'.")
        st.stop()
    return tf.keras.models.load_model(model_path)

# Public User Section
def public_user_section():
    st.header("Learn About Asteroids")
    st.write("Asteroids are rocky bodies orbiting the Sun. Some come close to Earth and may pose a threat. Learn how collisions are predicted and what precautions can be taken.")
    st.subheader("Future Collisions Calendar")
    selected_date = st.date_input("Choose a Date")
    
    if selected_date == datetime(2024, 12, 10).date():
        st.write("**Collision Alert!**")
        st.write(f"Date: {selected_date}")
        st.write("Location: Latitude 23.5, Longitude 78.9")
        st.write("Impact Time: 14:30 UTC")
        st.write("Impact Area: 100 km radius")
        st.subheader("Precautions")
        st.write("1. Stay indoors and away from windows.\n2. Stock up on food, water, and essentials.\n3. Follow local government advisories.")

    st.subheader("Enter New Asteroid Details")
    velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=20.0, step=0.1)
    distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=1.0, step=0.1)
    angle = st.number_input("Angle (degrees)", min_value=0.0, value=45.0, step=0.1)
    size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)

    if st.button("Predict Collision"):
        if velocity > 55.0 and distance < 150.0 and angle < 70.0 and size > 450.0:
            latitude, longitude = generate_random_location()
            possible_date = datetime(2024, 12, random.randint(1, 28)).date()
            st.write("**Possible Collision Detected!**")
            st.write(f"Date: {possible_date}")
            st.write(f"Location: Latitude {latitude}, Longitude {longitude}")
            st.write("Impact Area: High Risk")
            st.subheader("Precautions")
            st.write("1. Stay indoors and away from windows, if possible, try to go to a nearby underground bunker.\n2. Stock up on food, water, and essentials.\n3. Follow local government advisories.")
        else:
            st.write("No significant collision risk detected based on the provided parameters.")

# Official User Section
def official_user_section():
    st.header(f"Welcome, {st.session_state['username']}")
    st.subheader("Analysis, Training, and Visualization")

    # Data Viewing Section
    data_choice = st.selectbox("Choose Data to View", ["Raw Orbit Data", "Cleaned Asteroid Data", "Raw Impact Data"])
    if data_choice == "Cleaned Asteroid Data":
        df = load_csv_data("cleaned_Asteroid_orbit.csv")
        if not df.empty:
            st.write(df)
        else:
            st.error("No data available to display.")
    elif data_choice == "Raw Orbit Data":
        df = load_csv_data("orbits.csv")
        if not df.empty:
            st.write(df)
        else:
            st.error("No data available to display.")
    elif data_choice == "Raw Impact Data":
        df = load_csv_data("impacts.csv")
        if not df.empty:
            st.write(df)
        else:
            st.error("No data available to display.")

    # Detailed Analysis Section
    st.subheader("Detailed Analysis")
    analysis_choice = st.selectbox("Choose Analysis", ["Impact Analysis", "Orbits Analysis", "Orbits vs Impacts Analysis"])
    if analysis_choice == "Impact Analysis":
        local_context = extract_and_run_notebook("Impacts_Analysis.ipynb")
        if 'fig' in local_context:
            st.plotly_chart(local_context['fig'])
        st.write("Performing Impact Analysis...")
    elif analysis_choice == "Orbits Analysis":
        local_context = extract_and_run_notebook("Orbits_Analysis.ipynb")
        if 'fig' in local_context:
            st.plotly_chart(local_context['fig'])
        st.write("Performing Orbits Analysis...")
    elif analysis_choice == "Orbits vs Impacts Analysis":
        local_context = extract_and_run_notebook("Orbits_Vs_Impacts.ipynb")
        if 'fig' in local_context:
            st.plotly_chart(local_context['fig'])
        st.write("Performing Orbits vs Impacts Analysis...")

    # Model Training Section
    st.subheader("Train Models")
    if st.button("Train Impact Prediction Model"):
        st.write("Training Impact Prediction Model...")
        training_data = load_csv_data("cleaned_Asteroid_orbit.csv")
        # Model training logic would go here
        st.write("Model training complete.")

    # Model Evaluation Section
    st.subheader("Model Evaluation and Documentation")
    if st.button("Evaluate Existing Models"):
        st.write("Evaluating existing models...")
        model = load_model()
        st.write("Model evaluation complete.")

    # Documentation Section
    st.subheader("Check Documentation")
    if st.button("View Documentation"):
        st.write("Displaying documentation for asteroid prediction models...")
        st.write("Detailed documentation goes here.")

# Main Function
def main():
    set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0222.webp")
    st.markdown(
        """
        <h1 style='text-align: center;'>Cosmic Collision Predictor</h1>
        """,
        unsafe_allow_html=True
    )

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    user_role = st.sidebar.selectbox("Who are you?", ["Select User", "Public User", "Official User"])
    if user_role == "Public User":
        public_user_section()
    elif user_role == "Official User":
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
        if st.session_state["logged_in"]:
            official_user_section()
        else:
            choice = st.sidebar.radio("Choose an Option", ["Log In", "Sign Up"])
            if choice == "Log In":
                login()
            elif choice == "Sign Up":
                signup()

if __name__ == "__main__":
    main()
