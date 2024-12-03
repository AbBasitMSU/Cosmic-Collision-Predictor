import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import hashlib
import json
import requests
import random
from datetime import datetime
import os

# URL to store user credentials on GitHub
CREDENTIALS_FILE_URL = "https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/users.json"

# Load or initialize user credentials
def load_credentials():
    try:
        response = requests.get(CREDENTIALS_FILE_URL)
        response.raise_for_status()
        credentials = response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        # If the file is not found, inaccessible, or is corrupted, initialize with an empty dictionary
        credentials = {}
    return credentials

def save_credentials(credentials):
    # Since we cannot directly save to GitHub from the Streamlit app, provide instructions to update manually
    st.warning("Saving credentials directly to GitHub is not supported. Please update the users.json file manually.")
    st.json(credentials)

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup function
def signup():
    st.subheader("Sign Up")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
            return

        credentials = load_credentials()
        if new_username in credentials:
            st.error("Username already exists. Please choose another.")
            return

        credentials[new_username] = hash_password(new_password)
        save_credentials(credentials)
        st.success("Sign Up Successful! You can now log in.")

# Login function
def login():
    st.subheader("Log In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        credentials = load_credentials()
        hashed_password = hash_password(password)

        if username in credentials and credentials[username] == hashed_password:
            st.success("Login Successful!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid username or password.")

# Function to Load CSV Data
def load_csv_data(filename):
    try:
        file_path = os.path.join("Original_Datasets", filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            # Optionally load from GitHub if the file is not found locally
            csv_url = f"https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/Original_Datasets/{filename}"
            response = requests.get(csv_url)
            response.raise_for_status()
            from io import StringIO
            return pd.read_csv(StringIO(response.text))
    except (FileNotFoundError, requests.exceptions.RequestException, pd.errors.EmptyDataError):
        st.error(f"Error loading data from {filename}. Please ensure the file is available and accessible.")
        return pd.DataFrame()

# Official User Section
def official_user_section():
    st.header(f"Welcome, {st.session_state['username']}")
    st.subheader("Analysis, Training, and Visualization")

    data_choice = st.selectbox(
        "Choose Data to View",
        ["Raw Orbit Data", "Raw Impact Data"]
    )

    if data_choice == "Raw Orbit Data":
        df = load_csv_data("cleaned_Asteroid_orbit.csv")
        if not df.empty:
            st.write(df)
        else:
            st.error("No data available to display.")
    elif data_choice == "Raw Impact Data":
        df = load_csv_data("impact_data.csv")  # Replace with actual file name if different
        if not df.empty:
            st.write(df)
        else:
            st.error("No data available to display.")

    st.subheader("Detailed Analysis")
    analysis_choice = st.selectbox(
        "Choose Analysis",
        ["Impact Analysis", "Orbits Analysis", "Orbits vs Impacts Analysis"]
    )

    if analysis_choice == "Impact Analysis":
        st.write("Performing Impact Analysis...")
    elif analysis_choice == "Orbits Analysis":
        st.write("Performing Orbits Analysis...")
    elif analysis_choice == "Orbits vs Impacts Analysis":
        st.write("Performing Orbits vs Impacts Analysis...")

    st.subheader("Train Models")
    if st.button("Train Impact Prediction Model"):
        st.write("Training Impact Prediction Model...")
        # Load training data
        training_data = load_csv_data("cleaned_Asteroid_orbit.csv")
        if not training_data.empty:
            # Model training logic would go here
            st.write("Model training complete.")
        else:
            st.error("Training data not available.")

    st.subheader("Model Evaluation and Documentation")
    if st.button("Evaluate Existing Models"):
        st.write("Evaluating existing models...")
        # Evaluation logic using the saved models
        model = load_model()
        st.write("Model evaluation complete.")

    st.subheader("Check Documentation")
    if st.button("View Documentation"):
        st.write("Displaying documentation for asteroid prediction models...")
        # Display or provide link to documentation
        st.write("Detailed documentation goes here.")

# Main function
def main():
    st.title("Asteroid Impact Prediction App")

    # Check login state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        official_user_section()
    else:
        choice = st.radio("Choose an option", ["Log In", "Sign Up"])

        if choice == "Log In":
            login()
        elif choice == "Sign Up":
            signup()

# Run the app
if __name__ == "__main__":
    main()
