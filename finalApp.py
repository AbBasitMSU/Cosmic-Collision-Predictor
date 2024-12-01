import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import hashlib
import json
import random
from datetime import datetime
import os

# File to store user credentials
CREDENTIALS_FILE = "users.json"

# Background Image Function
def set_background(image_url):
    """
    Set a background image with CSS styling.
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
    except FileNotFoundError:
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

# Random Location Generator
def generate_random_location():
    latitude = round(random.uniform(-90, 90), 6)
    longitude = round(random.uniform(-180, 180), 6)
    return latitude, longitude

# Function to Load Model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "h5_Files/Asteroid_Impact_Model.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()
    return tf.keras.models.load_model(model_path)

# Public User Section
def public_user_section():
    st.header("Learn About Asteroids")
    st.write("""
    Asteroids are rocky bodies orbiting the Sun. Some come close to Earth and may pose a threat.
    Learn how collisions are predicted and what precautions can be taken.
    """)

    st.subheader("Future Collisions Calendar")
    
    selected_date = st.date_input("Choose a Date")

    # Fake collision data
    if selected_date == datetime(2024, 12, 10).date():
        st.write("**Collision Alert!**")
        st.write(f"Date: {selected_date}")
        st.write("Location: Latitude 23.5, Longitude 78.9")
        st.write("Impact Time: 14:30 UTC")
        st.write("Impact Area: 100 km radius")
        st.subheader("Precautions")
        st.write("""
        1. Stay indoors and away from windows.
        2. Stock up on food, water, and essentials.
        3. Follow local government advisories.
        """)
    st.subheader("Enter New Asteroid Details")
    velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=20.0, step=0.1)
    distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=1.0, step=0.1)
    angle = st.number_input("Angle (degrees)", min_value=0.0, value=45.0, step=0.1)
    size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)

    if st.button("Predict Collision"):
        model = load_model()
        input_data = np.array([[velocity, distance, angle, size]])
        prediction = model.predict(input_data)
        impact_probability = prediction[0][0]
        latitude, longitude = generate_random_location()

        st.write(f"**Impact Probability:** {impact_probability:.2%}")
        st.write(f"**Estimated Impact Location:** Latitude {latitude}, Longitude {longitude}")

# Official User Section
def official_user_section():
    st.header(f"Welcome, {st.session_state['username']}")
    st.subheader("Analysis and Visualization")

    data_choice = st.selectbox(
        "Choose Data to View",
        ["Raw Orbit Data", "Raw Impact Data"]
    )

    if data_choice == "Raw Orbit Data":
        st.write("Orbit data will appear here (placeholder).")
    elif data_choice == "Raw Impact Data":
        st.write("Impact data will appear here (placeholder).")

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

# Main Function
def main():
    # Set the background image
    set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0222.webp")

    # User Role Selection
    user_role = st.sidebar.selectbox("Who are you?", ["Select User","Public User", "Official User"])

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

# Run the app
if __name__ == "__main__":
    main()
