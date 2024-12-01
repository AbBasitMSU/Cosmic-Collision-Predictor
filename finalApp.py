import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import hashlib
import json
import random
from datetime import datetime

# File to store user credentials
CREDENTIALS_FILE = "users.json"

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

# Public User Section
def public_user_learn():
    st.subheader("Learn About Asteroids and Cosmic Collisions")
    st.write("""
    Asteroids are rocky bodies orbiting the Sun. Cosmic collisions occur when these asteroids
    come close to Earth. Learn how such events are predicted and managed.
    """)

def public_user_prediction():
    st.subheader("Collision Prediction Calendar")
    st.write("Explore potential collision dates.")
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

# Official User Section
def official_user_section():
    st.subheader(f"Welcome, {st.session_state['username']}")
    option = st.sidebar.radio("Choose an Option", ["Predict Collision", "Data Analysis and Documentation"])

    if option == "Predict Collision":
        st.subheader("Collision Prediction")
        velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=20.0, step=0.1)
        distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=1.0, step=0.1)
        angle = st.number_input("Angle (degrees)", min_value=0.0, value=45.0, step=0.1)
        size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)

        if st.button("Predict"):
            model = tf.keras.models.load_model("h5_Files/Asteroid_Impact_Model.h5")
            input_data = np.array([[velocity, distance, angle, size]])
            prediction = model.predict(input_data)
            st.write(f"**Collision Probability:** {prediction[0][0]:.2%}")

    elif option == "Data Analysis and Documentation":
        st.subheader("Data Analysis")
        st.write("Raw Data and Analysis Details (placeholders).")

# Main Function
def main():
    # Set the background
    set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0222.webp")

    st.title("Welcome to the Cosmic Collision Prediction App")

    # Step 1: Choose User Type
    if "user_type" not in st.session_state:
        st.session_state["user_type"] = None

    if st.session_state["user_type"] is None:
        user_type = st.selectbox("Select User Type", ["Public User", "Official User"])

        if user_type == "Public User":
            st.session_state["user_type"] = "Public User"
        elif user_type == "Official User":
            st.session_state["user_type"] = "Official User"

    # Public User Flow
    elif st.session_state["user_type"] == "Public User":
        tab = st.sidebar.radio("Navigate", ["Learn", "Look for Next Prediction"])
        if tab == "Learn":
            public_user_learn()
        elif tab == "Look for Next Prediction":
            public_user_prediction()

    # Official User Flow
    elif st.session_state["user_type"] == "Official User":
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False

        if st.session_state["logged_in"]:
            official_user_section()
        else:
            st.write("Sign Up or Log In")
            choice = st.radio("Choose an Option", ["Log In", "Sign Up"])
            if choice == "Log In":
                login()
            elif choice == "Sign Up":
                signup()

# Run the app
if __name__ == "__main__":
    main()
