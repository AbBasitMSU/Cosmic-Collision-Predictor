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
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
import streamlit.components.v1 as components

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
    except requests.exceptions.RequestException as e:
        st.error(f"File not found: {filename}. Please ensure that the file exists locally or on GitHub.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

# Random Location Generator
def generate_random_location():
    latitude = round(random.uniform(-90, 90), 6)
    longitude = round(random.uniform(-180, 180), 6)
    return latitude, longitude

# Public User Section
def public_user_section():
    st.sidebar.header("Learn About Asteroids")
    if st.sidebar.button("Learn"):
        st.write("### **Here’s some information about asteroids, their origin, composition, and typical properties:**")
        # [Information about asteroids here, keeping it the same as your original code]

    st.header("Future Collisions Calendar")
    
    # Define collision prediction dates
    collision_dates = [datetime(2024, 12, 10).date(), datetime(2024, 12, 15).date()]  # Add more collision dates as needed

    # Generate custom HTML for calendar
    calendar_html = """
    <style>
        .calendar-table {
            display: table;
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        .calendar-cell {
            display: table-cell;
            padding: 8px;
            text-align: center;
            vertical-align: middle;
            border: 1px solid #ddd;
        }
        .calendar-cell.green {
            background-color: #d4edda; /* Light green for safe dates */
        }
        .calendar-cell.red {
            background-color: #f8d7da; /* Light red for collision dates */
            font-weight: bold;
        }
    </style>
    <div class="calendar-table">
    """

    # Generate the calendar for December 2024
    days_in_december = [datetime(2024, 12, day).date() for day in range(1, 32)]
    for week_start in range(0, 31, 7):
        calendar_html += "<div class='calendar-row'>"
        for day in days_in_december[week_start:week_start + 7]:
            cell_class = "red" if day in collision_dates else "green"
            calendar_html += f"<div class='calendar-cell {cell_class}'>{day.day}</div>"
        calendar_html += "</div>"
    calendar_html += "</div>"

    # Render the custom HTML calendar
    components.html(calendar_html, height=300)

    # Collision Prediction Details
    selected_date = st.date_input("Choose a Date")

    if selected_date in collision_dates:
        st.write("**Collision Alert!**")
        st.write(f"Date: {selected_date}")
        st.write("Location: Latitude 23.5, Longitude 78.9")
        st.write("Impact Time: 14:30 UTC")
        st.write("Impact Area: 100 km radius")
        st.subheader("Precautions")
        st.write("1. Stay indoors and away from windows.\n2. Stock up on food, water, and essentials.\n3. Follow local government advisories.")
    else:
        st.write(f"No collision predicted on {selected_date}.")

# Function to Load Model
@st.cache_data
def load_model():
    model_path = os.path.join("h5_Files", "Asteroid_Impact_Model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Ensure the file exists in 'h5_Files'.")
        st.stop()
    return tf.keras.models.load_model(model_path)

# Official User Section
def official_user_section():
    # Original official_user_section function here
    pass

# Main Function
def main():
    set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0222.webp")
    st.markdown("<h1 style='text-align: center;'>Cosmic Collision Predictor</h1>", unsafe_allow_html=True)

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
