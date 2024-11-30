import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf  # For loading .h5 models
import random
import os

# Background Image Function
def set_background(image_url):
    """
    Set a blurry background image in the Streamlit app using custom CSS.
    :param image_url: URL or local path of the image.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            filter: blur(10px); /* Blurs the image */
            -webkit-filter: blur(10px);
        }}
        .content {{
            backdrop-filter: blur(0px); /* Makes content readable */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0220.jpeg")

# Title and Content Styling
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5 {{
        color: #FFD700; /* Gold color for headings */
    }}
    p, label {{
        color: #FFFFFF; /* White text for content */
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.6); /* Adds a transparent dark layer */
        padding: 20px;
        border-radius: 10px;
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
    latitude = round(random.uniform(-90, 90), 6)
    longitude = round(random.uniform(-180, 180), 6)
    return latitude, longitude

# Section: Public User
if user_type == "Public User":
    st.header("Welcome, Public User!")

    # Predict Impact
    st.subheader("Asteroid Impact Prediction")
    st.write("Enter the details of the asteroid below to predict the impact probability.")

    velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=10.0, step=0.1)
    distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=1.0, step=0.1)
    angle = st.number_input("Angle (degrees)", min_value=0.0, value=45.0, step=0.1)
    size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)

    if st.button("Predict Impact"):
        model = load_model()
        input_data = np.array([[velocity, distance, angle, size]])
        prediction = model.predict(input_data)
        impact_probability = prediction[0][0]

        latitude, longitude = generate_random_location()
        st.write(f"**Impact Probability:** {impact_probability:.2%}")
        st.write(f"**Random Estimated Impact Location:** Latitude {latitude}, Longitude {longitude}")

    # About Asteroids
    st.subheader("About Asteroids")
    st.write("""
        Asteroids are rocky bodies orbiting the Sun. While most asteroids remain in the asteroid belt, 
        some pass near Earth, presenting potential risks. If a collision occurs, the impact depends on 
        the asteroid's size, velocity, and angle. Modern technology enables scientists to predict 
        asteroid impacts and take precautions to reduce risks.
    """)

# Section: Official User
elif user_type == "Official User":
    st.header("Welcome, Official User!")

    # Login or Sign Up
    login_signup = st.radio("Do you want to log in or sign up?", ["Log In", "Sign Up"])

    if login_signup == "Log In":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            # Placeholder login logic
            if username == "admin" and password == "admin123":
                st.success("Login Successful!")
            else:
                st.error("Invalid credentials! Please try again.")
    elif login_signup == "Sign Up":
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        if st.button("Sign Up"):
            st.success("Sign Up Successful! Please log in to continue.")

    # After login
    if st.button("Proceed to Dashboard (Demo Login Required)"):
        # Predict Impact
        st.subheader("Asteroid Impact Prediction")
        velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=10.0, step=0.1)
        distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=1.0, step=0.1)
        angle = st.number_input("Angle (degrees)", min_value=0.0, value=45.0, step=0.1)
        size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)

        if st.button("Predict Impact (Official)"):
            model = load_model()
            input_data = np.array([[velocity, distance, angle, size]])
            prediction = model.predict(input_data)
            impact_probability = prediction[0][0]
            st.write(f"Impact Probability: {impact_probability:.2%}")

        # Data Analysis
        st.subheader("Data Analysis")
        orbit_data, impact_data = load_data()
        st.write("### Orbit Data")
        st.write(orbit_data.head())
        st.write("### Impact Data")
        st.write(impact_data.head())

        # View Analysis Files
        st.subheader("Run Analysis and View Files")
        analysis_file = st.selectbox("Select Analysis File", list(OFFICIAL_FILES.keys()))
        st.write(f"Selected File: {OFFICIAL_FILES[analysis_file]}")

        if st.button("Run Analysis"):
            st.write(f"Running analysis for {OFFICIAL_FILES[analysis_file]}... (This is a placeholder)")

# Default Selection
else:
    st.info("Please select your user type to proceed.")
