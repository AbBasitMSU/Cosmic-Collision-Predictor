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
    except requests.exceptions.RequestException as e:
        st.error(f"File not found: {filename}. Please ensure that the file exists locally or on GitHub.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

# Random Location Generator
def generate_random_location():
    latitude = round(random.uniform(-90, 90), 6)
    longitude = round(random.uniform(-180, 180), 6)
    return latitude, longitude

# Function to Load Model
@st.cache_data
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
            st.write("1. Stay indoors and away from windows.\n2. Stock up on food, water, and essentials.\n3. Follow local government advisories.")
        else:
            st.write("No significant collision risk detected based on the provided parameters.")

# Official User Section
def official_user_section():
    st.header(f"Welcome, {st.session_state['username']}")
    st.subheader("Analysis, Training, and Visualization")

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

    st.subheader("Detailed Analysis")
    analysis_choice = st.selectbox("Choose Analysis", ["Impact Analysis", "Orbits Analysis", "Orbits vs Impacts Analysis"])
    
    # Analysis based on selected choice
    if analysis_choice == "Impact Analysis":
        st.write("Performing Impact Analysis...")
        # Simple Analysis Example - Histogram
        impacts_df = load_csv_data("impacts.csv")
        if not impacts_df.empty:
            fig, ax = plt.subplots()
            sns.histplot(impacts_df['Asteroid Magnitude'], ax=ax, bins=20, kde=True)
            ax.set_title("Asteroid Magnitude Distribution")
            ax.set_xlabel("Asteroid Magnitude")
            st.pyplot(fig)

    elif analysis_choice == "Orbits Analysis":
        st.write("Performing Orbits Analysis...")
        # Scatter Plot Example
        orbits_df = load_csv_data("orbits.csv")
        if not orbits_df.empty:
            fig = px.scatter(orbits_df, x='Orbit Eccentricity', y='Orbit Inclination (deg)', color='Object Classification',
                             title="Orbit Eccentricity vs Inclination")
            st.plotly_chart(fig)

    elif analysis_choice == "Orbits vs Impacts Analysis":
        st.write("Performing Orbits vs Impacts Analysis...")
        # Combined Analysis Example
        orbits_df = load_csv_data("orbits.csv")
        impacts_df = load_csv_data("impacts.csv")
        if not orbits_df.empty and not impacts_df.empty:
            comparison_df = pd.DataFrame({
                'Orbit Eccentricity': orbits_df['Orbit Eccentricity'].mean(),
                'Orbit Inclination (deg)': orbits_df['Orbit Inclination (deg)'].mean(),
                'Asteroid Magnitude': impacts_df['Asteroid Magnitude'].mean()
            }, index=[0])
            fig = px.bar(comparison_df, barmode='group', title="Orbits vs Impacts Data Comparison")
            st.plotly_chart(fig)

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
                st.write("""
                1. Stay indoors and away from windows.
                2. Stock up on food, water, and essentials.
                3. Follow local government advisories.
                """)
        else:
            st.write("No significant collision risk detected based on the provided parameters.")

    st.subheader("Train Models")
    if st.button("Train Impact Prediction Model"):
        st.write("Training Impact Prediction Model...")
        # Load training data
        st.write("Original_Datasets:
impacts.csv
orbits.csv
cleaned_Asteroid_orbit.csv <-- csv file created via Jupyter Notebook after data was cleaned prior to creating the ML models (NN model version)
h5_Files
Asteroid_Impact_Model.h5
Asteroid_Impact_Optimization_Model.h5
Asteroid definitions.pptx <-- Powerpoint presentation Intro to the project and definitions of the different columns in the dataset.
Asteroid_Predictions.ipynb <-- File started in Jupyter Notebook for data cleanup prior to Neural Network ML training
Asteroid_Predictions_Colab.ipynb <-- File worked on via Google Colab after cleanup, to train our Neural Network ML model prior to optimization.
Asteroid_Predictions_Optimization_Colab.ipynb <-- File worked on via Google Colab. Optimized version after training our Neural Network ML model.
asteroid-impact-prediction-SL-CFM.ipynb <-- File worked on via Jupyter Notebook for Supervised Learning, with unbalanced data.
asteroid_impact-prediction-SL-OverSample.ipynb <-- File worked on via Jupyter Notebook for SL, with OverSampling of the data.
asteroid-impact-prediction-SL-UnderSample.ipynb <-- File worked on via Jupyter Notebook for SL, with UnderSampling of the data.
cleaned_Asteroid_orbit.csv <-- csv file created via Jupyter Notebook after data was cleaned prior to creating the ML models (NN model version)
Impacts_Analysis.ipynb <-- This file is used to do the preprocessing seperate IDA and EDA on Impacts data file
Orbits_Analysis.ipynb <-- This file is used to do the preprocessing seperate IDA and EDA on Orbits data file
Impacts_vs_Orbits_Analysis.ipynb <-- This file is used to do the preprocessing seperate IDA and EDA on combined file of Impacts data file and Otbits data file after merging
Guide to the Project

Guidelines for the Project

Collaborating with our team to pool knowledge and share ideas
Outline a scope and purpose for our project, utilzing our machine learning skills to analyze,solve, or visualize our findings
Finding reliable data to use for our project, being mindful of copyrights, licenses, or terms of use
Track all processes in Jupyter Notebook used for cleanup, and techniques used for Data Analysis
Present our findings to the class on Presentation Day, with each member of our group taking turns in speaking
Submit the URL of our GitHub repository to be graded
Graduate and attain employment from utilizing our knowlwdge acquired from this class
Processes used

Reading the csv files
Cleaning the data
Normalize and stabalize the data
Splitting the data
Training the Machine Learning models
Neural Network model implementation
Created a different Jupyter notebook with the same cleanup process to test Supervised Learning model
Supervised Learning model implementation
Confusion Matrix and Visualization
Compared observations and searched for improved accuracy for each model.
Accuracy for the Neural Network Model (Pre-optimization and Optimized results)

NN_model_AccuracyComparison

Accuracy for the Supervised Learning Model

Low precision and recall due to imbalance of data classes
SL_model_Unbalanced

Results when over sampling the data
SL_model_OverSampling

Results when under sampling the data
SL_model_UnderSampling

References

References for the data source(s):

Datasets for this project: https://www.kaggle.com/datasets/nasa/asteroid-impacts
References for the column definitions:

https://cneos.jpl.nasa.gov/about/neo_groups.html#:~:text=The%20vast%20majority%20of%20NEOs,%2Dmajor%20axes%20(%20a%20).
https://howthingsfly.si.edu/ask-an-explainer/what-orbit-eccentricity
https://en.wikipedia.org/wiki/Orbital_inclination
https://astronomy.swin.edu.au/cosmos/A/Argument+Of+Perihelion
https://cneos.jpl.nasa.gov/glossary/
https://www.britannica.com/science/mean-anomaly
https://en.wikipedia.org/wiki/Minimum_orbit_intersection_distance#:~:text=Minimum%20orbit%20intersection%20distance%20(MOID,collision%20risks%20between%20astronomical%20objects.
References for code:

Uploading a CSV file to Google Colab:

https://stackoverflow.com/questions/60347596/uploading-csv-file-google-colab
Using the strip() method for white spaces:

https://saturncloud.io/blog/how-to-remove-space-from-columns-in-pandas-a-data-scientists-guide/#:~:text=Using%20the%20str.strip()%20method&text=strip()%20method%20removes%20leading,column%20names%20or%20column%20values
Confusion Matrix Visualization:

https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
Using Keras for Machine Learning:

https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
Learning Rate Scheduler:

https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
https://keras.io/api/callbacks/learning_rate_scheduler/
https://d2l.ai/chapter_optimization/lr-scheduler.html
https://stackoverflow.com/questions/61981929/how-to-change-the-learning-rate-based-on-the-previous-epoch-accuracy-using-keras
https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler
Validation_Split function:

https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work
Activation Functions:

https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
Optimizers:

https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0
https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6
Callbacks:

https://www.kdnuggets.com/2019/08/keras-callbacks-explained-three-minutes.html
https://medium.com/@ompramod9921/callbacks-your-secret-weapon-in-machine-learning-b08ded5678f0
https://www.tensorflow.org/guide/keras/writing_your_own_callbacks
Saving and Loading Models:

https://colab.research.google.com/github/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb
https://stackoverflow.com/questions/64808087/how-do-i-save-files-from-google-colab-to-google-drive
https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory
Image Resources:

ReadMe image is:

Ai generated image
Introduction and Definition of Features in the DataSet Slide images

https://pixabay.com/illustrations/asteroid-space-stars-meteor-1477065/
https://pixabay.com/illustrations/armageddon-apocalypse-earth-2104385/
https://en.wikipedia.org/wiki/Orbital_eccentricity
https://www.sciencedirect.com/topics/physics-and-astronomy/true-anomaly
https://www.researchgate.net/figure/Minimum-Orbital-Intersection-Distance_fig7_36174303
https://pixabay.com/illustrations/asteroid-planet-land-space-span-4376113/
")
        training_data = load_csv_data("cleaned_Asteroid_data.csv")
        # Model training logic would go here
        st.write("Model training complete.")

    st.subheader("Model Evaluation and Documentation")
    if st.button("Evaluate Existing Models"):
        st.write("Evaluating existing models...")
        # Evaluation logic using the saved models
        model = load_model()
        # Create a DataFrame containing training history
        history_df = pd.DataFrame(model.history)

        # Increase the index by 1 to match the number of epochs
        history_df.index += 1

        # Plot the loss
        history_df.plot(y="loss")
        st.plt.show()
        st.write("Model evaluation complete.")

    st.subheader("Check Documentation")
    if st.button("View Documentation"):
        st.write("Displaying documentation for asteroid prediction models...")
        # Display or provide link to documentation
        st.write("Detailed documentation goes here.")

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
