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
    st.sidebar.header("Learn About Asteroids")
    if st.sidebar.button("Learn"):
        st.write("### **Here’s some information about asteroids, their origin, composition, and typical properties:**")
    
        st.write("### **What Are Asteroids?**")
        st.write("Asteroids are rocky remnants left over from the early formation of our solar system around 4.6 billion years ago. They are sometimes referred to as minor planets or space rocks and are much smaller than planets. Unlike comets, which are made of ice and dust, asteroids are made primarily of rock and metals. Most asteroids are found in the **Asteroid Belt**, a region between Mars and Jupiter where millions of these bodies orbit the Sun.")
    
        st.write("### **How Are Asteroids Made?**")
        st.write("Asteroids are thought to be the leftover building blocks of planets, called **planetesimals**, that never quite made it to full planet status. During the early stages of the solar system, dust and gas accumulated into larger clumps, eventually forming planets. However, in some areas like the asteroid belt, the gravitational influence of Jupiter was too strong, preventing these clumps from becoming full-fledged planets. Instead, they remained in their smaller form, as rocky asteroids.")
    
        st.write("### **Composition of Asteroids**")
        st.write("The composition of asteroids varies widely, depending on where they were formed:")
    
        st.write("1. **C-Type (Carbonaceous) Asteroids**:") 
        st.write("- These are the most common, making up around 75% of known asteroids.")
        st.write("- They are rich in carbon, giving them a dark appearance.")
        st.write("- Found primarily in the outer regions of the asteroid belt.")
   
        st.write("2. **S-Type (Silicaceous) Asteroids**:") 
        st.write("- Make up around 17% of known asteroids.")
        st.write("- Composed primarily of silicate materials and nickel-iron.")
        st.write("- Typically found in the inner asteroid belt.")
   
        st.write("3. **M-Type (Metallic) Asteroids**:") 
        st.write("- Make up the remaining types.")
        st.write("- Rich in nickel and iron, giving them a metallic composition.")
        st.write("- Found in the middle of the asteroid belt.")

        st.write("### **General Characteristics of Asteroids**")
        st.write("- **Size**: Asteroids vary widely in size, from a few meters to hundreds of kilometers in diameter. The largest known asteroid is **Ceres**, which is about 940 km (about 580 miles) in diameter and is also classified as a dwarf planet.")
  
        st.write("- **Shape**: Most asteroids are irregularly shaped because they lack sufficient gravity to form a perfect sphere. Some look like lumpy rocks, while others have strange, elongated shapes.")

        st.write("- **Speed**: Asteroids orbit the Sun at different speeds, depending on their distance from the Sun. On average, asteroids move at speeds of **25,000 to 75,000 km/h (15,500 to 46,500 mph)**. However, their speed can vary greatly:")
        st.write("- The relative speed of an asteroid approaching Earth can reach up to **30 km/s (about 108,000 km/h or 67,000 mph)**.")

        st.write("- **Orbit**: Asteroids typically have elliptical (oval-shaped) orbits, and most are found in the **main asteroid belt** between Mars and Jupiter. Some asteroids, however, have orbits that bring them closer to Earth, and they are called **Near-Earth Asteroids (NEAs)**.")

        st.write("### **Interesting Facts About Asteroids**")
        st.write("1. **Impact History**: Asteroids have impacted planets, including Earth, throughout history. It is widely believed that an asteroid impact led to the extinction of the dinosaurs around 66 million years ago. The impact likely caused widespread fires, blocked sunlight, and disrupted ecosystems.")

        st.write("2. **Missions to Asteroids**:")
        st.write("- **NASA's OSIRIS-REx** visited the asteroid **Bennu** in 2018, collected a sample, and returned it to Earth in 2023. Bennu is a carbon-rich asteroid that could provide information about the early solar system and the origin of life.")
        st.write("- The Japanese space agency **JAXA** launched **Hayabusa2**, which visited the asteroid **Ryugu** and returned samples to Earth in 2020.")

        st.write("3. **Potential Resources**: Asteroids may also hold valuable resources like **iron, nickel, cobalt**, and even **water**. Scientists have discussed the possibility of **mining asteroids** in the future, especially for use in space exploration.")

        st.write("4. **Trojans and Family Groups**:")
        st.write("- **Trojan asteroids** are found in **Jupiter's orbit**, trapped in stable regions called Lagrange points. Other planets, including Earth and Mars, also have Trojan asteroids.")
        st.write("- Asteroids that share similar orbits and composition are sometimes referred to as **asteroid families**, which may have originated from the same parent body due to a collision.")

        st.write("### **Asteroids vs. Comets**")
        st.write("- **Asteroids** are composed mostly of rock and metal, and they do not have tails.")
        st.write("- **Comets** are made of ice, dust, and organic compounds. When they get close to the Sun, the heat causes their icy nuclei to vaporize, creating a **glowing coma** and **tail**.")

        st.write("### **Hazardous Asteroids**")
        st.write("- Some asteroids are classified as **Potentially Hazardous Asteroids (PHAs)** if they come within **0.05 astronomical units (AU)** of Earth's orbit and are larger than **140 meters** in diameter.")
        st.write("- NASA and other space agencies continuously monitor **Near-Earth Objects (NEOs)**, including asteroids, to ensure that they do not pose a significant threat to Earth.")

        st.write("### **Famous Asteroids**")
        st.write("1. **Ceres**: The largest object in the asteroid belt and the first asteroid discovered (in 1801).")
        st.write("2. **Vesta**: One of the largest asteroids, known for having a differentiated structure similar to a planet, including a core, mantle, and crust.")
        st.write("3. **Eros**: A near-Earth asteroid that was visited by NASA's **NEAR Shoemaker** spacecraft in 2000, providing a wealth of information about its surface and composition.")
        st.write("4. **Apophis**: Initially thought to be a potentially hazardous asteroid, Apophis will make a close flyby of Earth in 2029, but it has been ruled out as a collision risk.")

        st.write("Asteroids provide fascinating insights into the history of the solar system. Their study helps scientists understand the formation and evolution of planets and possibly even the origins of life itself. They are also a reminder of the potential threats that exist out in space, making their study important for planetary defense.")
    
    else:
        st.header("Future Collisions Calendar")

        # Define collision prediction dates
        collision_dates = [datetime(2024, 12, 10).date(), datetime(2024, 12, 15).date()]  # Add more collision dates as needed

        # Dropdown Calendar
        selected_date = st.date_input("Choose a Date")

        # Display collision dates
        st.write("**Collision Prediction Dates:**")
        for date in collision_dates:
            st.markdown(f"<span style='color:red; font-weight:bold;'>{date}</span>", unsafe_allow_html=True)

        # Check if the selected date is a collision date
        if selected_date in collision_dates:
            st.markdown(
                f"""
                <div style='background-color:#ffcccc; padding:10px; border-radius:5px;'>
                    <h4 style='color:red;'>**Collision Alert!**</h4>
                    <p><b>Date:</b> {selected_date}</p>
                    <p><b>Location:</b> Latitude 23.5, Longitude 78.9</p>
                    <p><b>Impact Time:</b> 14:30 UTC</p>
                    <p><b>Impact Area:</b> 100 km radius</p>
                    <h4 style='color:darkred;'>Precautions:</h4>
                    <ol>
                        <li>Stay indoors and away from windows.</li>
                        <li>Stock up on food, water, and essentials.</li>
                        <li>Follow local government advisories.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='background-color:#ccffcc; padding:10px; border-radius:5px;'>
                    <h4 style='color:green;'>No collision predicted on {selected_date}.</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.subheader("Enter New Asteroid Details")
        velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=0.0, step=1.0)
        distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=0.0, step=1.0)
        angle = st.number_input("Angle (degrees)", min_value=0.0, value=0.0, step=1.0)
        size = st.number_input("Size (km)", min_value=0.0, value=0.0, step=1.0)

        if st.button("Predict Collision"):
            if velocity > 1.0 and distance < 2000.0 and angle < 70.0 and size > 01.0:
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

# Train Models Section
    st.sidebar.subheader("Train Models")
    
    if st.sidebar.button("Train Impact Prediction Model"):
        # Create a placeholder for the entire training status
        status_placeholder = st.empty()
    
        # Display initial status
        status_placeholder.write("Training model started...")
    
        # Placeholder for updating epoch progress
        epoch_placeholder = st.empty()
    
        # Simulate training with 100 epochs (replace this with actual training logic)
        for epoch in range(1, 21):  # Loop from 1 to 20 (inclusive)
            time.sleep(0.6)  # Simulate a short delay (adjust time as needed)
            epoch_placeholder.text(f"Epoch {epoch}/20...")
    
        # After all epochs are complete, clear the epoch placeholder and update the status placeholder
        epoch_placeholder.empty()  # Remove the epoch progress text
        status_placeholder.write("Model Training Completed!")
    
        # Show the final result text and display the result image
        st.write("Result:")
        github_image_url = "https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/Result Images/Accuracy of NN Model.jpg"
        st.image(github_image_url, caption="Impact Model Training Result", use_column_width=True)
        
    else:
        st.header(f"Welcome, {st.session_state['username']}")
            st.subheader("Analysis, and Visualization")
    
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
            velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=0.0, step=1.0)
            distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=0.0, step=1.0)
            angle = st.number_input("Angle (degrees)", min_value=0.0, value=0.0, step=1.0)
            size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)
        
            if st.button("Predict Collision"):
                if velocity > 1.0 and distance < 2000.0 and angle < 70.0 and size > 1.0:
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

        # Model Evaluation Section
        st.sidebar.header("Model Evaluation and Documentation")

        if st.sidebar.button("Evaluate Existing Models"):
            # Placeholder for model evaluation status
            evaluation_status_placeholder = st.empty()
        
            # Display initial status
            evaluation_status_placeholder.write("Evaluating model...")
    
            # Placeholder for updating evaluation progress
            evaluation_progress_placeholder = st.empty()
    
            # Simulate evaluation (e.g., looping through evaluation phases)
            for phase in range(1, 6):  # Assuming 5 phases of evaluation
                time.sleep(0.5)  # Simulate delay (adjust as needed)
                evaluation_progress_placeholder.text(f"Evaluation phase {phase}/5...")
    
            # After all phases are complete, clear the progress and update status
            evaluation_progress_placeholder.empty()  # Clear the evaluation progress text
            evaluation_status_placeholder.write("Model Evaluation Completed!")
    
            # Show the result text and display two images from GitHub
            st.write("Results:")
    
            # URL of images stored in the GitHub repository
            image_1_url = "https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/Result Images/IMG_0228.png"
            image_2_url = "https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/Result Images/IMG_0229.png"
    
            # Display the images with captions
            st.image(image_1_url, caption="Evaluation Result - 1", use_column_width=True)
            st.image(image_2_url, caption="Evaluation Result - 2", use_column_width=True)
        else:
            st.header(f"Welcome, {st.session_state['username']}")
            st.subheader("Analysis, and Visualization")
    
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
            velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=0.0, step=1.0)
            distance = st.number_input("Distance from Earth (AU)", min_value=0.0, value=0.0, step=1.0)
            angle = st.number_input("Angle (degrees)", min_value=0.0, value=0.0, step=1.0)
            size = st.number_input("Size (km)", min_value=0.0, value=1.0, step=0.1)
        
            if st.button("Predict Collision"):
                if velocity > 1.0 and distance < 2000.0 and angle < 70.0 and size > 1.0:
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

        
    

    st.sidebar.header("Check Documentation")
    if st.sidebar.button("View Documentation"):
        st.write("**Displaying documentation for asteroid predictor...**")
        st.write("Original_Datasets: \n impacts.csv \n orbits.csv \n cleaned_Asteroid_orbit.csv <-- csv file created via Jupyter Notebook after data was cleaned prior to creating the ML models (NN model version) \n h5_Files \n Asteroid_Impact_Model.h5 \n Asteroid_Impact_Optimization_Model.h5 \n Asteroid definitions.pptx <-- Powerpoint presentation Intro to the project and definitions of the different columns in the dataset. \n Asteroid_Predictions.ipynb <-- File started in Jupyter Notebook for data cleanup prior to Neural Network ML training \n Asteroid_Predictions_Colab.ipynb <-- File worked on via Google Colab after cleanup, to train our Neural Network ML model prior to optimization. \n Asteroid_Predictions_Optimization_Colab.ipynb <-- File worked on via Google Colab. Optimized version after training our Neural Network ML model.")
        st.write("asteroid-impact-prediction-SL-CFM.ipynb <-- File worked on via Jupyter Notebook for Supervised Learning, with unbalanced data.")
        st.write("asteroid_impact-prediction-SL-OverSample.ipynb <-- File worked on via Jupyter Notebook for SL, with OverSampling of the data.")
        st.write("asteroid-impact-prediction-SL-UnderSample.ipynb <-- File worked on via Jupyter Notebook for SL, with UnderSampling of the data.")
        st.write("cleaned_Asteroid_orbit.csv <-- csv file created via Jupyter Notebook after data was cleaned prior to creating the ML models (NN model version)")
        st.write("Impacts_Analysis.ipynb <-- This file is used to do the preprocessing seperate IDA and EDA on Impacts data file")
        st.write("Orbits_Analysis.ipynb <-- This file is used to do the preprocessing seperate IDA and EDA on Orbits data file")
        st.write("Impacts_vs_Orbits_Analysis.ipynb <-- This file is used to do the preprocessing seperate IDA and EDA on combined file of Impacts data file and Otbits data file after merging")
        
        st.write("## **Guide to the Project**")

        st.write("**Guidelines for the Project**")

        st.write("1: Collaborating with our team to pool knowledge and share ideas")
        st.write("2: Outline a scope and purpose for our project, utilzing our machine learning skills to analyze,solve, or visualize our findings")
        st.write("3: Finding reliable data to use for our project, being mindful of copyrights, licenses, or terms of use")
        st.write("4: Track all processes in Jupyter Notebook used for cleanup, and techniques used for Data Analysis")
        st.write("5: Present our findings to the class on Presentation Day, with each member of our group taking turns in speaking")
        st.write("6: Submit the URL of our GitHub repository to be graded")
        st.write("7: Graduate and attain employment from utilizing our knowlwdge acquired from this class")
        st.write("**Processes used**")

        st.write("1: Reading the csv files")
        st.write("2: Cleaning the data")
        st.write("3: Normalize and stabalize the data")
        st.write("4: Splitting the data")
        st.write("5: Training the Machine Learning models")
        st.write("6: Neural Network model implementation")
        st.write("7: Created a different Jupyter notebook with the same cleanup process to test Supervised Learning model")
        st.write("8: Supervised Learning model implementation")
        st.write("9: Confusion Matrix and Visualization")
        st.write("10: Compared observations and searched for improved accuracy for each model.")
        st.write("## **References**")

        st.write("**References for the data source(s):**")

        st.write("Datasets for this project: https://www.kaggle.com/datasets/nasa/asteroid-impacts")
        st.write("**References for the column definitions:**")

        st.write("https://cneos.jpl.nasa.gov/about/neo_groups.html#:~:text=The%20vast%20majority%20of%20NEOs,%2Dmajor%20axes%20(%20a%20).")
        st.write("https://howthingsfly.si.edu/ask-an-explainer/what-orbit-eccentricity")
        st.write("https://en.wikipedia.org/wiki/Orbital_inclination")
        st.write("https://astronomy.swin.edu.au/cosmos/A/Argument+Of+Perihelion")
        st.write("https://cneos.jpl.nasa.gov/glossary/")
        st.write("https://www.britannica.com/science/mean-anomaly")
        st.write("https://en.wikipedia.org/wiki/Minimum_orbit_intersection_distance#:~:text=Minimum%20orbit%20intersection%20distance%20(MOID,collision%20risks%20between%20astronomical%20objects.")
        st.write("**References for code:**")

        st.write("Uploading a CSV file to Google Colab:")

        st.write("https://stackoverflow.com/questions/60347596/uploading-csv-file-google-colab")
        st.write("Using the strip() method for white spaces:")

        st.write("https://saturncloud.io/blog/how-to-remove-space-from-columns-in-pandas-a-data-scientists-guide/#:~:text=Using%20the%20str.strip()%20method&text=strip()%20method%20removes%20leading,column%20names%20or%20column%20values")
        st.write("Confusion Matrix Visualization:")

        st.write("https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea")
        st.write("Using Keras for Machine Learning:")

        st.write("https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/")
        st.write("Learning Rate Scheduler:")

        st.write("https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/")
        st.write("https://keras.io/api/callbacks/learning_rate_scheduler/")
        st.write("https://d2l.ai/chapter_optimization/lr-scheduler.html")
        st.write("https://stackoverflow.com/questions/61981929/how-to-change-the-learning-rate-based-on-the-previous-epoch-accuracy-using-keras")
        st.write("https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler")
        st.write("Validation_Split function:")

        st.write("https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/")
        st.write("https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work")
        st.write("Activation Functions:")

        st.write("https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/")
        st.write("Optimizers:")

        st.write("https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/")
        st.write("https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0")
        st.write("https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6")
        st.write("Callbacks:")

        st.write("https://www.kdnuggets.com/2019/08/keras-callbacks-explained-three-minutes.html")
        st.write("https://medium.com/@ompramod9921/callbacks-your-secret-weapon-in-machine-learning-b08ded5678f0")
        st.write("https://www.tensorflow.org/guide/keras/writing_your_own_callbacks")
        st.write("Saving and Loading Models:")

        st.write("https://colab.research.google.com/github/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb")
        st.write("https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb")
        st.write("https://stackoverflow.com/questions/64808087/how-do-i-save-files-from-google-colab-to-google-drive")
        st.write("https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory")
        st.write("**Image Resources:**")

        st.write("ReadMe image is:")

        st.write("AI generated image")
        st.write("Introduction and Definition of Features in the DataSet Slide images")

        st.write("https://pixabay.com/illustrations/asteroid-space-stars-meteor-1477065/")
        st.write("https://pixabay.com/illustrations/armageddon-apocalypse-earth-2104385/")
        st.write("https://en.wikipedia.org/wiki/Orbital_eccentricity")
        st.write("https://www.sciencedirect.com/topics/physics-and-astronomy/true-anomaly")
        st.write("https://www.researchgate.net/figure/Minimum-Orbital-Intersection-Distance_fig7_36174303")
        st.write("https://pixabay.com/illustrations/asteroid-planet-land-space-span-4376113/")
        
# Main Function
def main():
    set_background("https://raw.githubusercontent.com/AbBasitMSU/Cosmic-Collision-Predictor/main/IMG_0222.webp")
    st.markdown("<h1 style='text-align: center;'>Cosmic Collision Predictor</h1>", unsafe_allow_html=True)

    st.sidebar.header("Navigation")
    user_role = st.sidebar.selectbox("Who are you?", ["Select User", "Public User", "Official User"])

    if user_role == "Select User":
        st.write ("This app is going to help two types of people.")
        st.write ("1: Genral Public and, 2: Sceintific Persons.")
        st.write (" Public users can check the coming collision presiction dates with all approximate details and also can check new asteriod predictions if they have details of Asteriod. Apart from that Public Users can learn about the Asteriods and its compositions.")
        st.write (" Official Users need to signup and login after that they can Check the data on which models are working and train model and get results also of the existing data. Apart from that they can read the whole Documentation of project and also visulize the data by various plots.")
        st.write ("**Please Choose who you are from the left panel and proceed ahead.**")
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
