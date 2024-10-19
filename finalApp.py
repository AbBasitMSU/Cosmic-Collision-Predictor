import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Custom CSS for light green-themed background and text styling
st.markdown(
    """
    <style>
    body {
        background-color: #d4f7dc;  /* Light green background */
        color: #1f2937;  /* Darker color for better text readability */
    }
    .stApp {
        background: linear-gradient(120deg, #d4f7dc 0%, #b7e9c8 50%, #a2dab2 100%);
        color: #1f2937;  /* Darker text color for better readability */
    }
    h1, h2, h3, h4 {
        color: #0a4f0f;  /* Dark green text for headers */
    }
    .css-1d391kg {
        background-color: #a2dab2 !important;  /* Sidebar light green color */
        color: #1f2937 !important;  /* Darker text for sidebar readability */
    }
    </style>
    """, unsafe_allow_html=True
)

# App title
st.title("Asteroid Data Analysis with Interactive Features")

# Sidebar for file uploads
st.sidebar.header("Upload your datasets")

# Uploading orbit dataset
orbit_file = st.sidebar.file_uploader("Upload Orbit Dataset (CSV)", type=["csv"])
# Uploading impacts dataset
impacts_file = st.sidebar.file_uploader("Upload Impacts Dataset (CSV)", type=["csv"])

# Sidebar for filtering options
st.sidebar.header("Filter Options")


# Check if both files are uploaded
if orbit_file is not None and impacts_file is not None:
    # Load datasets
    orbit_df = pd.read_csv(orbit_file)
    impacts_df = pd.read_csv(impacts_file)

    # Display Raw Data
    if st.sidebar.checkbox("Show Raw Orbit Data"):
        st.subheader("Orbit Data")
        st.write(orbit_df.head())
        
    if st.sidebar.checkbox("Show Raw Impacts Data"):
        st.subheader("Impacts Data")
        st.write(impacts_df.head())

    # Data Cleaning Section
    st.sidebar.header("Data Cleaning")
    
    if st.sidebar.button("Clean Orbit Data"):
        # Implement your cleaning code here (copy from your existing code)
        orbit_df = orbit_df.copy()  # Copy for safety
        # Data cleaning steps...
        orbit_df.dropna(subset=["Object Name"], inplace=True)
        # More cleaning code...
        
        st.success("Orbit data cleaned!")
    
    if st.sidebar.button("Clean Impacts Data"):
        impacts_df.dropna(inplace=True)  # Implement cleaning for impacts data
        st.success("Impacts data cleaned!")

    # Dropdown for selecting object name from Orbits data
    object_name = st.sidebar.selectbox("Select Object Name", orbit_df['Object Name'].unique())

    # Dropdown to filter data by object classification (add in the sidebar)
    object_classification = st.sidebar.selectbox("Select Object Classification", orbit_df['Object Classification'].unique())

    # Filter data based on selected object name and classification
    filtered_orbit_data = orbit_df[(orbit_df['Object Name'] == object_name) & 
                                   (orbit_df['Object Classification'] == object_classification)]
    

    # Slider for selecting a range of asteroid magnitudes
    mag_min, mag_max = st.sidebar.slider("Select Asteroid Magnitude Range", 
                                          float(impacts_df['Asteroid Magnitude'].min()), 
                                          float(impacts_df['Asteroid Magnitude'].max()), 
                                          (float(impacts_df['Asteroid Magnitude'].min()), float(impacts_df['Asteroid Magnitude'].max())))

    # Check if the 'Hazardous' column exists before adding the filter
    if 'Hazardous' in impacts_df.columns:
        hazardous_status = st.sidebar.selectbox("Select Hazardous Status", impacts_df['Hazardous'].unique())
        # Filter impacts data based on magnitude range and hazardous status
        filtered_impacts_data = impacts_df[(impacts_df['Asteroid Magnitude'] >= mag_min) & 
                                           (impacts_df['Asteroid Magnitude'] <= mag_max) &
                                           (impacts_df['Hazardous'] == hazardous_status)]
    else:
        # Filter impacts data based on magnitude range only (if Hazardous column does not exist)
        filtered_impacts_data = impacts_df[(impacts_df['Asteroid Magnitude'] >= mag_min) & 
                                           (impacts_df['Asteroid Magnitude'] <= mag_max)]

    # Display filtered data
    st.subheader("Filtered Orbit Data")
    st.write(filtered_orbit_data)

    st.subheader("Filtered Impacts Data")
    st.write(filtered_impacts_data)

    # Plot: Asteroid Magnitude Distribution
    st.subheader("Asteroid Magnitude Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_impacts_data['Asteroid Magnitude'], bins=20, kde=True)
    st.pyplot(plt)

    # **NEW**: Interactive Comparison of Orbits and Impacts Data using Plotly
    st.subheader("Interactive Comparison of Orbits vs Impacts")

    # Create a grouped bar chart to compare Orbit and Impacts data
    comparison_df = pd.DataFrame({
        'Orbit Eccentricity': filtered_orbit_data['Orbit Eccentricity'].mean(),
        'Orbit Inclination (deg)': filtered_orbit_data['Orbit Inclination (deg)'].mean(),
        'Asteroid Magnitude': filtered_impacts_data['Asteroid Magnitude'].mean()
    }, index=[0])

    fig = go.Figure(data=[
        go.Bar(name='Orbit Eccentricity', x=['Orbits'], y=comparison_df['Orbit Eccentricity'], marker_color='lightblue'),
        go.Bar(name='Orbit Inclination (deg)', x=['Orbits'], y=comparison_df['Orbit Inclination (deg)'], marker_color='orange'),
        go.Bar(name='Asteroid Magnitude', x=['Impacts'], y=comparison_df['Asteroid Magnitude'], marker_color='green')
    ])

    fig.update_layout(barmode='group', title="Comparison of Orbits vs Impacts Data", xaxis_title="Data Type", yaxis_title="Mean Value")
    st.plotly_chart(fig)

    # **NEW**: Add an interactive scatter plot
    st.subheader("Orbit Eccentricity vs Orbit Inclination (Interactive)")
    fig_scatter = px.scatter(orbit_df, x='Orbit Eccentricity', y='Orbit Inclination (deg)', color='Object Classification',
                             title="Interactive Scatter Plot: Orbit Eccentricity vs Inclination",
                             labels={'Orbit Eccentricity': 'Eccentricity', 'Orbit Inclination (deg)': 'Inclination'})
    st.plotly_chart(fig_scatter)

    # Model Training Button
    if st.sidebar.button("Train Model"):
        # Call your model training code here
        st.success("Model training started!")
        
        # After training, you could display the results
        # model_accuracy = ...  # Get accuracy from model
        # st.write(f"Model Accuracy: {model_accuracy:.2f}")

    # Documentation
    st.markdown("""
    ### Documentation
    This app allows you to analyze asteroid orbit and impact data. 
    - Use the dropdown menus to select object names and classifications from the orbit data.
    - Adjust the slider to filter impacts based on asteroid magnitude.
    - The histogram visualizes the distribution of asteroid magnitudes for the filtered impacts.
    - The bar plot compares mean values of Orbit Eccentricity, Inclination, and Asteroid Magnitude between Orbits and Impacts.
    - Interactive scatter plots allow for detailed analysis of the relationship between Orbit Eccentricity and Inclination.
    """)

else:
    st.warning("Please upload both CSV files to proceed.")
