import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Custom CSS for light space-themed background and text styling
st.markdown(
    """
    <style>
    body {
        background-color: #1f2937;  /* Dark space-like background */
        color: #f0f8ff;  /* Light color for better readability */
    }
    .stApp {
        background: linear-gradient(120deg, #1f2937 0%, #3a4b5c 50%, #526d83 100%);
        color: #f0f8ff;
    }
    h1, h2, h3, h4 {
        color: #a0c4ff;  /* Light blue text for headers */
    }
    .css-1d391kg {
        background-color: #526d83 !important;  /* Sidebar background color */
        color: white !important;
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
