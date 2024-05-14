import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, time
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Set the page configuration
st.set_page_config(layout="wide", page_title="Building Energy Consumption Predictor")

# Using HTML and CSS to adjust the spacing, alignment, and responsiveness
st.markdown("""
<style>
    @media (min-width: 800px) {
        .title {
            font-size: 32px; /* Larger size when there is enough space */
        }
    }
    @media (max-width: 799px) {
        .title {
            font-size: 24px; /* Smaller size when the sidebar is open or screen is smaller */
        }
    }
    .title {
        text-align: center;
        color: black;
        margin-top: -20px; /* Adjust this value to reduce the space */
        margin-bottom: -30px; /* Reduce space between subtitle and image */
    }
    .subtitle {
        text-align: center;
        color: grey;
        font-size: 18px; /* Adjust font size as needed */
        margin-top: -10px; /* Adjust this value to reduce the space */
        margin-bottom: -10px; /* Reduce space between subtitle and image */
    }
    .img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%; /* Adjust the width as needed */
    }
</style>
""", unsafe_allow_html=True)

# Title with larger font and styling to appear as if being lifted
st.markdown("<h1 class='title'>⚡ Building Energy Consumption Predictor ⚡</h1>", unsafe_allow_html=True)

# Adding a subtitle or description below the title
st.markdown("<div class='subtitle'>Optimize energy usage with your eco-conultant friend.</div>", unsafe_allow_html=True)

# Use columns to position the Superman image just below the middle of the title
col1, col2, col3 = st.columns([1,0.5,1])

with col2:
    st.image("superman.png", width=160)  # Adjust the width as needed

# Define other elements of the sidebar or main page below this
st.sidebar.header('Enter the features')

# Load your model
model = joblib.load('model_with_imputation1.pkl')  # Ensure the model includes preprocessing

# List of unique campus buildings obtained from your dataset
unique_campus_buildings = ['14', '16', '115', '116', '117', '118', '120', '123', '125', '126', '127', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '140', '142', '144', '149', '151', '154', '156', '158', '159', '160', '161', '162', '163', '164', '150', '139', '17', '124', '157', '28', '29', '127', '125', '120', '247', '248', '322', '352', '353', '210', '211', '212', '213', '214']

# Function to handle preprocessing
def preprocess(input_data):
    return input_data  # This should reflect the actual preprocessing used




st.sidebar.header('Enter the features')

# Sidebar Inputs
# Ensure all values are properly formatted with their units
campus_building = st.sidebar.selectbox('Campus Building ID', unique_campus_buildings)
built_year = st.sidebar.number_input('Built Year', 1899, 2019, 1967)
gross_floor_area = st.sidebar.number_input('Gross Floor Area (ft²)', 4250, 5459749, 145558)
room_area = st.sidebar.number_input('Room Area (ft²)', 253, 15176, 1788)  # Added ft² for consistency
capacity = st.sidebar.number_input('Capacity (people)', 0, 1595, 79)  # Added people for clarity
apparent_temperature = st.sidebar.slider('Apparent Temperature (°C)', -7.0, 42.4, 16.0)
air_temperature = st.sidebar.slider('Air Temperature (°C)', -3.0, 44.4, 15.9)
dew_point_temperature = st.sidebar.slider('Dew Point Temperature (°C)', -6.0, 23.6, 13.6)
relative_humidity = st.sidebar.slider('Relative Humidity (%)', 7, 100, 86)  # Added % for clarity
wind_speed = st.sidebar.slider('Wind Speed (km/h)', 0.0, 63.0, 5.4)
wind_direction = st.sidebar.number_input('Wind Direction (degrees)', 0, 359, 134)
is_holiday = st.sidebar.selectbox('Is it a Holiday?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_semester = st.sidebar.selectbox('Is it a Semester?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_exam = st.sidebar.selectbox('Is it an Exam period?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
date = st.sidebar.date_input("Date", datetime(2018, 1, 1))
time = st.sidebar.time_input("Time", time(0, 15))
timestamp = datetime.combine(date, time)
minute = timestamp.minute
hour = timestamp.hour
day = timestamp.day
day_of_week = timestamp.weekday()
month = timestamp.month
year = timestamp.year
category = st.sidebar.selectbox('Category', ['mixed use', 'other', 'residence', 'office', 'teaching', 'sport', 'library'], index=0)

# Button for prediction
if st.sidebar.button('Predict Consumption'):
    # Create DataFrame from input
    features = pd.DataFrame([[campus_building, built_year, gross_floor_area, room_area, capacity,
                              apparent_temperature, air_temperature, dew_point_temperature,
                              relative_humidity, wind_speed, wind_direction, is_holiday, is_semester,
                              is_exam, minute, hour, day_of_week, month, year, category]],
                            columns=['campus_building', 'built_year', 'gross_floor_area',
                                     'room_area', 'capacity', 'apparent_temperature', 'air_temperature',
                                     'dew_point_temperature', 'relative_humidity', 'wind_speed',
                                     'wind_direction', 'is_holiday', 'is_semester', 'is_exam', 'minute',
                                     'hour', 'day_of_week', 'month', 'year', 'category'])

    # Display key feature values
    # CSS to inject
    css_style = """
    <style>
        .feature-box {
            margin: 10px 0px;
            padding: 20px;
            border-radius: 10px;
            background-color: #f1f1f1;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        }
        .feature-header {
            color: #2c3e50;
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }
        .feature-content {
            font-size: 18px;
            color: #333;
            margin: 5px 0px;
        }
        .feature-content strong {
            color: #2c3e50;
        }
    </style>
    """
    # CSS with updated animation definitions
    css_animation = """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .animated {
            opacity: 0;  # Start with invisible
            animation-name: fadeIn;
            animation-duration: 1s;
            animation-fill-mode: forwards;  # Keeps the element visible after animation
        }
        .visible {
            opacity: 1;  # Ensure it stays visible after first animation
        }
    </style>
    """ 

    # Inject CSS
    st.markdown(css_style, unsafe_allow_html=True)
    # Display styled key feature values
    st.markdown(f"""
    <div class="feature-box">
        <div class="feature-header">Input Features for Prediction</div>
        <p class="feature-content"><strong>Date and Time:</strong> {date.strftime('%Y-%m-%d')} at {time.strftime('%H:%M')}</p>
        <p class="feature-content"><strong>Building ID:</strong> {campus_building}</p>
        <p class="feature-content"><strong>Temperature:</strong> Apparent {apparent_temperature}°C, Air {air_temperature}°C</p>
        <p class="feature-content"><strong>Humidity:</strong> {relative_humidity}%</p>
        <p class="feature-content"><strong>Wind:</strong> {wind_speed} km/h from {wind_direction}°</p>
        <p class="feature-content"><strong>Other Conditions:</strong> {'Holiday' if is_holiday else 'Not a holiday'}, {'Semester' if is_semester else 'No semester'}, {'Exam period' if is_exam else 'No exams'}</p>
    </div>
    """, unsafe_allow_html=True)



    # CSS and HTML setup
    css_style = """
    <style>
        .prediction-title {
            display: flex;  # Enables flexbox
            align-items: center;  # Vertically centers the items in the container
            justify-content: center;  # Horizontally centers the content
            text-align: center;
        }
        .blinking {
            animation: blink-animation 1.5s steps(5, start) infinite;
            -webkit-animation: blink-animation 1.5s steps(5, start) infinite;
            color: gold;
            font-size: 24px;  # Adjust this if the icon size doesn't match the text size
            margin-right: 5px;  # Adjust spacing between the icon and the text
            vertical-align: middle;  # Helps align the icon with the text
        }
        h2 {
            margin: 0;
            white-space: nowrap;  # Prevents the text from wrapping
        }
        @keyframes blink-animation {
            to {
                visibility: hidden;
            }
        }
        @-webkit-keyframes blink-animation {
            to {
                visibility: hidden;
            }
        }
    </style>
    """

    # Inject CSS
    st.markdown(css_style, unsafe_allow_html=True)

    # Predict consumption
    processed_features = preprocess(features)
    prediction = model.predict(processed_features)

    # Display title and prediction with the blinking emoji
    st.markdown(f"""
    <div class="prediction-title">
    <span class="blinking">⚡</span><h2 style="display: inline;">Predicted Energy Consumption: {prediction[0]:.3f} kWh</h2>
    </div>

    """, unsafe_allow_html=True)


    # # Predict consumption
    # processed_features = preprocess(features)
    # prediction = model.predict(processed_features)

    # # Display title and prediction with the blinking emoji
    # st.markdown(f"""
    # <div class="prediction-title">
    #     <span class="blinking">⚡</span><h2 style="display: inline; margin-left: 10px;">Predicted Energy Consumption: {prediction[0]:.3f} kWh</h2>
    # </div>
    # """, unsafe_allow_html=True)

    # Optional: Add the feature importance visualization code below
    # Feature importance visualization (Optional)
    # Visualize the importance of features if necessary
    plt.show()
    # Feature importance visualization (Normalized ratios)
    max_values = {
        'gross_floor_area': 5459749,
        'room_area': 15176,
        'capacity': 1595,
        'apparent_temperature': 42.4,
        'air_temperature': 44.4,
        'dew_point_temperature': 23.6,
        'relative_humidity': 100,
        'wind_speed': 63.0,
        'wind_direction': 359,
        
    }
    relevant_features = features[['gross_floor_area', 'room_area', 'capacity', 'apparent_temperature',
                                  'air_temperature', 'dew_point_temperature', 'relative_humidity',
                                  'wind_speed', 'wind_direction']]
    ratios = relevant_features.iloc[0] / pd.Series(max_values)
    z = ratios.sum()
    normalized_ratios = 100*ratios / z

    
    fig = px.bar(
    normalized_ratios,
    x=normalized_ratios.index,
    y=normalized_ratios.values,  # explicitly referring to the values
    title="Feature Contributions to Predicted Energy Consumption",
    labels={"index": "Features", "y": "Feature Contribution %"}  # Changing 'value' to 'y'
    )
    fig.update_layout(
    transition_duration=500,
    yaxis_title="Normalized Feature Contribution %"  # Explicitly setting y-axis title
    )
    st.plotly_chart(fig, use_container_width=True)


    # # Assuming 'normalized_ratios' is calculated as before
    # fig = px.bar(
    #     normalized_ratios,
    #     x=normalized_ratios.index,
    #     y=normalized_ratios,
    #     title="Normalized Feature Contributions to Predicted Energy Consumption",
    #     labels={"index": "Features", "value": "Normalized Feature Contribution %"}
    # )
    # fig.update_layout(transition_duration=500)  # adds a simple animation

    # st.plotly_chart(fig, use_container_width=True)



    # plt.figure(figsize=(10, 4))
    # plt.bar(normalized_ratios.index, normalized_ratios, color='lightblue')
    # plt.xticks(rotation=90)
    # plt.ylabel('Normalized Feature Contribution %')
    # plt.title('Normalized Feature Contributions to Predicted Energy Consumption')
    # st.pyplot(plt.gcf())
