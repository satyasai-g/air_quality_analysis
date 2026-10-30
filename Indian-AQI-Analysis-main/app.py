import streamlit as st
import pandas as pd
import joblib

# Load models
rfclass = joblib.load('aqi_classifier.pkl')
rfreg = joblib.load('aqi_regressor.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Air Quality Index Prediction System")

st.write("Enter the pollutant levels to predict AQI and AQI Bucket.")

# Input fields
pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
no = st.number_input("NO", min_value=0.0, value=10.0)
no2 = st.number_input("NO2", min_value=0.0, value=20.0)
nox = st.number_input("NOx", min_value=0.0, value=30.0)
co = st.number_input("CO", min_value=0.0, value=0.5)
so2 = st.number_input("SO2", min_value=0.0, value=5.0)
o3 = st.number_input("O3", min_value=0.0, value=20.0)

if st.button("Predict"):
    # Prepare input for regression
    input_reg = pd.DataFrame([[pm25, no, no2, nox, co, so2, o3]], columns=['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3'])
    aqi_pred = rfreg.predict(input_reg)[0]
    
    # Prepare input for classification (include predicted AQI)
    input_class = pd.DataFrame([[pm25, no, no2, nox, co, so2, o3, aqi_pred]], columns=['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'AQI'])
    bucket_pred = rfclass.predict(input_class)[0]
    bucket_label = le.inverse_transform([bucket_pred])[0]
    
    st.write(f"Predicted AQI: {aqi_pred:.2f}")
    st.write(f"Predicted AQI Bucket: {bucket_label}")
    
    # AQI categories
    if aqi_pred <= 50:
        st.success("Good air quality")
    elif aqi_pred <= 100:
        st.info("Moderate air quality")
    elif aqi_pred <= 150:
        st.warning("Unhealthy for sensitive groups")
    elif aqi_pred <= 200:
        st.error("Unhealthy")
    elif aqi_pred <= 300:
        st.error("Very unhealthy")
    else:
        st.error("Hazardous")