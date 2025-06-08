"""import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and column list
model = pickle.load(open("best_solar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
required_columns = pickle.load(open("columns.pkl", "rb"))  # list of columns used during training

# UI Inputs
st.title("🔋 Solar Panel Efficiency Predictor")

st.sidebar.header("Enter Sensor & Environment Data")

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 100.0, 25.0)
irradiance = st.sidebar.number_input("Irradiance (W/m²)", 0.0, 1500.0, 800.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 40.0)
panel_age = st.sidebar.number_input("Panel Age (Years)", 0, 40, 5)
maintenance_count = st.sidebar.number_input("Maintenance Count", 0, 100, 3)
soiling_ratio = st.sidebar.number_input("Soiling Ratio", 0.0, 1.0, 0.9)
voltage = st.sidebar.number_input("Voltage (V)", 0.0, 1000.0, 300.0)
current = st.sidebar.number_input("Current (A)", 0.0, 100.0, 5.0)
module_temp = st.sidebar.number_input("Module Temperature (°C)", 0.0, 100.0, 35.0)
cloud_coverage = st.sidebar.number_input("Cloud Coverage (%)", 0.0, 100.0, 10.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)
pressure = st.sidebar.number_input("Pressure (hPa)", 800.0, 1100.0, 1013.0)

string_id = st.sidebar.selectbox("String ID", ['A1', 'D4'])  # match training values
error_code = st.sidebar.selectbox("Error Code", ['E00', 'E01'])  # match training values
installation_type = st.sidebar.selectbox("Installation Type", ['fixed', 'dual-axis'])  # match training

# Create input dictionary
input_data = {
    'temperature': temperature,
    'irradiance': irradiance,
    'humidity': humidity,
    'panel_age': panel_age,
    'maintenance_count': maintenance_count,
    'soiling_ratio': soiling_ratio,
    'voltage': voltage,
    'current': current,
    'module_temperature': module_temp,
    'cloud_coverage': cloud_coverage,
    'wind_speed': wind_speed,
    'pressure': pressure,
    'string_id': string_id,
    'error_code': error_code,
    'installation_type': installation_type
}

input_df = pd.DataFrame([input_data])
df_encoded = pd.get_dummies(input_df)

# Add missing columns & reorder them as during training
for col in required_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[required_columns]



#df_encoded = pd.get_dummies(input_df)

# Add missing columns & reorder
for col in required_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[required_columns]

# Scale and predict
df_scaled = scaler.transform(df_encoded)
prediction = model.predict(df_scaled)

# Output
st.subheader("🔮 Predicted Solar Panel Efficiency:")
st.success(f"{prediction[0]:.2f} %")
                     
    

    

    # Align columns with training model
        # Replace this line:
    # required_columns = model.feature_names_in_

    # With the actual list of columns used during training:
   # Define the exact list of features used during training
   

required_columns =[
    'temperature', 'irradiance', 'humidity', 'panel_age',
    'maintenance_count', 'soiling_ratio', 'voltage', 'current',
    'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure',
    'string_id_A1', 'string_id_B2', 'string_id_C3', 'string_id_D4',
    'error_code_E00', 'error_code_E01', 'error_code_E02',
    'installation_type_dual-axis', 'installation_type_fixed', 'installation_type_tracking'
]

# Add any missing columns that were in training but not in current input
    for col in required_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

# Ensure correct order of columns
    df_encoded = df_encoded[required_columns]


    # Scale numeric data
    df_scaled = scaler.transform(df_encoded)

    # Predict
    prediction = model.predict(df_scaled)[0]

     """
#st.success(f"🔮 Predicted Solar Panel Efficiency: **{prediction:.2f}**")




#Enhanced UI Code


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
#st.set_page_config(page_title="Solar Panel Efficiency Predictor", page_icon="🔋", layout="centered")

# Load model and scaler
model = pickle.load(open("best_solar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
required_columns = pickle.load(open("columns.pkl", "rb"))  # list of required training columns



# --- Main Title ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔋 Solar Panel Efficiency Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sample inputs - replace with st.number_input or st.selectbox as needed
st.markdown("### 📥 Input Solar Panel Parameters:")
temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=100.0, value=25.0)
irradiance = st.number_input("Irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=1000.0)
humidity = st.slider("Humidity (%)", 0, 100, 40)
panel_age = st.number_input("Panel Age (years)", min_value=0, max_value=30, value=2)
maintenance_count = st.slider("Maintenance Count", 0, 20, 3)
soiling_ratio = st.number_input("Soiling Ratio", min_value=0.0, max_value=1.0, value=0.9)
voltage = st.number_input("Voltage (V)", value=36.0)
current = st.number_input("Current (A)", value=8.5)
module_temperature = st.number_input("Module Temperature (°C)", value=45.0)
cloud_coverage = st.slider("Cloud Coverage (%)", 0, 100, 10)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.2)
pressure = st.number_input("Pressure (hPa)", value=1013.0)

string_id = st.selectbox("String ID", ["A1", "B2", "C3", "D4"])
error_code = st.selectbox("Error Code", ["E00", "E01", "E02"])
installation_type = st.selectbox("Installation Type", ["dual-axis", "fixed", "tracking"])

# Create input DataFrame
input_dict = {
    'temperature': temperature,
    'irradiance': irradiance,
    'humidity': humidity,
    'panel_age': panel_age,
    'maintenance_count': maintenance_count,
    'soiling_ratio': soiling_ratio,
    'voltage': voltage,
    'current': current,
    'module_temperature': module_temperature,
    'cloud_coverage': cloud_coverage,
    'wind_speed': wind_speed,
    'pressure': pressure,
    f'string_id_{string_id}': 1,
    f'error_code_{error_code}': 1,
    f'installation_type_{installation_type}': 1,
}

df = pd.DataFrame([input_dict])
df_encoded = pd.get_dummies(df)

# Add any missing columns
for col in required_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0     

# Arrange columns
df_encoded = df_encoded[required_columns]

# Predict
df_scaled = scaler.transform(df_encoded)
prediction = model.predict(df_scaled)[0]

# Show result
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🔮 Predicted Solar Panel Efficiency</h3>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: #2196F3;'>{prediction:.2f} %</h1>", unsafe_allow_html=True)
st.markdown("---")

# Footer
st.markdown("<p style='text-align: center; color: gray;'>⚡ Powered by Machine Learning & Sustainable Tech</p>", unsafe_allow_html=True)
