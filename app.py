# ============================================================
# IRRIGATION REQUIREMENT CLASSIFICATION - STREAMLIT APP
# Learn Depth Machine Learning Internship - Track 1
# ============================================================

# 1. IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Irrigation Requirement Prediction",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# 3. LOAD DATASET AND TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("irrigation_model.pkl")
    return model


@st.cache_data
def load_data():
    data = pd.read_csv("irrigation_capstone_10000.csv")
    return data


model = load_model()
df = load_data()


# ============================================================
# 4. TITLE AND INTRODUCTION
# ============================================================

st.title("🌱 Irrigation Requirement Classification")

st.write(
    """
    This Machine Learning application predicts the irrigation
    requirement of a field based on soil, weather, crop and
    irrigation-related information.
    """
)

st.info(
    "The model predicts one of three irrigation requirement levels: "
    "Low, Medium, or High."
)


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.header("Project Information")

st.sidebar.write(
    """
    **Project:** Irrigation Requirement Classification

    **Machine Learning Task:** Classification

    **Model:** Decision Tree

    **Dataset:** Kaggle Irrigation Dataset

    **Output:** Low / Medium / High
    """
)


# ============================================================
# 6. CREATE INPUT FORM
# ============================================================

st.header("Enter Field Information")

with st.form("irrigation_form"):

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # SOIL INFORMATION
    # --------------------------------------------------------

    with col1:

        st.subheader("Soil Information")

        soil_type = st.selectbox(
            "Soil Type",
            sorted(df["Soil_Type"].unique())
        )

        soil_ph = st.number_input(
            "Soil pH",
            min_value=float(df["Soil_pH"].min()),
            max_value=float(df["Soil_pH"].max()),
            value=float(df["Soil_pH"].median()),
            step=0.1
        )

        soil_moisture = st.number_input(
            "Soil Moisture",
            min_value=float(df["Soil_Moisture"].min()),
            max_value=float(df["Soil_Moisture"].max()),
            value=float(df["Soil_Moisture"].median()),
            step=0.1
        )

        organic_carbon = st.number_input(
            "Organic Carbon",
            min_value=float(df["Organic_Carbon"].min()),
            max_value=float(df["Organic_Carbon"].max()),
            value=float(df["Organic_Carbon"].median()),
            step=0.1
        )

        electrical_conductivity = st.number_input(
            "Electrical Conductivity",
            min_value=float(df["Electrical_Conductivity"].min()),
            max_value=float(df["Electrical_Conductivity"].max()),
            value=float(df["Electrical_Conductivity"].median()),
            step=0.1
        )

    # --------------------------------------------------------
    # WEATHER INFORMATION
    # --------------------------------------------------------

    with col2:

        st.subheader("Weather Information")

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=float(df["Temperature_C"].min()),
            max_value=float(df["Temperature_C"].max()),
            value=float(df["Temperature_C"].median()),
            step=0.1
        )

        humidity = st.number_input(
            "Humidity",
            min_value=float(df["Humidity"].min()),
            max_value=float(df["Humidity"].max()),
            value=float(df["Humidity"].median()),
            step=0.1
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=float(df["Rainfall_mm"].min()),
            max_value=float(df["Rainfall_mm"].max()),
            value=float(df["Rainfall_mm"].median()),
            step=0.1
        )

        sunlight = st.number_input(
            "Sunlight Hours",
            min_value=float(df["Sunlight_Hours"].min()),
            max_value=float(df["Sunlight_Hours"].max()),
            value=float(df["Sunlight_Hours"].median()),
            step=0.1
        )

        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=float(df["Wind_Speed_kmh"].min()),
            max_value=float(df["Wind_Speed_kmh"].max()),
            value=float(df["Wind_Speed_kmh"].median()),
            step=0.1
        )


    # ========================================================
    # CROP INFORMATION
    # ========================================================

    st.subheader("Crop Information")

    col3, col4 = st.columns(2)

    with col3:

        crop_type = st.selectbox(
            "Crop Type",
            sorted(df["Crop_Type"].unique())
        )

        crop_growth_stage = st.selectbox(
            "Crop Growth Stage",
            sorted(df["Crop_Growth_Stage"].unique())
        )

        season = st.selectbox(
            "Season",
            sorted(df["Season"].unique())
        )

        region = st.selectbox(
            "Region",
            sorted(df["Region"].unique())
        )

    with col4:

        irrigation_type = st.selectbox(
            "Irrigation Type",
            sorted(df["Irrigation_Type"].unique())
        )

        water_source = st.selectbox(
            "Water Source",
            sorted(df["Water_Source"].unique())
        )

        field_area = st.number_input(
            "Field Area (hectare)",
            min_value=float(df["Field_Area_hectare"].min()),
            max_value=float(df["Field_Area_hectare"].max()),
            value=float(df["Field_Area_hectare"].median()),
            step=0.1
        )

        mulching_used = st.selectbox(
            "Mulching Used",
            sorted(df["Mulching_Used"].unique())
        )

        previous_irrigation = st.number_input(
            "Previous Irrigation (mm)",
            min_value=float(df["Previous_Irrigation_mm"].min()),
            max_value=float(df["Previous_Irrigation_mm"].max()),
            value=float(df["Previous_Irrigation_mm"].median()),
            step=0.1
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    submitted = st.form_submit_button(
        "🔍 Predict Irrigation Requirement"
    )


# ============================================================
# 7. MAKE PREDICTION
# ============================================================

if submitted:

    # Create one-row DataFrame using exactly the
    # feature names used during model training.

    input_data = pd.DataFrame({
        "Soil_Type": [soil_type],
        "Soil_pH": [soil_ph],
        "Soil_Moisture": [soil_moisture],
        "Organic_Carbon": [organic_carbon],
        "Electrical_Conductivity": [electrical_conductivity],
        "Temperature_C": [temperature],
        "Humidity": [humidity],
        "Rainfall_mm": [rainfall],
        "Sunlight_Hours": [sunlight],
        "Wind_Speed_kmh": [wind_speed],
        "Crop_Type": [crop_type],
        "Crop_Growth_Stage": [crop_growth_stage],
        "Season": [season],
        "Irrigation_Type": [irrigation_type],
        "Water_Source": [water_source],
        "Field_Area_hectare": [field_area],
        "Mulching_Used": [mulching_used],
        "Previous_Irrigation_mm": [previous_irrigation],
        "Region": [region]
    })


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(input_data)[0]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.header("Prediction Result")

    if prediction == "Low":

        st.success(
            "🌱 Irrigation Requirement: LOW"
        )

        st.write(
            "The model predicts a low irrigation requirement."
        )

    elif prediction == "Medium":

        st.warning(
            "💧 Irrigation Requirement: MEDIUM"
        )

        st.write(
            "The model predicts a medium irrigation requirement."
        )

    elif prediction == "High":

        st.error(
            "🚨 Irrigation Requirement: HIGH"
        )

        st.write(
            "The model predicts a high irrigation requirement."
        )

    else:

        st.info(
            f"Predicted Irrigation Requirement: {prediction}"
        )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Irrigation Requirement Classification | "
    "Learn Depth Machine Learning Internship - Track 1"
)