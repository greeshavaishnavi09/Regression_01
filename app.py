import streamlit as st

from Regression_01.pipeline.stage_06_predictionpipeline import PredictionPipelineTrainingPipeline



# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 House Price Prediction")

st.write(
    "Enter the house details below to predict its price."
)


# --------------------------------------------------
# Input Features
# --------------------------------------------------

st.subheader("House Details")


area = st.number_input(
    "Area (sq ft)",
    min_value=100,
    value=3000,
    step=100
)


bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)


bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)


stories = st.number_input(
    "Number of Stories",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)


mainroad = st.selectbox(
    "Main Road",
    ["yes", "no"]
)


guestroom = st.selectbox(
    "Guest Room",
    ["yes", "no"]
)


basement = st.selectbox(
    "Basement",
    ["yes", "no"]
)


hotwaterheating = st.selectbox(
    "Hot Water Heating",
    ["yes", "no"]
)


airconditioning = st.selectbox(
    "Air Conditioning",
    ["yes", "no"]
)


parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)


prefarea = st.selectbox(
    "Preferred Area",
    ["yes", "no"]
)


furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("Predict House Price"):

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    try:

        # Create Prediction Pipeline
        prediction_pipeline = (
            PredictionPipelineTrainingPipeline()
        )

        # Get Prediction
        prediction = prediction_pipeline.main(input_data)

        # Display Result
        st.success("Prediction completed successfully!")

        st.subheader("Predicted House Price")

        st.metric(
            label="Estimated Price",
            value=f"₹ {prediction:,.0f}"
        )

        # Also show price in lakhs
        price_lakhs = prediction / 100000

        st.write(
            f"**Approximately ₹ {price_lakhs:.2f} Lakhs**"
        )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)