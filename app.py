import streamlit as st

from Regression_01.pipeline.stage_06_predictionpipeline import PredictionPipelineTrainingPipeline



# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🏠 House Price Prediction")

st.markdown(
    """
    **Machine Learning Regression Application**

    Enter the property details below to estimate the house price
    using our trained machine learning model.
    """
)

st.divider()


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.subheader("📐 Property Details")

col1, col2, col3 = st.columns(3)

with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        value=3000,
        step=100
    )

with col2:

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

with col3:

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


col4, col5 = st.columns(2)

with col4:

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col5:

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )


# ============================================================
# PROPERTY FEATURES
# ============================================================

st.subheader("🏡 Property Features")

col1, col2, col3 = st.columns(3)

with col1:

    mainroad = st.selectbox(
        "Main Road",
        ["yes", "no"]
    )

with col2:

    guestroom = st.selectbox(
        "Guest Room",
        ["yes", "no"]
    )

with col3:

    basement = st.selectbox(
        "Basement",
        ["yes", "no"]
    )


col4, col5, col6 = st.columns(3)

with col4:

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"]
    )

with col5:

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"]
    )

with col6:

    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"]
    )


# ============================================================
# FURNISHING
# ============================================================

st.subheader("🛋️ Furnishing")

furnishingstatus = st.selectbox(
    "Furnishing Status",
    [
        "furnished",
        "semi-furnished",
        "unfurnished"
    ]
)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

predict_button = st.button(
    "🔮 Predict House Price",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Create Prediction Pipeline
        # ----------------------------------------------------

        prediction_pipeline = (
            PredictionPipelineTrainingPipeline()
        )


        # ----------------------------------------------------
        # Make Prediction
        # ----------------------------------------------------

        prediction = prediction_pipeline.main(
            input_data
        )


        # ----------------------------------------------------
        # Display Result
        # ----------------------------------------------------

        st.success(
            "Prediction completed successfully!"
        )

        st.subheader("💰 Estimated House Price")


        # Convert prediction into lakhs

        price_lakhs = prediction / 100000


        # Display price

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Price",
                f"₹ {prediction:,.0f}"
            )

        with col2:

            st.metric(
                "Price in Lakhs",
                f"₹ {price_lakhs:.2f} Lakhs"
            )


        st.info(
            "This is a machine-learning model estimate based "
            "on the property details provided."
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About the Project")

    st.write(
        """
        This application predicts house prices using a
        machine-learning regression model.
        """
    )

    st.markdown("### 🔧 ML Pipeline")

    st.write(
        """
        • Data Ingestion  
        • Data Validation  
        • Data Transformation  
        • Model Training  
        • Model Evaluation  
        • Prediction Pipeline  
        • Streamlit
        """
    )

    st.markdown("### 🤖 Model")

    st.write(
        "XGBoost Regression"
    )

    st.markdown("### 📊 Evaluation")

    st.write(
        "The models were compared using MAE, RMSE and R²."
    )