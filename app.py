# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Forecaster", layout="wide", initial_sidebar_state="expanded"
)


# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, and data needed for the app."""
    try:
        model = joblib.load("models/energy_forecasting_model.pkl")
        scaler = joblib.load("models/feature_scaler.pkl")
        # Load the original data to get historical context for predictions
        full_data = pd.read_csv("data/energydata_complete.csv", parse_dates=["date"])
    except FileNotFoundError:
        st.error("Model/scaler/data not found. Please run `train.py` first.")
        return None, None, None
    return model, scaler, full_data


model, scaler, full_data = load_artifacts()

# --- Import Feature Engineering Logic ---
# This ensures 100% consistency between training and prediction
from src.feature_engineer import FeatureEngineer


# --- SHAP Explainer Function ---
@st.cache_data
def get_shap_explainer(_model):
    """Creates and caches the SHAP explainer."""
    return shap.TreeExplainer(_model)


if model:
    explainer = get_shap_explainer(model)

# --- App Layout ---
st.title("💡 Interactive Energy Consumption Forecaster")
st.markdown(
    """
This application demonstrates a full machine learning pipeline. It uses a LightGBM model
trained on historical data to predict appliance energy consumption.

**How to use:**
1.  Select a date and time from the test set period in the sidebar.
2.  The app will use the real historical data leading up to that point to generate features.
3.  (Optional) You can override key weather features to see how the forecast changes.
4.  The app will display the prediction, its context, and an explanation of *why* the model made that prediction using SHAP.
"""
)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Forecasting Options")

    # We'll use the last 20% of the data as our interactive test period
    if full_data is not None:
        test_period_start = full_data["date"].iloc[int(len(full_data) * 0.8)]
        test_period_end = full_data["date"].iloc[-1]

        selected_date = st.slider(
            "Select a Date and Time for Prediction",
            min_value=test_period_start.to_pydatetime(),
            max_value=test_period_end.to_pydatetime(),
            value=test_period_start.to_pydatetime(),
            step=pd.Timedelta(minutes=10),
            format="YYYY-MM-DD HH:mm",
        )

        st.subheader("Override Weather Conditions (Optional)")
        override_weather = st.checkbox("Enable Weather Override")
        T_out_override = (
            st.slider("Outside Temp (°C)", -10.0, 40.0, 15.0)
            if override_weather
            else None
        )
        RH_out_override = (
            st.slider("Outside Humidity (%)", 0.0, 100.0, 70.0)
            if override_weather
            else None
        )


# --- Main Panel for Displaying Results ---
if model is not None:
    # Find the data row corresponding to the user's selected time
    input_row_index = full_data[full_data["date"] == selected_date].index[0]

    # Get the full history needed to generate features for that one row
    # We need at least 144 previous rows for the longest lag
    required_history_df = full_data.iloc[
        input_row_index - 144 : input_row_index + 1
    ].copy()

    # Apply overrides if enabled
    if override_weather:
        required_history_df.loc[required_history_df.index[-1], "T_out"] = T_out_override
        required_history_df.loc[required_history_df.index[-1], "RH_out"] = (
            RH_out_override
        )

    # Use the FeatureEngineer class to create features for this slice of data
    feature_engineer = FeatureEngineer(required_history_df.set_index("date"))
    df_featured = feature_engineer.engineer_features()

    # The last row of the result is our single prediction input
    prediction_input = df_featured.iloc[[-1]]

    # Define features and scale
    FEATURES = [col for col in df_featured.columns if col != "Appliances"]
    X_pred = prediction_input[FEATURES]
    X_pred_scaled = scaler.transform(X_pred)

    # Make prediction
    prediction = model.predict(X_pred_scaled)[0]
    actual_value = prediction_input["Appliances"].iloc[0]

    st.header(f"Forecast for: {selected_date.strftime('%Y-%m-%d %H:%M')}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Prediction (Wh)", f"{prediction:.2f}")
    with col2:
        st.metric(
            "Actual Value (Wh)",
            f"{actual_value:.2f}",
            delta=f"{(prediction - actual_value):.2f}",
        )

    # --- Visualization and Explanation ---
    st.subheader("Prediction in Context")

    # Plot the recent history and the prediction
    history_to_plot = required_history_df.set_index("date").iloc[-144:]  # Last 24 hours
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        history_to_plot.index,
        history_to_plot["Appliances"],
        label="Historical Actuals",
        color="cornflowerblue",
    )
    ax.axvline(
        selected_date,
        color="gray",
        linestyle="--",
        label=f'Forecast Time: {selected_date.strftime("%H:%M")}',
    )
    ax.plot(
        selected_date,
        prediction,
        "ro",
        markersize=8,
        label=f"Model Prediction: {prediction:.2f} Wh",
    )
    ax.plot(
        selected_date,
        actual_value,
        "go",
        markersize=8,
        label=f"Actual Value: {actual_value:.2f} Wh",
    )
    ax.set_title("Energy Consumption: 24h History and Forecast")
    ax.set_ylabel("Appliance Energy (Wh)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Why did the model make this prediction?")

    # Generate and display SHAP force plot
    shap_values = explainer.shap_values(X_pred_scaled)
    st.markdown(
        "This SHAP plot shows which features pushed the prediction higher (in red) or lower (in blue)."
    )
    st.pyplot(
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_pred,
            matplotlib=True,
            show=False,
        )
    )
