"""
app.py - Streamlit web app for customer churn prediction.

Lets users fill in customer details and see if they're likely to churn.
Design kept intentionally simple — this is a demo, not a product UI.

Run with: streamlit run app/app.py
(from project root)
"""
import os

import sys
import streamlit as st
import pandas as pd
import numpy as np

# Make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_model
from src.feature_engineering import add_features

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "churn_model.pkl"
)


@st.cache_resource
def load_churn_model():
    return load_model(MODEL_PATH)


def build_input_df(inputs: dict) -> pd.DataFrame:
    """Convert form inputs into the dataframe format the model expects."""
    return pd.DataFrame([inputs])


# --- Page config
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Customer Churn Predictor")
st.markdown(
    "Fill in a customer's details below to see whether they're likely to churn. "
    "The model was trained on the IBM Telco dataset (~7k customers)."
)
st.divider()

# --- Load model (cached)
try:
    model = load_churn_model()
except FileNotFoundError:
    st.error("❌ Model not found. Please run `python main.py` first to train the model.")
    st.stop()

# --- Input form
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.subheader("Billing")
col3, col4 = st.columns(2)
with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
with col4:
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)

# Estimate TotalCharges from tenure × monthly (reasonable approximation)
total_charges = round(tenure * monthly_charges, 2)
st.caption(f"Estimated Total Charges: **${total_charges:,.2f}** (tenure × monthly)")

st.divider()

# --- Predict button
if st.button("🔍 Predict Churn", type="primary", use_container_width=True):

    raw_input = {
        "customerID": "streamlit-user",
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = build_input_df(raw_input)
    input_df = add_features(input_df)

    # Drop customerID before passing to model
    input_df = input_df.drop(columns=["customerID"])

    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN**")
        st.metric("Churn Probability", f"{proba:.1%}")
        st.markdown("""
        **Suggested actions:**
        - Offer a loyalty discount or contract upgrade
        - Proactively reach out via customer success
        - Check if they have unresolved support tickets
        """)
    else:
        st.success(f"✅ **This customer is unlikely to churn**")
        st.metric("Churn Probability", f"{proba:.1%}")
        st.markdown("Customer appears stable. Continue monitoring monthly charges and service satisfaction.")

    # Show a simple probability bar
    st.progress(float(proba), text=f"Churn risk: {proba:.1%}")

st.divider()
st.caption("Model: Telco Customer Churn Dataset · IBM Watson Analytics Sample Data")
