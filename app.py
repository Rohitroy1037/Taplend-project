from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_loan_model.joblib"
FEATURES_PATH = BASE_DIR / "model_features.joblib"


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        import train_model

        train_model.train_and_save_model()

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


st.set_page_config(page_title="TapLend Risk Underwriting", layout="wide")
st.title("🏦 TapLend Loan Risk Underwriting")
st.caption("Model-based credit review for new loan applications.")


try:
    model, expected_features = load_assets()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model files: {e}")
    st.stop()


st.subheader("Applicant Profile")
with st.form("loan_risk_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        amount_requested = st.number_input("Requested Amount ($)", min_value=100, max_value=10000, value=800, step=50)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=50000, value=2800, step=100)
        monthly_rent = st.number_input("Monthly Rent ($)", min_value=0, max_value=10000, value=600, step=50)
        age = st.number_input("Applicant Age", min_value=18, max_value=80, value=32, step=1)

    with col2:
        residence_rent_or_own = st.selectbox("Housing Status", ["Rent", "Own"])
        direct_deposit = st.checkbox("Direct Deposit", value=True)
        payment_ach = st.checkbox("ACH Repayment", value=True)
        loan_duration = st.slider("Loan Duration (Months)", min_value=1, max_value=24, value=12)
        num_payments = st.number_input("Number of Scheduled Payments", min_value=1, max_value=48, value=12)

    with col3:
        payment_frequency = st.selectbox("Payment Frequency", ["Bi-weekly", "Monthly", "Semi-monthly", "Weekly"])
        raw_l2c_score = st.slider("Alternative Credit Score (L2C)", min_value=50, max_value=850, value=620)
        raw_FICO_bank_card = st.slider("FICO Bank Card Score", min_value=222, max_value=850, value=660)
        how_use_money = st.selectbox("Loan Purpose", ["Other", "Car", "Rent", "Medical", "School", "Gifts / Leisure"])

    submitted = st.form_submit_button("Assess Risk", use_container_width=True)


if submitted:
    input_row = pd.DataFrame(0, index=[0], columns=expected_features)

    input_row["amount_requested"] = amount_requested
    input_row["monthly_rent_amount"] = monthly_rent
    input_row["loan_duration"] = loan_duration
    input_row["num_payments"] = num_payments
    input_row["monthly_income_amount"] = monthly_income
    input_row["raw_l2c_score"] = raw_l2c_score
    input_row["raw_FICO_bank_card"] = raw_FICO_bank_card
    input_row["age_at_application"] = age

    if residence_rent_or_own == "Rent":
        input_row["residence_rent_or_own_True"] = 1
    if direct_deposit:
        input_row["bank_account_direct_deposit_True"] = 1
    if payment_ach:
        input_row["payment_ach_True"] = 1

    freq_col = f"payment_frequency_{payment_frequency}"
    use_col = f"how_use_money_{how_use_money}"

    if freq_col in input_row.columns:
        input_row[freq_col] = 1
    if use_col in input_row.columns:
        input_row[use_col] = 1

    probability_good = model.predict_proba(input_row)[0][1] * 100
    prediction = model.predict(input_row)[0]

    st.markdown("---")
    st.subheader("Risk Outcome")

    if prediction == 1:
        st.balloons()
        st.success(
            f"✅ Low Risk: Applicant is expected to perform well. Estimated probability of successful repayment: {probability_good:.1f}%"
        )
    else:
        st.error(
            f"⚠️ High Risk: Applicant may require additional review. Estimated probability of successful repayment: {probability_good:.1f}%"
        )

    st.progress(probability_good / 100)
    st.caption("This tool supports underwriting decisions and should be used alongside policy review and manual assessment.")
