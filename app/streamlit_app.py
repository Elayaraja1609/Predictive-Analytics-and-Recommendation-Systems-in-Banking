import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import LOAN_FILE, MODELS_DIR, PRODUCTS_FILE, REPORTS_DIR

st.set_page_config(
    page_title="Banking Analytics & Recommendations",
    page_icon="🏦",
    layout="wide",
)

st.title("Predictive Analytics & Recommendation Systems in Banking")
st.markdown(
    "Demo deployment for **loan default prediction**, "
    "**customer segmentation**, and **product recommendations**."
)


@st.cache_resource
def load_loan_model():
    path = MODELS_DIR / "loan_default_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_loan_data():
    return pd.read_csv(LOAN_FILE) if LOAN_FILE.exists() else None


@st.cache_data
def load_segments():
    path = REPORTS_DIR / "customer_segments.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_recommendations():
    path = REPORTS_DIR / "product_recommendations.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_products():
    return pd.read_csv(PRODUCTS_FILE) if PRODUCTS_FILE.exists() else None


def _compute_monthly_payment(loan_amount: float, interest_rate: float, loan_term: int) -> float:
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate > 0:
        return round(
            loan_amount
            * (monthly_rate * (1 + monthly_rate) ** loan_term)
            / ((1 + monthly_rate) ** loan_term - 1),
            2,
        )
    return round(loan_amount / loan_term, 2)


def _build_loan_features(
    age: int,
    income: float,
    credit_score: int,
    loan_amount: float,
    interest_rate: float,
    loan_term: int,
    employment_years: int,
    num_existing_loans: int,
    savings_balance: float,
    credit_utilization: float,
) -> dict:
    monthly_payment = _compute_monthly_payment(loan_amount, interest_rate, loan_term)
    monthly_income = income / 12
    return {
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "loan_term": loan_term,
        "employment_years": employment_years,
        "num_existing_loans": num_existing_loans,
        "savings_balance": savings_balance,
        "debt_to_income_ratio": round(loan_amount / max(income, 1), 4),
        "monthly_payment": monthly_payment,
        "payment_to_income_ratio": round(monthly_payment / max(monthly_income, 1), 4),
        "credit_utilization": credit_utilization,
        "loan_to_income_ratio": round(loan_amount / max(income, 1), 4),
        "credit_risk_band": (
            "high" if credit_score < 580 else "medium" if credit_score < 700 else "low"
        ),
    }


def show_metrics():
    cols = st.columns(3)
    for col, fname, title in zip(
        cols,
        ["loan_metrics.json", "segmentation_metrics.json", "recommendation_metrics.json"],
        ["Loan Default", "Segmentation", "Recommendations"],
    ):
        path = REPORTS_DIR / fname
        with col:
            st.subheader(title)
            if path.exists():
                st.json(json.loads(path.read_text(encoding="utf-8")))
            else:
                st.warning("Run pipeline first: `python run_pipeline.py`")


tab1, tab2, tab3, tab4 = st.tabs(
    ["Loan Default", "Segmentation", "Recommendations", "Model Metrics"]
)

with tab1:
    st.header("Loan Default Prediction")
    model = load_loan_model()
    df = load_loan_data()
    if model is None or df is None:
        st.error("Model or data not found. Run `python run_pipeline.py` first.")
    else:
        term_options = [12, 24, 36, 48, 60, 72, 84]

        def _init_loan_form(row):
            st.session_state.loan_age = int(row["age"])
            st.session_state.loan_income = float(row["income"])
            st.session_state.loan_credit_score = int(row["credit_score"])
            st.session_state.loan_amount = float(row["loan_amount"])
            st.session_state.loan_interest_rate = float(row["interest_rate"])
            term = int(row["loan_term"])
            st.session_state.loan_term = term if term in term_options else 36
            st.session_state.loan_employment_years = int(row["employment_years"])
            st.session_state.loan_num_existing_loans = int(row["num_existing_loans"])
            st.session_state.loan_savings_balance = float(row["savings_balance"])
            st.session_state.loan_credit_utilization = float(row["credit_utilization"])

        if "loan_age" not in st.session_state:
            _init_loan_form(df.sample(1).iloc[0])

        st.write("Adjust customer profile or load a random sample:")
        if st.button("Load random sample", key="load_random_loan"):
            _init_loan_form(df.sample(1).iloc[0])
            st.rerun()

        c1, c2, c3 = st.columns(3)
        age = int(c1.number_input("Age", 22, 70, key="loan_age"))
        income = float(c2.number_input("Annual Income", 10000.0, 5000000.0, key="loan_income"))
        credit_score = int(c3.number_input("Credit Score", 300, 850, key="loan_credit_score"))
        c4, c5, c6 = st.columns(3)
        loan_amount = float(c4.number_input("Loan Amount", 1000.0, 2000000.0, key="loan_amount"))
        interest_rate = float(c5.number_input("Interest Rate %", 3.0, 25.0, key="loan_interest_rate"))
        loan_term = int(c6.selectbox("Loan Term (months)", term_options, key="loan_term"))

        max_employment = max(0, age - 22)
        if st.session_state.loan_employment_years > max_employment:
            st.session_state.loan_employment_years = max_employment

        c7, c8, c9 = st.columns(3)
        employment_years = int(
            c7.number_input(
                "Employment Years",
                0,
                max_employment if max_employment > 0 else 0,
                key="loan_employment_years",
            )
        )
        num_existing_loans = int(
            c8.number_input("Existing Loans", 0, 4, key="loan_num_existing_loans")
        )
        savings_balance = float(
            c9.number_input("Savings Balance", 0.0, 5000000.0, key="loan_savings_balance")
        )
        credit_utilization = float(
            st.number_input(
                "Credit Utilization (0–1)",
                0.0,
                1.0,
                step=0.01,
                format="%.3f",
                key="loan_credit_utilization",
            )
        )

        features = _build_loan_features(
            age,
            income,
            credit_score,
            loan_amount,
            interest_rate,
            loan_term,
            employment_years,
            num_existing_loans,
            savings_balance,
            credit_utilization,
        )
        input_df = pd.DataFrame([features])
        if st.button("Predict Default Risk"):
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            if pred == 1:
                st.error(f"High default risk — probability: {prob:.1%}")
            else:
                st.success(f"Low default risk — probability: {prob:.1%}")

with tab2:
    st.header("Customer Segmentation")
    segments = load_segments()
    if segments is None:
        st.error("Segmentation data not found. Run pipeline first.")
    else:
        st.metric("Customers segmented", len(segments))
        st.metric("Clusters", segments["cluster"].nunique())
        st.dataframe(
            segments[["customer_id", "cluster", "segment_label", "txn_count", "total_amount"]].head(50),
            use_container_width=True,
        )
        seg_filter = st.selectbox(
            "Filter by segment",
            ["All"] + sorted(segments["segment_label"].unique().tolist()),
            key="seg_filter",
        )
        filtered = segments if seg_filter == "All" else segments[segments["segment_label"] == seg_filter]
        st.bar_chart(filtered.groupby("segment_label")["total_amount"].mean())

with tab3:
    st.header("Product Recommendations")
    recs = load_recommendations()
    products = load_products()
    if recs is None:
        st.error("Recommendations not found. Run pipeline first.")
    else:
        customers = sorted(recs["customer_id"].unique().tolist())
        if "rec_customer" not in st.session_state:
            st.session_state.rec_customer = customers[0]
        selected = st.selectbox("Select Customer", customers, key="rec_customer")
        cust_recs = recs[recs["customer_id"] == selected][["rank", "product_name", "category"]]
        st.dataframe(cust_recs, use_container_width=True, hide_index=True)

with tab4:
    show_metrics()