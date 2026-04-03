import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PulseIQ Dashboard", layout="wide")

# ---------------- LOAD MODELS ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# ---------------- CACHE DATA (VERY IMPORTANT FIX) ----------------
@st.cache_data
def load_data():
    df = pd.read_excel("sample_data.xlsx")
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 PulseIQ Dashboard")
page = st.sidebar.radio("Navigation", [
    "Home",
    "Churn Prediction",
    "EDA Dashboard"
])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🚀 PulseIQ Dashboard")
    st.subheader("Real-Time Business Intelligence System")

    st.markdown("""
    ### 📌 Features:
    - Customer Churn Prediction  
    - Data Analysis Dashboard  
    - Machine Learning Integration  

    👈 Use sidebar to navigate
    """)

# ---------------- CHURN ----------------
elif page == "Churn Prediction":

    st.title("⚠️ Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        f1 = st.slider("Feature 1", 0.0, 100.0)
        f2 = st.slider("Feature 2", 0.0, 100.0)
        f3 = st.slider("Feature 3", 0.0, 100.0)

    with col2:
        f4 = st.slider("Feature 4", 0.0, 100.0)
        f5 = st.slider("Feature 5", 0.0, 100.0)
        category = st.selectbox("Category", list(encoder.classes_))

    if st.button("Predict Churn"):

        cat_encoded = encoder.transform([category]).reshape(-1, 1)

        full_input = np.array([[f1, f2, f3, f4, f5]])
        full_input = np.hstack((full_input, cat_encoded))

        scaled_input = scaler.transform(full_input)

        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        st.subheader(f"Churn Probability: {probability:.2f}")

        if prediction[0] == 1:
            st.error("⚠️ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

# ---------------- EDA ----------------
elif page == "EDA Dashboard":

    st.title("📊 Data Dashboard")

    # ✅ FAST LOAD (cached)
    df = load_data()

    # KPI
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Stats
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Charts
    st.subheader("Distribution Plot")
    fig1 = px.histogram(df, x=df.columns[0])
    st.plotly_chart(fig1, use_container_width=True)

    if len(df.columns) > 1:
        st.subheader("Scatter Plot")
        fig2 = px.scatter(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig2, use_container_width=True)
