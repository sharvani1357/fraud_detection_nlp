import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import plotly.express as px
import pickle

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🚨",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================
st.title("🚨 Deep Learning Fraud Detection Dashboard")
st.markdown("Upload transaction data and predict fraud probabilities.")

# =====================================================
# SYSTEM PATH RESOLVER
# =====================================================
def find_file_path(filename):
    search_root = os.path.dirname(os.path.abspath(__file__))
    direct_path = os.path.join(search_root, filename)
    if os.path.exists(direct_path):
        return direct_path
    
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None

# =====================================================
# ARTIFACTS LOADING (Cleaned)
# =====================================================
@st.cache_resource
def load_artifacts():
    model_path = find_file_path("attention_model.keras")
    tokenizer_path = find_file_path("tokenizer.pkl")

    # 1. Load Keras Model
    if not model_path:
        raise FileNotFoundError("attention_model.keras could not be found anywhere in the repository structure.")
    model = tf.keras.models.load_model(model_path, compile=False)
        
    # 2. Load Tokenizer (Optional fallback if not found)
    tokenizer = None
    if tokenizer_path:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    
    # Returning only model and tokenizer now
    return model, tokenizer

# Unpacking modified to remove label_encoder
try:
    model, tokenizer = load_artifacts()
    st.sidebar.success("Model & Tokenizer Loaded Successfully!")
except Exception as e:
    st.error(f"Critical Deployment Error: {e}")
    st.info("Check your GitHub repository to ensure 'attention_model.keras' is fully uploaded inside your project folder.")
    st.stop()

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload CSV File for Analysis", type=["csv"])

# =====================================================
# MAIN APP LOGIC
# =====================================================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Extract numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.error("Dataset contains no numeric columns for prediction.")
            st.stop()

        # Gather expected model properties
        input_shape = model.input_shape 
        sequence_length = input_shape[1] if input_shape[1] is not None else 1
        expected_features = input_shape[2]

        if numeric_df.shape[1] < expected_features:
            st.error(f"Feature Mismatch: Model requires {expected_features} features, but CSV only has {numeric_df.shape[1]} numeric columns.")
            st.stop()
        
        input_data = numeric_df.iloc[:, :expected_features].values

        if len(input_data) < sequence_length:
            st.error(f"Data contains too few records. Minimum required rows: {sequence_length}")
            st.stop()

        # Build structural array sequences
        X = []
        for i in range(len(input_data) - sequence_length + 1):
            X.append(input_data[i : i + sequence_length])
        X = np.array(X)

        # Run Predictions
        with st.spinner("Processing deep learning sequence inferences..."):
            predictions = model.predict(X, verbose=0)
            predictions = predictions.flatten()

        # Align lengths with sliced output data
        results = df.iloc[sequence_length - 1 :].copy()
        results["Fraud_Probability"] = predictions

        def classify(prob):
            if prob >= 0.8: return "High Risk"
            elif prob >= 0.5: return "Medium Risk"
            return "Low Risk"

        results["Risk_Level"] = results["Fraud_Probability"].apply(classify)

        # Dashboard UI Visualizations
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows Evaluated", len(results))
        m2.metric("High Risk Flags", len(results[results["Risk_Level"] == "High Risk"]))
        m3.metric("Mean Fraud Score", f"{results['Fraud_Probability'].mean():.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Trend Analytics")
            fig_trend = px.line(results, y="Fraud_Probability", color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig_trend, use_container_width=True)
        with c2:
            st.subheader("Risk Segmentation Breakdown")
            fig_pie = px.pie(results, names="Risk_Level", hole=0.4,
                             color="Risk_Level",
                             color_discrete_map={"Low Risk":"green", "Medium Risk":"orange", "High Risk":"red"})
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🚨 Flagged Transactions Log")
        st.dataframe(results[results["Risk_Level"] != "Low Risk"].sort_values("Fraud_Probability", ascending=False), use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button("📩 Export Threat Report (CSV)", csv_data, "fraud_intelligence_report.csv", "text/csv")

    except Exception as e:
        st.error(f"An unexpected data processing error occurred: {e}")
        st.exception(e)
else:
    st.info("👋 Systems Online. Please upload a valid CSV transaction dataset to evaluate metrics.")