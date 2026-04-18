import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AI Fraud Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('models/fraud_pipeline.pkl')
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

st.title("🛡️ Banking AI Fraud Detection System")
st.markdown("Enter transaction details below to evaluate the risk of fraud in real-time.")

if not artifacts:
    st.error("Model not found! Please run `python3 train_model.py` first.")
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']

st.sidebar.header("Transaction Parameters")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
time = st.sidebar.slider("Time of Day (Seconds)", min_value=0, max_value=86400, value=36000)
v1 = st.sidebar.slider("Feature V1 (Location/Device)", min_value=-5.0, max_value=5.0, value=0.0)
v2 = st.sidebar.slider("Feature V2 (User History)", min_value=-5.0, max_value=5.0, value=0.0)
v3 = st.sidebar.slider("Feature V3 (Network Type)", min_value=-5.0, max_value=5.0, value=0.0)

if st.sidebar.button("Analyze Transaction", type="primary"):
    input_df = pd.DataFrame({'Time': [time], 'Amount': [amount], 'V1': [v1], 'V2': [v2], 'V3': [v3]})
    input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])
    
    probability = model.predict_proba(input_df)[0][1]
    is_fraud = probability > 0.70
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Risk Assessment")
        if is_fraud:
            st.error(f"🚨 DECLINED: High Fraud Probability ({probability:.1%})")
        else:
            st.success(f"✅ APPROVED: Transaction looks normal ({probability:.1%} Risk)")
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if is_fraud else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "salmon"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Feature Importance")
        importance = model.feature_importances_
        features = ['Time', 'Amount', 'V1', 'V2', 'V3']
        fig_bar = go.Figure([go.Bar(x=features, y=importance, marker_color='royalblue')])
        fig_bar.update_layout(title="Decision Drivers", yaxis_title="Importance Weight")
        st.plotly_chart(fig_bar, use_container_width=True)
