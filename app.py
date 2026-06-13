import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configure the page
st.set_page_config(page_title="Air Quality Risk Predictor", page_icon="🌤️", layout="wide")

# The URL of your running FastAPI server
API_URL = "http://localhost:8000/predict"

# Risk colors for UI styling
RISK_COLORS = {
    "Good": "#22c55e",       # Green
    "Moderate": "#eab308",   # Yellow
    "Unhealthy": "#f97316",  # Orange
    "Hazardous": "#ef4444"   # Red
}

st.title("🌤️ Air Quality Health Risk Predictor")
st.markdown("Adjust the sensor readings below to predict the current health risk tier using our PyTorch MLP.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Sensor Inputs")
    
    # Default to current time
    now = datetime.now()
    
    pm25 = st.slider("PM2.5 (µg/m³)", min_value=0.0, max_value=250.0, value=35.0, step=0.1)
    no2 = st.slider("NO2 (µg/m³)", min_value=0.0, max_value=150.0, value=25.0, step=0.1)
    o3 = st.slider("O3 (µg/m³)", min_value=0.0, max_value=200.0, value=40.0, step=0.1)
    
    st.divider()
    st.header("Time Context")
    hour = st.slider("Hour of Day", min_value=0, max_value=23, value=now.hour)
    month = st.slider("Month", min_value=1, max_value=12, value=now.month)
    
    st.divider()
    explain = st.toggle("Generate SHAP Explanations", value=True)
    
    submit = st.button("Predict Risk", type="primary", use_container_width=True)

# --- Main Dashboard ---
if submit:
    payload = {
        "pm25": pm25,
        "no2": no2,
        "o3": o3,
        "hour_of_day": hour,
        "month": month,
        "explain": explain
    }

    with st.spinner("Running inference..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            label = result["risk_label"]
            conf = result["confidence"]
            probs = result["probabilities"]
            shap_vals = result.get("shap_values")

            # Top Metrics Row
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Predicted Risk Tier", value=label)
            with col2:
                st.metric(label="Model Confidence", value=f"{conf * 100:.1f}%")
            with col3:
                # Add a visual color indicator
                st.markdown(f"""
                    <div style="background-color: {RISK_COLORS.get(label, 'gray')}; 
                                height: 80px; border-radius: 10px; display: flex; 
                                align-items: center; justify-content: center; color: white; 
                                font-weight: bold; font-size: 24px;">
                        {label.upper()}
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Charts Row
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("**Class Probabilities**")
                prob_df = pd.DataFrame(list(probs.items()), columns=["Risk Level", "Probability"])
                # Map colors
                prob_df["Color"] = prob_df["Risk Level"].map(RISK_COLORS)
                
                fig_prob = px.bar(
                    prob_df, x="Risk Level", y="Probability", 
                    color="Risk Level", color_discrete_map=RISK_COLORS,
                    text_auto='.1%', range_y=[0, 1]
                )
                fig_prob.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_prob, use_container_width=True)

            with col_chart2:
                if explain and shap_vals:
                    st.markdown("**SHAP Feature Attributions**")
                    st.caption(f"How each feature contributed to the '{label}' prediction:")
                    
                    shap_df = pd.DataFrame(list(shap_vals.items()), columns=["Feature", "Impact"])
                    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=True)
                    shap_df["Sign"] = shap_df["Impact"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
                    
                    fig_shap = px.bar(
                        shap_df, x="Impact", y="Feature", orientation="h",
                        color="Sign", color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#3b82f6"}
                    )
                    fig_shap.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_shap, use_container_width=True)
                elif explain and not shap_vals:
                    st.info("SHAP values were requested but not returned by the API.")
                else:
                    st.info("Enable SHAP explanations in the sidebar to see feature impact.")

        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not connect to the FastAPI server. Is it running on port 8000?")
        except requests.exceptions.HTTPError as e:
            st.error(f"🚨 API Error: {e.response.text}")
else:
    # Display a placeholder when the app first loads
    st.info("👈 Adjust the parameters in the sidebar and click 'Predict Risk' to see the model output.")