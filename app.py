import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import time
import os
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from src.analysis import (
    compute_health_risk_index,
    add_risk_category,
    train_ml_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Farm Health Monitor",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------------------------------
# MODERN UI STYLE
# ---------------------------------------------------

st.markdown("""
<style>
.main {background-color: #0E1117;}
h1, h2, h3 {color: white;}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stMetric {
    background-color: #1F2937;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 Smart Farm Worker Health Monitoring System")
st.markdown("### AI-Powered Occupational Risk Prediction Dashboard")
st.markdown("---")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/farm_worker_health.csv")
    df = compute_health_risk_index(df)
    df = add_risk_category(df)
    return df

df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard Overview", "🤖 Train Model", "🔎 Risk Prediction"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
#st.sidebar.info("Portfolio-Level AI Research System")

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

if section == "📊 Dashboard Overview":

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Avg Temperature", f"{df['Temperature'].mean():.2f} °C")
    col3.metric("Avg Heart Rate", f"{df['Heart_Rate'].mean():.2f} bpm")

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

elif section == "🤖 Train Model":

    st.subheader("Train Machine Learning Model")

    if st.button("🚀 Train Model"):

        with st.spinner("Training model... Please wait ⏳"):

            start_time = time.time()

            model, feature_names, X_test, y_test, acc = train_ml_model(df)
            joblib.dump(model, "model.pkl")

            end_time = time.time()
            training_time = end_time - start_time

        st.balloons()
        st.success("Model trained successfully!")

        st.metric("Model Accuracy", f"{acc:.4f}")
        st.metric("Training Time (seconds)", f"{training_time:.2f}")

        # Save accuracy history
        history_file = "accuracy_history.csv"
        new_entry = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Accuracy": [acc],
        "Training_Time": [training_time]
   })


        if os.path.exists(history_file):
            history = pd.read_csv(history_file)
            history = pd.concat([history, new_entry])
        else:
            history = new_entry

        history.to_csv(history_file, index=False)

        st.markdown("### 📈 Accuracy History")
        history["Date"] = history["Date"].astype(str)
        st.line_chart(history.set_index("Date")["Accuracy"])

        # Feature Importance
        st.markdown("### 📊 Feature Importance")
        plot_feature_importance(model, feature_names)
        st.pyplot(plt.gcf())

        # Confusion Matrix
        st.markdown("### 📌 Confusion Matrix")
        plot_confusion_matrix(model, X_test, y_test)
        st.pyplot(plt.gcf())

        # ROC Curve
        st.markdown("### 📈 ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot(plt.gcf())

# ---------------------------------------------------
# RISK PREDICTION
# ---------------------------------------------------

elif section == "🔎 Risk Prediction":

    st.subheader("🔎 Real-Time Worker Health Prediction")

    try:
        model = joblib.load("model.pkl")

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider("🌡 Temperature (°C)", 28.0, 45.0, 32.0)
            humidity = st.slider("💧 Humidity (%)", 35.0, 85.0, 50.0)

        with col2:
            heart_rate = st.slider("❤️ Heart Rate (bpm)", 60.0, 130.0, 80.0)
            working_hours = st.slider("⏱ Working Hours", 4.0, 12.0, 8.0)

        input_df = pd.DataFrame(
            [[temperature, humidity, heart_rate, working_hours]],
            columns=["Temperature", "Humidity", "Heart_Rate", "Working_Hours"]
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        risk_labels = ["LOW", "MEDIUM", "HIGH"]
        risk = risk_labels[prediction]
        confidence = max(probabilities) * 100

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        if risk == "LOW":
            st.success(f"🟢 Risk Level: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"🟡 Risk Level: {risk}")
        else:
            st.error(f"🔴 Risk Level: {risk}")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")

        # -------------------------
        # SHAP SECTION
        # -------------------------

        st.markdown("---")
        st.markdown("## 🧠 SHAP Feature Contribution")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            fig, ax = plt.subplots()
            # For multi-class, select predicted class
            shap.summary_plot(
                shap_values[prediction],
                input_df,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)

            

        except Exception:
            st.warning("SHAP visualization not supported for this model type.")

        # -------------------------
        # PDF REPORT
        # -------------------------

        st.markdown("---")
        st.markdown("## 📄 Download Prediction Report")

        if st.button("Generate PDF Report"):

            doc = SimpleDocTemplate("prediction_report.pdf")
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("Smart Farm Worker Health Prediction Report", styles['Title']))
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph(f"Predicted Risk: {risk}", styles['Normal']))
            elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))
            elements.append(Paragraph(f"Temperature: {temperature}", styles['Normal']))
            elements.append(Paragraph(f"Humidity: {humidity}", styles['Normal']))
            elements.append(Paragraph(f"Heart Rate: {heart_rate}", styles['Normal']))
            elements.append(Paragraph(f"Working Hours: {working_hours}", styles['Normal']))

            doc.build(elements)

            with open("prediction_report.pdf", "rb") as f:
                st.download_button(
                    "Download Report",
                    f,
                    file_name="prediction_report.pdf",
                    mime="application/pdf"
                )

    except:
        st.warning("Please train the model first.")
