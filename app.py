import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("heart_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.title("Project Information")

st.sidebar.write("Model: Random Forest")

st.sidebar.write("Accuracy: 98.5%")

st.sidebar.write("Health Risk Prediction System")

st.title("❤️ Heart Disease Risk Prediction System")

st.markdown("""
This application predicts the risk of heart disease using a Random Forest Machine Learning Model.
""")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 50, 250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict Risk"):

    patient = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs,
          restecg, thalach, exang, oldpeak,
          slope, ca, thal]],
        columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal'
        ]
    )

    patient_scaled = scaler.transform(patient)

    probability = model.predict_proba(patient_scaled)[0][1]

    score = probability * 100

    st.progress(int(score))

    st.write(f"Risk Score: {score:.2f}%")

    st.bar_chart([score])

    if score < 30:
        st.success(f"Risk Score: {score:.2f}%")
        st.success("🟢 Low Risk")

    elif score < 70:
        st.warning(f"Risk Score: {score:.2f}%")
        st.warning("🟡 Medium Risk")

    else:
        st.error(f"Risk Score: {score:.2f}%")
        st.error("🔴 High Risk")