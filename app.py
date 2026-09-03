import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD MODEL FILES
# ============================================================
model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ============================================================
# CSS DESIGN
# ============================================================
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(255,0,0,0.15), transparent 30%),
        radial-gradient(circle at 90% 80%, rgba(180,0,0,0.15), transparent 30%),
        #050505;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* =========================
   HEADER
   ========================= */
.heart-header {
    text-align: center;
    padding: 25px;
    margin-bottom: 30px;
    background: rgba(20,0,0,0.75);
    border-radius: 20px;
    border: 1px solid rgba(255,50,50,0.5);
    box-shadow: 0 0 20px rgba(255,0,0,0.4);
}

.heart-header h1 {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(255,0,0,0.7);
    margin-bottom: 5px;
}

.heart-header p {
    font-size: 17px;
    color: #cccccc;
}

/* =========================
   INPUT LABELS
   ========================= */
label {
    color: #eeeeee !important;
    font-weight: 600 !important;
}

/* =========================
   NUMBER & SELECT INPUTS
   ========================= */
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background-color: rgba(20,20,20,0.90);
    border-radius: 10px;
    border: 1px solid rgba(255,60,60,0.35);
    transition: 0.3s;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border: 1px solid #ff3333;
    box-shadow: 0 0 10px rgba(255,0,0,0.6);
}

/* =========================
   PREDICT BUTTON
   ========================= */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    background: linear-gradient(90deg, #8b0000, #ff1a1a);
    color: white;
    font-size: 20px;
    font-weight: 700;
    border: 1px solid #ff4444;
    box-shadow: 0 0 15px rgba(255,0,0,0.45);
    transition: all 0.3s ease;
    margin-top: 20px;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #ff1a1a, #ff5555);
    box-shadow: 0 0 25px rgba(255,0,0,0.8);
}

/* =========================
   RESULT CARD
   ========================= */
.result-card {
    margin-top: 30px;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    background: rgba(15, 15, 20, 0.95);
    color: white;
}

.high-risk-card {
    border: 2px solid #ff2222;
    box-shadow: 0 0 20px rgba(255,0,0,0.7), 0 0 40px rgba(255, 0, 0, 0.4);
}

.low-risk-card {
    border: 2px solid #00ff66;
    box-shadow: 0 0 20px rgba(0,255,100,0.6), 0 0 40px rgba(0, 255, 100, 0.3);
}

.result-card h1 {
    font-size: 38px;
    margin-bottom: 10px;
    padding-bottom: 0;
}

.result-card h2 {
    color: white;
    font-size: 24px;
    margin-bottom: 10px;
}

.result-card p {
    color: #dddddd;
    font-size: 18px;
    margin-bottom: 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="heart-header">
    <h1>❤️ HEART DISEASE PREDICTION</h1>
    <p>Machine Learning Based Cardiovascular Risk Analysis</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("### 🩺 Patient Information")

age = st.slider("AGE",18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80,200,120)
cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL",[0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
max_hr = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise-Induced Angina",["Y","N"])
oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope", ["Up","Flat","Down"])

if st.button("❤️ ANALYZE HEART HEALTH"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    if prediction == 0:
        # Change background to green
        st.markdown("""
        <style>
        .stApp {
            background:
                radial-gradient(circle at 20% 20%, rgba(0,255,100,0.20), transparent 35%),
                radial-gradient(circle at 80% 80%, rgba(0,180,70,0.15), transparent 35%),
                #020b05;
        }
        </style>
        """, unsafe_allow_html=True)

        # Removed the heavy spaces before the tags so Markdown doesn't render it as a code block
        st.markdown("""
<div class="result-card low-risk-card">
    <h1>💚 LOW RISK</h1>
    <h2>Heart Disease Not Detected</h2>
    <p>The model predicts a low risk of heart disease.</p>
</div>
""", unsafe_allow_html=True)

        st.success(f"Confidence: {probabilities[0]:.2%}")

    # ========================================================
    # HIGH RISK
    # ========================================================
    else:
        # Change background to red
        st.markdown("""
        <style>
        .stApp {
            background:
                radial-gradient(circle at 20% 20%, rgba(255,0,0,0.25), transparent 35%),
                radial-gradient(circle at 80% 80%, rgba(180,0,0,0.20), transparent 35%),
                #0b0202;
        }
        </style>
        """, unsafe_allow_html=True)

        # Removed the heavy spaces before the tags so Markdown doesn't render it as a code block
        st.markdown("""
<div class="result-card high-risk-card">
    <h1>🚨 HIGH RISK</h1>
    <h2>Heart Disease Detected</h2>
    <p>The model predicts a higher risk of heart disease.</p>
</div>
""", unsafe_allow_html=True)

        st.error(f"Risk Probability: {probabilities[1]:.2%}")

    # ========================================================
    # PROBABILITY
    # ========================================================
    st.markdown("### 📊 Model Probability")

    st.write("No Heart Disease:", f"{probabilities[0]:.2%}")
    st.write("Heart Disease:", f"{probabilities[1]:.2%}")