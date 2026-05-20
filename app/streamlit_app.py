import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# 获取项目根目录（假设 app/ 在根目录下）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载模型
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在：{model_path}。请确保已上传模型。")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# 特征列名（必须与训练时一致）
feature_names = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca',
    'sex_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'fbs_1.0',
    'restecg_1.0', 'restecg_2.0', 'exang_1.0', 'slope_2.0', 'slope_3.0',
    'thal_6.0', 'thal_7.0'
]

# 用户输入界面
def user_input_features():
    st.sidebar.header("Input Patient Characteristics")
    
    age = st.sidebar.slider("Age (years)", 30, 80, 55)
    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    cp = st.sidebar.selectbox("Chest Pain Type", 
                              options=[1,2,3,4],
                              format_func=lambda x: {1:"Typical angina",2:"Atypical angina",3:"Non-anginal pain",4:"Asymptomatic"}[x])
    trestbps = st.sidebar.slider("Resting BP (mm Hg)", 90, 200, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 250)
    fbs = st.sidebar.selectbox("Fasting blood sugar >120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG results", 
                                   options=[0,1,2],
                                   format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"Left ventricular hypertrophy"}[x])
    thalach = st.sidebar.slider("Maximum heart rate", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise induced angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("ST segment slope", 
                                 options=[1,2,3],
                                 format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
    ca = st.sidebar.selectbox("Number of major vessels (0-3)", [0,1,2,3])
    thal = st.sidebar.selectbox("Thalassemia", 
                                options=[3,6,7],
                                format_func=lambda x: {3:"Normal",6:"Fixed defect",7:"Reversible defect"}[x])
    
    # 构造独热编码
    sex_val = 1 if sex == "Male" else 0
    cp_2 = 1 if cp == 2 else 0
    cp_3 = 1 if cp == 3 else 0
    cp_4 = 1 if cp == 4 else 0
    fbs_val = 1 if fbs == "Yes" else 0
    restecg_1 = 1 if restecg == 1 else 0
    restecg_2 = 1 if restecg == 2 else 0
    exang_val = 1 if exang == "Yes" else 0
    slope_2 = 1 if slope == 2 else 0
    slope_3 = 1 if slope == 3 else 0
    thal_6 = 1 if thal == 6 else 0
    thal_7 = 1 if thal == 7 else 0
    
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,
        'ca': ca,
        'sex_1.0': sex_val,
        'cp_2.0': cp_2,
        'cp_3.0': cp_3,
        'cp_4.0': cp_4,
        'fbs_1.0': fbs_val,
        'restecg_1.0': restecg_1,
        'restecg_2.0': restecg_2,
        'exang_1.0': exang_val,
        'slope_2.0': slope_2,
        'slope_3.0': slope_3,
        'thal_6.0': thal_6,
        'thal_7.0': thal_7
    }
    features = pd.DataFrame([data], columns=feature_names)
    return features

# 主界面
st.set_page_config(page_title="MACE Risk Predictor", layout="wide")
st.title("❤️ MACE Risk Prediction for Cardiology Inpatients")
st.markdown("Based on logistic regression model (UCI Heart Disease dataset). Enter patient data to predict in-hospital Major Adverse Cardiovascular Events (MACE).")

input_df = user_input_features()

if st.button("Predict Risk"):
    prob = model.predict_proba(input_df)[0, 1]
    st.subheader("Prediction Result")
    st.metric("MACE Probability", f"{prob*100:.1f}%")
    
    if prob < 0.2:
        risk_level = "Low risk"
        advice = "Routine follow-up, control risk factors."
        color = "green"
    elif prob < 0.5:
        risk_level = "Moderate risk"
        advice = "Consider further examination (e.g., coronary CTA), intensify medical therapy."
        color = "orange"
    else:
        risk_level = "High risk"
        advice = "Strongly recommend inpatient intensive treatment, consider interventional evaluation."
        color = "red"
    
    st.markdown(f"**Risk level: <span style='color:{color}'>{risk_level}</span>**", unsafe_allow_html=True)
    st.info(f"Clinical advice: {advice}")
    
    # SHAP 解释图
    st.subheader("Model Explanation (SHAP)")
    st.markdown("The following plots show the impact of each feature on the prediction.")
    
    col1, col2 = st.columns(2)
    shap_summary_path = os.path.join(BASE_DIR, "output", "shap_summary.png")
    shap_bar_path = os.path.join(BASE_DIR, "output", "shap_bar.png")
    
    with col1:
        if os.path.exists(shap_summary_path):
            img = Image.open(shap_summary_path)
            st.image(img, caption="SHAP Summary Plot", use_column_width=True)
        else:
            st.warning("shap_summary.png not found. Please run the R script to generate SHAP plots.")
    with col2:
        if os.path.exists(shap_bar_path):
            img = Image.open(shap_bar_path)
            st.image(img, caption="SHAP Feature Importance", use_column_width=True)
        else:
            st.warning("shap_bar.png not found. Please run the R script to generate SHAP plots.")