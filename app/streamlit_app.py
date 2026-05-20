import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# 页面配置
st.set_page_config(page_title="MACE 风险预测工具 | MACE Risk Predictor", layout="wide")

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

# 特征列名（与训练时一致）
feature_names = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca',
    'sex_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'fbs_1.0',
    'restecg_1.0', 'restecg_2.0', 'exang_1.0', 'slope_2.0', 'slope_3.0',
    'thal_6.0', 'thal_7.0'
]


# 用户输入界面（中英文对照）
def user_input_features():
    st.sidebar.header("输入患者特征 | Patient Characteristics")

    age = st.sidebar.slider("年龄 | Age (years)", 30, 80, 55)
    sex = st.sidebar.selectbox("性别 | Sex", ["女性 | Female", "男性 | Male"])
    cp = st.sidebar.selectbox("胸痛类型 | Chest Pain Type",
                              options=[1, 2, 3, 4],
                              format_func=lambda x: {1: "典型心绞痛 | Typical angina",
                                                     2: "非典型心绞痛 | Atypical angina",
                                                     3: "非心绞痛性胸痛 | Non-anginal pain",
                                                     4: "无症状 | Asymptomatic"}[x])
    trestbps = st.sidebar.slider("静息血压 | Resting BP (mm Hg)", 90, 200, 130)
    chol = st.sidebar.slider("胆固醇 | Cholesterol (mg/dl)", 100, 400, 250)
    fbs = st.sidebar.selectbox("空腹血糖 >120 mg/dl | Fasting blood sugar >120", ["否 | No", "是 | Yes"])
    restecg = st.sidebar.selectbox("静息心电图结果 | Resting ECG results",
                                   options=[0, 1, 2],
                                   format_func=lambda x: {0: "正常 | Normal",
                                                          1: "ST-T异常 | ST-T abnormality",
                                                          2: "左室肥厚 | Left ventricular hypertrophy"}[x])
    thalach = st.sidebar.slider("最大心率 | Maximum heart rate", 70, 220, 150)
    exang = st.sidebar.selectbox("运动诱发心绞痛 | Exercise induced angina", ["否 | No", "是 | Yes"])
    oldpeak = st.sidebar.slider("ST段压低 | ST depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("ST段斜率 | ST segment slope",
                                 options=[1, 2, 3],
                                 format_func=lambda x: {1: "上斜 | Upsloping",
                                                        2: "平坦 | Flat",
                                                        3: "下斜 | Downsloping"}[x])
    ca = st.sidebar.selectbox("主要血管数 | Number of major vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("地中海贫血 | Thalassemia",
                                options=[3, 6, 7],
                                format_func=lambda x: {3: "正常 | Normal",
                                                       6: "固定缺陷 | Fixed defect",
                                                       7: "可逆缺陷 | Reversible defect"}[x])

    # 构造独热编码
    sex_val = 1 if sex.startswith("男性") else 0
    cp_2 = 1 if cp == 2 else 0
    cp_3 = 1 if cp == 3 else 0
    cp_4 = 1 if cp == 4 else 0
    fbs_val = 1 if fbs.startswith("是") else 0
    restecg_1 = 1 if restecg == 1 else 0
    restecg_2 = 1 if restecg == 2 else 0
    exang_val = 1 if exang.startswith("是") else 0
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
st.title("❤️ 心内科 MACE 风险预测工具")
st.markdown("### MACE Risk Prediction for Cardiology Inpatients")
st.markdown("基于逻辑回归模型（UCI Heart Disease 数据集）。输入患者信息即可预测住院期间主要不良心血管事件风险。")

input_df = user_input_features()

if st.button("预测风险 | Predict Risk"):
    prob = model.predict_proba(input_df)[0, 1]
    st.subheader("预测结果 | Prediction Result")
    st.metric("MACE 发生概率 | MACE Probability", f"{prob * 100:.1f}%")

    if prob < 0.2:
        risk_level_cn = "低危"
        risk_level_en = "Low risk"
        advice_cn = "常规随访，控制危险因素。"
        advice_en = "Routine follow-up, control risk factors."
        color = "green"
    elif prob < 0.5:
        risk_level_cn = "中危"
        risk_level_en = "Moderate risk"
        advice_cn = "建议进一步检查（如冠脉CTA），加强药物治疗。"
        advice_en = "Consider further examination (e.g., coronary CTA), intensify medical therapy."
        color = "orange"
    else:
        risk_level_cn = "高危"
        risk_level_en = "High risk"
        advice_cn = "强烈建议住院强化治疗，考虑介入评估。"
        advice_en = "Strongly recommend inpatient intensive treatment, consider interventional evaluation."
        color = "red"

    st.markdown(f"**风险等级 | Risk level: <span style='color:{color}'>{risk_level_cn} ({risk_level_en})</span>**",
                unsafe_allow_html=True)
    st.info(f"临床建议 | Clinical advice：{advice_cn} {advice_en}")

    # SHAP 解释图
    st.subheader("模型解释 (SHAP)")
    st.markdown("下图展示了各个特征对预测结果的影响。")

    col1, col2 = st.columns(2)
    shap_summary_path = os.path.join(BASE_DIR, "output", "shap_summary.png")
    shap_bar_path = os.path.join(BASE_DIR, "output", "shap_bar.png")

    with col1:
        if os.path.exists(shap_summary_path):
            img = Image.open(shap_summary_path)
            st.image(img, caption="SHAP Summary Plot", use_column_width=True)
        else:
            st.warning("shap_summary.png 未找到，请运行 R 脚本生成。")
    with col2:
        if os.path.exists(shap_bar_path):
            img = Image.open(shap_bar_path)
            st.image(img, caption="SHAP Feature Importance", use_column_width=True)
        else:
            st.warning("shap_bar.png 未找到，请运行 R 脚本生成。")