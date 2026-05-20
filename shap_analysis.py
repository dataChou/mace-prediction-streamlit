import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# 加载数据与模型
df = pd.read_csv("data/heart_clean.csv")
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lr = joblib.load("models/logistic_model.pkl")

# 使用 KernelExplainer（适用于任何模型，但较慢）或 LinearExplainer（更快）
# 尝试使用 shap.explainers.Linear（如果可用）
try:
    # 新版 shap 路径
    explainer = shap.explainers.Linear(lr, X_train)
except AttributeError:
    # 旧版或兼容方式
    explainer = shap.LinearExplainer(lr, X_train, feature_dependence="independent")

shap_values = explainer.shap_values(X_test)

# 如果是二分类，shap_values 可能是一个列表，取正类
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# SHAP summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("output/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# SHAP bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("output/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("SHAP 图已保存至 output/shap_summary.png 和 output/shap_bar.png")