import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import joblib
import os

# 读取数据
df = pd.read_csv("data/heart_clean.csv")

# 特征工程（独热编码）
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集样本量: {len(X_train)} (MACE=1: {y_train.sum()})")
print(f"测试集样本量: {len(X_test)} (MACE=1: {y_test.sum()})")

# 逻辑回归：增大 max_iter 到 3000，其他保持默认
lr = LogisticRegression(max_iter=3000, random_state=42)
lr.fit(X_train, y_train)

# 预测
y_pred_prob = lr.predict_proba(X_test)[:, 1]
y_pred = lr.predict(X_test)

# 评估指标
auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ks = max(tpr - fpr)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_test, y_pred)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n=== 逻辑回归性能 ===")
print(f"AUC: {auc:.3f}")
print(f"KS: {ks:.3f}")
print(f"准确率: {accuracy:.3f}")
print(f"敏感度 (Recall): {sensitivity:.3f}")
print(f"特异度: {specificity:.3f}")
print("混淆矩阵:")
print(cm)

# 保存模型
os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/logistic_model.pkl")
print("逻辑回归模型已保存至 models/logistic_model.pkl")

# 可选：输出系数（用于风险评分解释）
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lr.coef_[0],
    "odds_ratio": np.exp(lr.coef_[0])
})
print("\n特征系数（原始尺度）:")
print(coef_df.round(4))