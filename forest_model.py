import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import joblib
import os

# 读取数据并准备特征
df = pd.read_csv("data/heart_clean.csv")
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

def evaluate(y_true, y_prob, y_pred, model_name):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = max(tpr - fpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp+fn)>0 else 0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    print(f"\n=== {model_name} ===")
    print(f"AUC: {auc:.3f}")
    print(f"KS: {ks:.3f}")
    print(f"准确率: {acc:.3f}")
    print(f"敏感度: {sens:.3f}")
    print(f"特异度: {spec:.3f}")
    return auc, ks, acc, sens, spec

rf_metrics = evaluate(y_test, rf_prob, rf_pred, "Random Forest")

os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.pkl")
print("随机森林模型已保存至 models/random_forest_model.pkl")