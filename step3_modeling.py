import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/heart_clean.csv")
cat_cols = ["sex","cp","fbs","restecg","exang","slope","thal"]
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_enc.drop("target", axis=1)
y = df_enc["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 逻辑回归
lr = LogisticRegression(max_iter=3000, random_state=42)
lr.fit(X_train, y_train)
joblib.dump(lr, "models/logistic_model.pkl")
lr_prob = lr.predict_proba(X_test)[:,1]
lr_pred = lr.predict(X_test)

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest_model.pkl")
rf_prob = rf.predict_proba(X_test)[:,1]
rf_pred = rf.predict(X_test)

def metrics(y_true, y_prob, y_pred):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = max(tpr - fpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    return auc, ks, acc, sens, spec

lr_auc, lr_ks, lr_acc, lr_sens, lr_spec = metrics(y_test, lr_prob, lr_pred)
rf_auc, rf_ks, rf_acc, rf_sens, rf_spec = metrics(y_test, rf_prob, rf_pred)

perf = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "AUC": [lr_auc, rf_auc],
    "KS": [lr_ks, rf_ks],
    "Accuracy": [lr_acc, rf_acc],
    "Sensitivity": [lr_sens, rf_sens],
    "Specificity": [lr_spec, rf_spec]
})
perf.to_csv("output/model_performance_table.csv", index=False)
perf.to_excel("output/model_performance_table.xlsx", index=False)

print("✅ 步骤3完成：模型已保存，性能表已生成")
print(perf.round(3))