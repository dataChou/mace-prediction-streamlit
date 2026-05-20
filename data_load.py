import pandas as pd
import requests
import urllib3
from io import StringIO
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 下载数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
response = requests.get(url, verify=False)
df = pd.read_csv(StringIO(response.text), names=column_names, na_values="?")

# 缺失值填补：使用新写法（避免 inplace 链式赋值）
for col in df.columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)   # ✅ 推荐写法

# 将 target 二分类（0=无事件，1=有事件）
df["target"] = (df["target"] > 0).astype(int)

# 保存
os.makedirs("data", exist_ok=True)
df.to_csv("data/heart_clean.csv", index=False)

print("数据已修正并保存")
print("target 分布:\n", df["target"].value_counts())