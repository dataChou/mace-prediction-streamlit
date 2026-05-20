import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/heart_clean.csv")
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_enc.drop("target", axis=1).astype(float)
y = df_enc["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train_const = sm.add_constant(X_train)
y_train = y_train.values.astype(float)

# 增加 maxiter 以提高收敛可能性
model = sm.Logit(y_train, X_train_const).fit(disp=0, maxiter=10000)

params = model.params
conf = model.conf_int()
pvals = model.pvalues

# 修复：使用 np.exp 而不是 .exp()
or_vals = np.exp(params)
or_lower = np.exp(conf[0])
or_upper = np.exp(conf[1])

coef_df = pd.DataFrame({
    "feature": params.index,
    "coefficient": params.values,
    "odds_ratio": or_vals,
    "OR_95%CI_lower": or_lower,
    "OR_95%CI_upper": or_upper,
    "p_value": pvals.values
})

os.makedirs("output", exist_ok=True)
coef_df.to_csv("output/logistic_coefficients.csv", index=False)

print("✅ 步骤4完成：逻辑回归系数表已保存至 output/logistic_coefficients.csv")
print(coef_df.round(4))