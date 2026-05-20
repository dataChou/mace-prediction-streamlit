import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
import os  # ✅ 添加此行

df = pd.read_csv("data/heart_clean.csv")

continuous_vars = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

var_labels = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalach": "Maximum heart rate",
    "oldpeak": "ST depression",
    "ca": "Number of major vessels",
    "sex": "Sex (male=1)",
    "cp": "Chest pain type",
    "fbs": "Fasting blood sugar >120 mg/dl",
    "restecg": "Resting ECG results",
    "exang": "Exercise induced angina",
    "slope": "ST segment slope",
    "thal": "Thalassemia"
}

total_n = len(df)
group0 = df[df["target"] == 0]
group1 = df[df["target"] == 1]
n0, n1 = len(group0), len(group1)

results = []

# 连续变量
for var in continuous_vars:
    mean0, std0 = group0[var].mean(), group0[var].std()
    mean1, std1 = group1[var].mean(), group1[var].std()
    t_stat, p_val = ttest_ind(group0[var], group1[var], equal_var=False)
    results.append({
        "Variable": var_labels[var],
        f"Total (N={total_n})": f"{df[var].mean():.1f}±{df[var].std():.1f}",
        f"MACE=0 (N={n0})": f"{mean0:.1f}±{std0:.1f}",
        f"MACE=1 (N={n1})": f"{mean1:.1f}±{std1:.1f}",
        "p-value": f"{p_val:.3f}"
    })

# 分类变量
for var in categorical_vars:
    total_counts = df[var].value_counts().sort_index()
    total_perc = total_counts / total_n * 100
    total_str = "; ".join([f"{idx}: {total_counts[idx]} ({total_perc[idx]:.1f}%)" for idx in total_counts.index])

    counts0 = group0[var].value_counts().sort_index()
    perc0 = counts0 / n0 * 100
    str0 = "; ".join([f"{idx}: {counts0[idx]} ({perc0[idx]:.1f}%)" for idx in counts0.index])

    counts1 = group1[var].value_counts().sort_index()
    perc1 = counts1 / n1 * 100
    str1 = "; ".join([f"{idx}: {counts1[idx]} ({perc1[idx]:.1f}%)" for idx in counts1.index])

    crosstab = pd.crosstab(df[var], df["target"])
    chi2, p_val, dof, expected = chi2_contingency(crosstab)
    results.append({
        "Variable": var_labels[var],
        f"Total (N={total_n})": total_str,
        f"MACE=0 (N={n0})": str0,
        f"MACE=1 (N={n1})": str1,
        "p-value": f"{p_val:.3f}"
    })

table1_df = pd.DataFrame(results)
os.makedirs("output", exist_ok=True)
table1_df.to_csv("output/table1_descriptive.csv", index=False)
table1_df.to_excel("output/table1_descriptive.xlsx", index=False)

print("Table 1 已保存至 output/table1_descriptive.xlsx")
print(table1_df.head())