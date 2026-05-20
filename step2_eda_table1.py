import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import os

os.makedirs("output", exist_ok=True)
df = pd.read_csv("data/heart_clean.csv")

num_cols = ["age","trestbps","chol","thalach","oldpeak","ca"]
cat_cols = ["sex","cp","fbs","restecg","exang","slope","thal"]

# 1. 箱线图
df_melt = df.melt(id_vars="target", value_vars=num_cols, var_name="variable", value_name="value")
plt.figure(figsize=(12,6))
sns.boxplot(data=df_melt, x="variable", y="value", hue="target", palette="Set2")
plt.title("Boxplot of numeric variables by target")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/boxplot_group.png", dpi=150)
plt.close()

# 2. 分类变量堆叠条形图
for col in cat_cols:
    ct = pd.crosstab(df[col], df["target"], normalize="index") * 100
    ax = ct.plot(kind="bar", stacked=True, figsize=(6,4), title=f"{col} vs target (%)")
    ax.set_ylabel("Percentage")
    ax.legend(title="MACE", labels=["No event","Event"])
    plt.tight_layout()
    plt.savefig(f"output/bar_{col}.png", dpi=150)
    plt.close()

# 3. 相关性热图
plt.figure(figsize=(10,8))
corr = df[num_cols + ["target"]].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True)
plt.title("Correlation matrix (numeric + target)")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=150)
plt.close()

# 4. Table 1（描述性统计）
total_n = len(df)
group0 = df[df["target"]==0]
group1 = df[df["target"]==1]
n0, n1 = len(group0), len(group1)
results = []

for var in num_cols:
    mean0, std0 = group0[var].mean(), group0[var].std()
    mean1, std1 = group1[var].mean(), group1[var].std()
    t_stat, p_val = ttest_ind(group0[var], group1[var], equal_var=False)
    results.append({
        "Variable": var.capitalize(),
        f"Total (N={total_n})": f"{df[var].mean():.1f}±{df[var].std():.1f}",
        f"MACE=0 (N={n0})": f"{mean0:.1f}±{std0:.1f}",
        f"MACE=1 (N={n1})": f"{mean1:.1f}±{std1:.1f}",
        "p-value": f"{p_val:.3f}"
    })

var_labels = {"sex":"Sex (male=1)","cp":"Chest pain type","fbs":"Fasting blood sugar >120",
              "restecg":"Resting ECG","exang":"Exercise induced angina","slope":"ST segment slope","thal":"Thalassemia"}
for var in cat_cols:
    total_counts = df[var].value_counts().sort_index()
    total_perc = total_counts / total_n * 100
    total_str = "; ".join([f"{idx}: {total_counts[idx]} ({total_perc[idx]:.1f}%)" for idx in total_counts.index])
    counts0 = group0[var].value_counts().sort_index()
    perc0 = counts0 / n0 * 100
    str0 = "; ".join([f"{idx}: {counts0[idx]} ({perc0[idx]:.1f}%)" for idx in counts0.index])
    counts1 = group1[var].value_counts().sort_index()
    perc1 = counts1 / n1 * 100
    str1 = "; ".join([f"{idx}: {counts1[idx]} ({perc1[idx]:.1f}%)" for idx in counts1.index])
    ctab = pd.crosstab(df[var], df["target"])
    chi2, p_val, dof, exp = chi2_contingency(ctab)
    results.append({
        "Variable": var_labels.get(var, var),
        f"Total (N={total_n})": total_str,
        f"MACE=0 (N={n0})": str0,
        f"MACE=1 (N={n1})": str1,
        "p-value": f"{p_val:.3f}"
    })

table1 = pd.DataFrame(results)
table1.to_csv("output/table1_descriptive.csv", index=False)
table1.to_excel("output/table1_descriptive.xlsx", index=False)

# 5. 修正：二分类 target 分布（覆盖旧文件）
dist = df["target"].value_counts().reset_index()
dist.columns = ["target", "count"]
dist["percentage"] = (dist["count"] / total_n * 100).round(1)
dist.to_csv("output/target_distribution.csv", index=False)

# 6. 删除无效的多分类 group_describe.csv（如果存在）
if os.path.exists("output/group_describe.csv"):
    os.remove("output/group_describe.csv")
    print("已删除无效的 group_describe.csv")

print("✅ 步骤2完成：所有图表和 Table1 已生成，target_distribution.csv 已更新（二分类）")