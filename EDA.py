import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/heart_clean.csv")

# 数值变量箱线图
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
df_melt = df.melt(id_vars="target", value_vars=num_cols, var_name="variable", value_name="value")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melt, x="variable", y="value", hue="target", palette="Set2")
plt.title("Boxplot of numeric variables by target")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/boxplot_group.png", dpi=150)
plt.close()

# 分类变量堆叠条形图
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
for col in cat_cols:
    ct = pd.crosstab(df[col], df["target"], normalize="index") * 100
    ax = ct.plot(kind="bar", stacked=True, figsize=(6,4), title=f"{col} vs target (%)")
    ax.set_ylabel("Percentage")
    ax.legend(title="MACE", labels=["No event", "Event"])
    plt.tight_layout()
    plt.savefig(f"output/bar_{col}.png", dpi=150)
    plt.close()

# 相关性热图
plt.figure(figsize=(10, 8))
corr = df[num_cols + ["target"]].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True)
plt.title("Correlation matrix (numeric + target)")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=150)
plt.close()

print("EDA 完成，所有图片已保存到 output/ 文件夹")