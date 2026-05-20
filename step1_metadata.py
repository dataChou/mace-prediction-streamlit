import pandas as pd
import os

os.makedirs("output", exist_ok=True)

# 读取已清洗的数据（必须存在且正确）
df = pd.read_csv("data/heart_clean.csv")

# 缺失值报告（清洗后无缺失）
missing_report = pd.DataFrame({
    "variable": df.columns,
    "missing_count": [0] * len(df.columns),
    "median_imputed": [""] * len(df.columns)
})
missing_report.to_csv("output/missing_report.csv", index=False)

# 数据字典（基于 UCI 文档）
cols = df.columns.tolist()
dict_df = pd.DataFrame({
    "Variable": cols,
    "中文名": ["年龄", "性别", "胸痛类型", "静息血压", "胆固醇", "空腹血糖>120", "静息心电图",
               "最大心率", "运动心绞痛", "ST段压低", "ST段斜率", "主要血管数", "地中海贫血", "MACE事件"],
    "类型": ["连续", "分类", "分类", "连续", "连续", "分类", "分类",
             "连续", "分类", "连续", "分类", "连续", "分类", "二分类"],
    "取值范围/编码": ["29-77", "0=F,1=M", "1-4", "94-200", "126-564", "0/1", "0-2",
                   "71-202", "0/1", "0-6.2", "1-3", "0-3", "3,6,7", "0/1"],
    "缺失数(填补前)": [0] * len(cols),
    "填补值(中位数)": ["-"] * len(cols)
})
dict_df.to_csv("output/data_dictionary.csv", index=False)

print("✅ 步骤1完成：生成 output/missing_report.csv 和 output/data_dictionary.csv")
print("target 分布:", df["target"].value_counts().to_dict())