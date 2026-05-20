import pandas as pd
import os

# 模型性能数据
data = {
    "Model": ["Logistic Regression", "Random Forest"],
    "AUC": [0.931, 0.912],
    "KS": [0.724, 0.724],
    "Accuracy": [0.846, 0.802],
    "Sensitivity": [0.810, 0.810],
    "Specificity": [0.878, 0.796]
}
table2 = pd.DataFrame(data)

# 保存
os.makedirs("output", exist_ok=True)
table2.to_csv("output/model_performance_table.csv", index=False)
table2.to_excel("output/model_performance_table.xlsx", index=False)

print("Table 2 已保存至 output/model_performance_table.xlsx")
print(table2)