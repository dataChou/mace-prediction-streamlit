# MACE Prediction for Cardiology Inpatients

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mace-prediction-app-datchou.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![R 4.2+](https://img.shields.io/badge/R-4.2+-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于预测住院患者主要不良心血管事件（MACE）风险的机器学习工具。基于UCI Heart Disease数据集，使用逻辑回归构建核心模型，提供交互式Web应用和SHAP模型解释，辅助临床医生进行风险分层和决策。

## 📌 项目亮点

- **临床可解释**：输出每个特征对预测的贡献（SHAP分析），风险判断透明可追溯
- **完整建模流程**：涵盖数据清洗、EDA、特征工程、逻辑回归 + 随机森林 + XGBoost对比、SHAP可解释性
- **交互式Web应用**：输入患者特征，实时输出MACE发生概率 + 风险等级（低/中/高危）+ 临床建议
- **双语界面**：中文/英文对照，适应国内外用户
- **多语言实现**：Python构建核心模型与前端，R完成高级建模（XGBoost、SHAP）与系数表输出

## 📊 模型性能

| 模型 | AUC | KS | 准确率 | 敏感度 | 特异度 |
|------|-----|----|--------|--------|--------|
| 逻辑回归（最终模型） | 0.931 | 0.724 | 0.846 | 0.810 | 0.878 |
| 随机森林 | 0.912 | 0.724 | 0.802 | 0.810 | 0.796 |
| XGBoost | 0.905 | 0.650 | 0.844 | 0.719 | 0.914 |

**结论**：逻辑回归AUC最高（0.931），KS=0.724，优于随机森林和XGBoost，因此作为最终部署模型。

## 🛠️ 技术栈

### Python
- **框架**：Streamlit（交互式前端）
- **建模**：scikit-learn（逻辑回归、随机森林）
- **数据处理**：pandas、numpy
- **模型可解释性**：SHAP（由R生成，前端展示静态图）

### R
- **高级建模**：xgboost
- **模型可解释性**：fastshap、shapviz
- **统计建模**：glm（逻辑回归系数表）、pROC（AUC/KS）

## 🚀 快速开始

### 在线体验（无需安装）
直接访问：[https://mace-prediction-app-datchou.streamlit.app/](https://mace-prediction-app-datchou.streamlit.app/)

## 📁 项目结构
mace-prediction-streamlit/
├── app/                         # Streamlit应用
│   └── streamlit_app.py         # 主页面
├── data/                        # 数据文件
│   └── heart_clean.csv          # 清洗后数据集
├── models/                      # 训练好的模型
│   ├── logistic_model.pkl       # 逻辑回归模型（最终部署）
│   └── random_forest_model.pkl  # 随机森林模型
├── output/                      # 输出文件
│   ├── table1_descriptive.xlsx  # 描述性统计表
│   ├── model_performance_table.xlsx # 模型性能对比表
│   ├── logistic_coefficients.csv    # 逻辑回归系数表（OR, CI, p值）
│   ├── shap_summary.png         # SHAP蜂群图
│   └── shap_bar.png             # SHAP特征重要性图
├── R/                           # R脚本（XGBoost、SHAP、系数表）
│   ├── 03b_xgboost.R            # XGBoost建模
│   ├── step4_coefficients.R     # 逻辑回归系数表（OR, CI, p值）
│   └── step5_shap.R             # SHAP计算与绘图
├── requirements.txt             # Python依赖
├── 心内科MACE预测模型.Rproj      # RStudio项目文件
└── README.md                    # 本文件

## 📖 案例：心内科MACE风险预测
业务背景：住院心内科患者需要快速评估发生主要不良心血管事件（MACE）的风险，以指导治疗方案和资源分配。

成果：
逻辑回归模型AUC=0.931，KS=0.724，临床可用

提供交互式Web工具，一键输入 → 输出风险概率 + 分级建议

SHAP分析显示，主要危险因素为：ca（血管数）、thal（地中海贫血）、sex（性别）、cp（胸痛类型）

业务建议：
低危（<20%）：常规随访，控制危险因素
中危（20%-50%）：建议进一步检查（如冠脉CTA），加强药物治疗
高危（>50%）：强烈建议住院强化治疗，考虑介入评估

## 🤝 如何获取帮助或商务合作
如果您有类似数据（医疗、金融、保险等），需要构建预测模型或评分卡：
免费数据可行性评估：发送匿名数据样例，2天内回复方法建议
完整项目合作：从数据清洗到交付业务工具，报价透明

## 联系方式
微信：Amydlmy
邮箱：scarlettzwm@gmail.com

## 📄 许可证
MIT License

## 🙏 致谢
数据来源：UCI Machine Learning Repository – Heart Disease Dataset (Cleveland)
R包作者：fastshap, shapviz, xgboost, pROC, caret
Python包作者：Streamlit, scikit-learn, pandas, numpy, shap, joblib, Pillow