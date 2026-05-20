# ============================================================
# 心内科 MACE 预测模型 - XGBoost 补充建模（最终修正版）
# 修正 KS 计算：使用排序法，避免 pROC 阈值序列导致的 KS=1 问题
# ============================================================

# 1. 加载包
if (!require(xgboost)) install.packages("xgboost")
if (!require(pROC)) install.packages("pROC")      # 用于 AUC
if (!require(caret)) install.packages("caret")    # 用于混淆矩阵
if (!require(dplyr)) install.packages("dplyr")

library(xgboost)
library(pROC)
library(caret)
library(dplyr)

# 2. 读取数据
heart_data <- read.csv("data/heart_clean.csv")
cat_cols <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "thal")
for (col in cat_cols) {
  heart_data[[col]] <- as.factor(heart_data[[col]])
}

# 3. 构造模型矩阵（独热编码）
train_matrix <- model.matrix(target ~ . - 1, data = heart_data)
y <- heart_data$target

# 4. 划分训练/测试集（70/30，分层）
set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
x_train <- train_matrix[train_index, ]
y_train <- y[train_index]
x_test <- train_matrix[-train_index, ]
y_test <- y[-train_index]

# 5. 转换为 xgb.DMatrix
dtrain <- xgb.DMatrix(x_train, label = y_train)
dtest <- xgb.DMatrix(x_test, label = y_test)

# 6. XGBoost 参数
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 5,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0
)

# 7. 训练模型
set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose = 1,
  print_every_n = 10
)

# 8. 预测
y_pred_prob <- predict(xgb_model, dtest)
y_pred_class <- ifelse(y_pred_prob > 0.5, 1, 0)

# 9. AUC（使用 pROC）
roc_obj <- roc(y_test, y_pred_prob, quiet = TRUE)
auc_val <- as.numeric(roc_obj$auc)

# 10. KS 统计量（排序法，正确计算）
df_ks <- data.frame(prob = y_pred_prob, truth = y_test)
df_ks <- df_ks[order(df_ks$prob, decreasing = TRUE), ]
tpr <- cumsum(df_ks$truth) / sum(df_ks$truth)
fpr <- cumsum(1 - df_ks$truth) / sum(1 - df_ks$truth)
ks_val <- max(tpr - fpr)

# 11. 混淆矩阵及其他指标
cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_test), positive = "1")
accuracy <- cm$overall["Accuracy"]
sensitivity <- cm$byClass["Sensitivity"]
specificity <- cm$byClass["Specificity"]

# 12. 输出性能
cat("\n========== XGBoost Performance (Final Corrected) ==========\n")
cat(sprintf("AUC: %.3f\n", auc_val))
cat(sprintf("KS: %.3f\n", ks_val))
cat(sprintf("Accuracy: %.3f\n", accuracy))
cat(sprintf("Sensitivity: %.3f\n", sensitivity))
cat(sprintf("Specificity: %.3f\n", specificity))
cat("Confusion Matrix:\n")
print(cm$table)

# 13. 更新 model_performance_table.csv
perf_file <- "output/model_performance_table.csv"
perf_df <- read.csv(perf_file)
# 删除旧的 XGBoost 行（如果存在）
perf_df <- perf_df[perf_df$Model != "XGBoost", ]
new_row <- data.frame(Model = "XGBoost",
                      AUC = auc_val,
                      KS = ks_val,
                      Accuracy = accuracy,
                      Sensitivity = sensitivity,
                      Specificity = specificity)
perf_df <- rbind(perf_df, new_row)
write.csv(perf_df, perf_file, row.names = FALSE)
cat("\n✅ 性能表已更新，KS 使用排序法（正确值 =", ks_val, "）\n")

# 14. 保存模型
if (!dir.exists("models")) dir.create("models")
saveRDS(xgb_model, "models/xgboost_model_r.rds")
cat("✅ XGBoost 模型已保存至 models/xgboost_model_r.rds\n")

# 15. 特征重要性
importance_matrix <- xgb.importance(model = xgb_model, feature_names = colnames(x_train))
write.csv(importance_matrix, "output/xgboost_feature_importance.csv", row.names = FALSE)
cat("✅ 特征重要性已保存至 output/xgboost_feature_importance.csv\n")

# 16. 打印重要特征
cat("\n========== Top 10 Important Features ==========\n")
print(head(importance_matrix, 10))

# 17. 最终模型选择建议
lr_auc <- perf_df[perf_df$Model == "Logistic Regression", "AUC"]
if (auc_val - lr_auc < 0.03) {
  cat("\n结论：XGBoost AUC 提升小于 0.03，最终模型选择逻辑回归。\n")
} else {
  cat("\n结论：XGBoost 性能提升超过 0.03，建议改用 XGBoost。\n")
}

cat("\n========== XGBoost 补充建模完成 ==========\n")