heart_data <- read.csv("data/heart_clean.csv")
heart_data$sex <- as.factor(heart_data$sex)
heart_data$cp <- as.factor(heart_data$cp)
heart_data$fbs <- as.factor(heart_data$fbs)
heart_data$restecg <- as.factor(heart_data$restecg)
heart_data$exang <- as.factor(heart_data$exang)
heart_data$slope <- as.factor(heart_data$slope)
heart_data$thal <- as.factor(heart_data$thal)
heart_data$target <- as.factor(heart_data$target)

model <- glm(target ~ ., data = heart_data, family = binomial())
coef_summary <- summary(model)$coefficients
or_vals <- exp(coef_summary[, "Estimate"])
or_lower <- exp(coef_summary[, "Estimate"] - 1.96 * coef_summary[, "Std. Error"])
or_upper <- exp(coef_summary[, "Estimate"] + 1.96 * coef_summary[, "Std. Error"])

result <- data.frame(
  feature = rownames(coef_summary),
  coefficient = coef_summary[, "Estimate"],
  odds_ratio = or_vals,
  OR_95CI_lower = or_lower,
  OR_95CI_upper = or_upper,
  p_value = coef_summary[, "Pr(>|z|)"]
)
write.csv(result, "output/logistic_coefficients.csv", row.names = FALSE)
print("步骤4完成：系数表已保存")