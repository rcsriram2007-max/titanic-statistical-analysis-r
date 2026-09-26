
# 1. Package Management
required_packages <- c("titanic", "tidyverse", "caret", "pROC", "car", "e1071")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(titanic)
library(tidyverse)
library(caret)
library(pROC)
library(car)
library(e1071)

# Set seed for exact reproducibility
set.seed(42)

# 2. Data Ingestion & Preprocessing
data("titanic_train", package = "titanic")
df_raw <- titanic_train

# Structure and Missing Value Assessment
cat("--- Initial Data Summary ---\n")
str(df_raw)
colSums(is.na(df_raw))

# Data Cleaning & Feature Engineering
df_clean <- df_raw %>%
  # Impute Age with median conditional on Pclass
  group_by(Pclass) %>%
  mutate(Age = if_else(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup() %>%
  # Impute Embarked missing values with the mode ('S')
  mutate(Embarked = if_else(Embarked == "" | is.na(Embarked), "S", Embarked)) %>%
  # Feature selection & type conversions
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(
    Survived = factor(Survived, levels = c(0, 1), labels = c("Died", "Survived")),
    Pclass   = factor(Pclass, ordered = TRUE),
    Sex      = factor(Sex),
    Embarked = factor(Embarked)
  )

# Verify no remaining missing values
cat("\n--- Missing Values Post-Cleaning ---\n")
print(colSums(is.na(df_clean)))

# ==============================================================================
# 3. Exploratory Statistical Analysis & Hypothesis Testing
# ==============================================================================

# Hypothesis 1: Normality & Two-Sample Test for Continuous Feature (Fare)
# H0: Fare is normally distributed.
# H1: Fare is not normally distributed.
shapiro_died <- shapiro.test(df_clean$Fare[df_clean$Survived == "Died"][1:100]) # Sampled for Shapiro test limit
cat("\n--- Shapiro-Wilk Test (Fare - Died) ---\n")
print(shapiro_died)

# Non-Parametric Hypothesis Test: Mann-Whitney U / Wilcoxon Rank-Sum Test
# H0: The distribution of Fare is identical across Died and Survived groups.
# H1: The distribution of Fare differs significantly between groups.
wilcox_fare <- wilcox.test(Fare ~ Survived, data = df_clean)
cat("\n--- Wilcoxon Rank-Sum Test (Fare vs. Survived) ---\n")
print(wilcox_fare)

# Hypothesis 2: Categorical Independence Test (Sex vs. Survived)
# H0: Survival is independent of Passenger Sex.
# H1: Survival is significantly dependent on Passenger Sex.
contingency_sex <- table(df_clean$Sex, df_clean$Survived)
chisq_sex <- chisq.test(contingency_sex)
cat("\n--- Chi-Square Test of Independence (Sex vs. Survived) ---\n")
print(chisq_sex)

# Hypothesis 3: Categorical Independence Test (Pclass vs. Survived)
# H0: Survival is independent of Passenger Class.
# H1: Survival is significantly dependent on Passenger Class.
contingency_pclass <- table(df_clean$Pclass, df_clean$Survived)
chisq_pclass <- chisq.test(contingency_pclass)
cat("\n--- Chi-Square Test of Independence (Pclass vs. Survived) ---\n")
print(chisq_pclass)

# ==============================================================================
# 4. Data Partitioning & Model Building
# ==============================================================================

# Stratified 80/20 Train-Test Split
train_idx <- createDataPartition(df_clean$Survived, p = 0.80, list = FALSE)
train_set <- df_clean[train_idx, ]
test_set  <- df_clean[-train_idx, ]

# 10-Fold Cross-Validation Setup
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Fit Logistic Regression Model
logit_model <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_set,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = cv_control
)

cat("\n--- Cross-Validated Model Coefficients & Performance ---\n")
print(summary(logit_model$finalModel))
print(logit_model)

# Odds Ratios and 95% Confidence Intervals
odds_ratios <- exp(cbind(OR = coef(logit_model$finalModel), confint(logit_model$finalModel)))
cat("\n--- Odds Ratios ---\n")
print(odds_ratios)

# ==============================================================================
# 5. Diagnostic Analysis & Model Evaluation
# ==============================================================================

# Multicollinearity Check using Variance Inflation Factor (VIF)
cat("\n--- Variance Inflation Factors (VIF) ---\n")
vif_values <- vif(logit_model$finalModel)
print(vif_values)

# Predict on Unseen Test Set
test_preds_prob <- predict(logit_model, newdata = test_set, type = "prob")[, "Survived"]
test_preds_class <- predict(logit_model, newdata = test_set)

# Confusion Matrix & Performance Metrics
conf_matrix <- confusionMatrix(data = test_preds_class, reference = test_set$Survived, positive = "Survived")
cat("\n--- Test Set Confusion Matrix & Metrics ---\n")
print(conf_matrix)

# ROC Curve and Area Under the Curve (AUC)
roc_obj <- roc(test_set$Survived, test_preds_prob, levels = c("Died", "Survived"))
cat(sprintf("\nTest Set AUC: %.4f\n", auc(roc_obj)))

# ==============================================================================
# 6. Visualizations for Report Documentation
# ==============================================================================

# A. Residual Deviance Diagnostic Plot
par(mfrow = c(2, 2))
plot(logit_model$finalModel)
par(mfrow = c(1, 1))

# B. ROC Curve Plot
plot(roc_obj, col = "#1f77b4", lwd = 3, main = "ROC Curve - Logistic Regression (Test Set)")
abline(a = 0, b = 1, lty = 2, col = "grey50")
text(0.4, 0.2, sprintf("AUC = %.3f", auc(roc_obj)), font = 2)

# C. Variable Importance Plot
var_imp <- varImp(logit_model)
plot(var_imp, main = "Variable Importance (Logistic Regression)")
# ==============================================================================
# 6. Visualizations for Report Documentation (Saved to Disk)
# ==============================================================================

# A. Residual Deviance Diagnostic Plots
png(filename = "diagnostic_residuals.png", width = 1800, height = 1800, res = 300)
par(mfrow = c(2, 2))
plot(logit_model$finalModel)
par(mfrow = c(1, 1))
dev.off()
cat("Saved: diagnostic_residuals.png\n")

# B. ROC Curve Plot
png(filename = "roc_curve.png", width = 1800, height = 1500, res = 300)
plot(roc_obj, col = "#1f77b4", lwd = 3, main = "ROC Curve - Logistic Regression (Test Set)")
abline(a = 0, b = 1, lty = 2, col = "grey50")
text(0.4, 0.2, sprintf("AUC = %.3f", auc(roc_obj)), font = 2, cex = 1.1)
dev.off()
cat("Saved: roc_curve.png\n")

# C. Variable Importance Plot
png(filename = "variable_importance.png", width = 1800, height = 1500, res = 300)
var_imp <- varImp(logit_model)
plot(var_imp, main = "Variable Importance (Logistic Regression)")
dev.off()
cat("Saved: variable_importance.png\n")
