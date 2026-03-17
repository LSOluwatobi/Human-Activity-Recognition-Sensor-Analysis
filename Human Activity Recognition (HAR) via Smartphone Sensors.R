# ==============================================================================
# PROJECT: Human Activity Recognition (HAR) via Smartphone Sensors
# Models: Random Forest, SVM (Radial), XGBoost
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: Setup & Data Acquisition (Timeout Fix)
# ------------------------------------------------------------------------------
# 1. Increase timeout to 600 seconds (10 minutes)
options(timeout = 600)

if(!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, randomForest, xgboost, e1071, pROC, kernlab)

# Proceed to load as before
features <- read.table("UCI HAR Dataset/features.txt", col.names = c("n", "feature_name"))
activities <- read.table("UCI HAR Dataset/activity_labels.txt", col.names = c("n", "label"))

x_train <- read.table("UCI HAR Dataset/train/X_train.txt")
y_train <- read.table("UCI HAR Dataset/train/y_train.txt", col.names = "activity")

x_test <- read.table("UCI HAR Dataset/test/X_test.txt")
y_test <- read.table("UCI HAR Dataset/test/y_test.txt", col.names = "activity")

# ------------------------------------------------------------------------------
# PHASE 2: Data Preprocessing
# ------------------------------------------------------------------------------
# Map numeric activity codes (1-6) to descriptive factors
y_train$activity <- factor(y_train$activity, levels = activities$n, labels = activities$label)
y_test$activity  <- factor(y_test$activity,  levels = activities$n, labels = activities$label)

# Combine into dataframes
train_df <- cbind(activity = y_train$activity, x_train)
test_df  <- cbind(activity = y_test$activity, x_test)

# Identify near-zero variance predictors ONLY
nzv <- nearZeroVar(train_df[,-1])

# Keep activity column + filtered predictors
train_df <- train_df[, c(TRUE, !nzv)]
test_df  <- test_df[, c(TRUE, !nzv)]

# ------------------------------------------------------------------------------
# PHASE 3: Model 1 - Random Forest (Bagging)
# ------------------------------------------------------------------------------
# We limit ntree for speed given the high dimensionality

fit_rf <- randomForest(
  activity ~ .,
  data = train_df,
  ntree = 200,
  mtry = floor(sqrt(ncol(train_df)-1)),
  importance = TRUE
)


# 3. Train SVM using the same method
# Note: For e1071::svm, the arguments are slightly different
fit_svm <- svm(x = x_train, y = y_train, kernel = "radial", probability = TRUE)


# ------------------------------------------------------------------------------
# PHASE 4: Model 2 - SVM with Radial Basis Function (Kernel)
# ------------------------------------------------------------------------------
# SVM is the 'Gold Standard' for this specific dataset
fit_svm <- svm(activity ~ ., data = train_df, kernel = "radial", probability = TRUE)

# ------------------------------------------------------------------------------
# PHASE 5: Model 3 - XGBoost (Multi-class Boosting)
# ------------------------------------------------------------------------------
# Prepare matrix for multi-class classification
train_x_mat <- as.matrix(train_df[,-1])
train_y_vec <- as.numeric(train_df$activity) - 1 # XGBoost needs 0-indexed integers

# 1. Convert to DMatrix (The professional, high-speed format)
dtrain <- xgb.DMatrix(data = train_x_mat, label = train_y_vec)
dtest  <- xgb.DMatrix(data = as.matrix(test_df[,-1]))

# 2. Define Parameters for Multi-class
# 'multi:softprob' gives us the probability for each of the 6 activities
params <- list(
  objective = "multi:softprob",
  num_class = 6,
  eval_metric = "mlogloss"
)

# 3. Train Model
fit_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 50,
  verbose = 0
)

# ------------------------------------------------------------------------------
# PHASE 6: Predictions & Evaluation
# ------------------------------------------------------------------------------
# 1. Random Forest Preds
pred_rf <- predict(fit_rf, test_df)

# 2. SVM Preds
pred_svm <- predict(fit_svm, test_df)

# 3. XGBoost Preds (Choose class with highest probability)
pred_xgb_prob <- predict(fit_xgb, as.matrix(test_df[,-1]))
pred_xgb_mat  <- matrix(pred_xgb_prob, ncol = 6, byrow = TRUE)
pred_xgb      <- factor(max.col(pred_xgb_mat), levels = 1:6, labels = levels(train_df$activity))

# ------------------------------------------------------------------------------
# PHASE 7: Metrics Comparison Table
# ------------------------------------------------------------------------------
eval_model <- function(preds, actual, label) {
  cm <- confusionMatrix(preds, actual)
  return(data.frame(
    Model = label,
    Overall_Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"]
  ))
}

comparison_df <- rbind(
  eval_model(pred_rf, test_df$activity, "Random Forest"),
  eval_model(pred_svm, test_df$activity, "SVM (Radial)"),
  eval_model(pred_xgb, test_df$activity, "XGBoost")
)

print(comparison_df)

# ------------------------------------------------------------------------------
# PHASE 8: Visualizing the Confusion Matrix for Top Model
# ------------------------------------------------------------------------------
# Let's visualize the SVM Confusion Matrix (usually the strongest)
cm_svm <- confusionMatrix(pred_svm, test_df$activity)
plt_cm <- as.data.frame(cm_svm$table)

ggplot(plt_cm, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "grey", high = "#2980b9") +
  labs(title = "Confusion Matrix: SVM Performance") +
  theme_minimal()

# ------------------------------------------------------------------------------
# PHASE 9: Variable Importance
# ------------------------------------------------------------------------------
# Top features in sensor data are often 'Gravity' related
varImpPlot(fit_rf, n.var = 15, main = "Top 15 Sensor Features (Random Forest)")
