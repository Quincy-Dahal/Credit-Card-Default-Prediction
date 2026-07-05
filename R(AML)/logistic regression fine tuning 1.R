# 1. Load libraries
library(tidyverse)
library(caret)
library(pROC)
library(themis)
library(recipes)
library(glmnet)
library(doParallel)

# 2. Load dataset
data <- read.csv("C:/Users/dahal/OneDrive/school work/OneDrive/Desktop/R SEM 5/R(AML)/scaled_credit_data.csv")

# 3. Convert 'default' to factor with valid R names
data$default <- factor(data$default, levels = c(0, 1), labels = c("No", "Yes"))

# 4. Data split
set.seed(123)
train_index <- createDataPartition(data$default, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

cat("Training data size:", nrow(train_data), "\n")
cat("Testing data size:", nrow(test_data), "\n")

# 5. Apply SMOTE to training data
recipe_smote <- recipe(default ~ ., data = train_data) %>%
  step_smote(default)
prep_recipe <- prep(recipe_smote, training = train_data)
train_data_smote <- bake(prep_recipe, new_data = NULL)

# Enable parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# 6. Train logistic regression with glmnet (with tuning)
set.seed(123)
logistic_tuned <- train(
  default ~ ., 
  data = train_data_smote,
  method = "glmnet",
  family = "binomial",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    classProbs = TRUE, 
    summaryFunction = twoClassSummary
  ),
  tuneLength = 10,
  metric = "ROC"
)

# Stop parallel backend
stopCluster(cl)
registerDoSEQ()

# 7. Predict probabilities on test set
predictions_prob <- predict(logistic_tuned, newdata = test_data, type = "prob")[, "Yes"]

# 8. Create ROC and find optimal threshold
roc_obj <- roc(test_data$default, predictions_prob)
best_coords <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
best_threshold <- as.numeric(best_coords)

# 9. Convert probabilities to predicted class using threshold
predictions_binary <- ifelse(predictions_prob > best_threshold, "Yes", "No")

# 10. Confusion Matrix with correct factor levels
conf_matrix <- confusionMatrix(
  factor(predictions_binary, levels = c("No", "Yes")),
  factor(test_data$default, levels = c("No", "Yes"))
)
print(conf_matrix)

# 11. Print performance metrics
cat("Accuracy:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensitivity (Recall):", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("Precision (PPV):", round(conf_matrix$byClass["Pos Pred Value"], 4), "\n")
cat("F1 Score:", round(2 * ((conf_matrix$byClass["Sensitivity"] * conf_matrix$byClass["Pos Pred Value"]) /
                              (conf_matrix$byClass["Sensitivity"] + conf_matrix$byClass["Pos Pred Value"])), 4), "\n")
cat("AUC:", round(auc(roc_obj), 4), "\n")

# 12. Plot ROC curve
plot(roc_obj, main = paste("Logistic Regression ROC Curve - AUC:", round(auc(roc_obj), 4)))
abline(a = 0, b = 1, lty = 2, col = "gray")
