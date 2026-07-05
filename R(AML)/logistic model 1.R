#1. Load libraries
library(tidyverse)
library(caret)
library(pROC)

#load the dataset
data <- read.csv("C:/Users/dahal/OneDrive/school work/OneDrive/Desktop/R SEM 5/R(AML)/scaled_credit_data.csv")

#Convert 'default' to a factor
data$default <- as.factor(data$default)

# Convert 'default' to a factor 
data$default <- factor(data$default, levels = c(0, 1), labels = c("No", "Yes"))

#2. Identify the target column
target_column_name <- "default"

#3. Data splitting
set.seed(123)
train_index <- createDataPartition(data$default, p = 0.8, list = FALSE)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

cat("Training data size:", nrow(train_data), "\n")
cat("Testing data size:", nrow(test_data), "\n")

#5. Train logistic regression model using caret
set.seed(123)
model <- glm(default ~ ., data = train_data_smote, family = binomial())

#6. Model summary
print(model)

#7. Predict probabilities
predictions_prob <- predict(model, test_data, type = "response")

# Create ROC object
roc_obj <- roc(test_data$default, predictions_prob)

# Find best threshold 
best_coords <- coords(roc_obj, x= "best", ret = c("threshold", "sensitivity", "specificity"), best.method = "youden")
best_threshold <- as.numeric(best_coords["threshold"])

cat("Best threshold found:", round(best_threshold, 4), "\n")

# Apply best threshold to classify
predictions_binary <- ifelse(predictions_prob > best_threshold, "Yes", "No")

cat("Predictions distribution:\n")
print(table(predictions_binary))

cat("Test data distribution:\n")
print(table(test_data$default))

#8. Confusion Matrix
conf_matrix <- confusionMatrix(
  factor(predictions_binary, levels = c("No", "Yes")),
  factor(test_data[[target_column_name]], levels = c("No", "Yes"))
)
# Print confusion matrix
print(conf_matrix)

#9. Print individual metrics
cat("Accuracy:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("AUC Score:", round(auc(roc_obj), 4), "\n")

#10. Calculate and plot ROC curve
plot(roc_obj, main = paste("ROC Curve - AUC:", round(auc(roc_obj), 4)))
abline(a = 0, b = 1, lty = 2, col = "gray")


