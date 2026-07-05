# Load library
library(tidyverse)
library(rpart)
library(caret)
library(pROC)
library(rpart.plot)

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

#Model Training
set.seed(123)
model_dt <- train(
  form = default ~ .,
  data = train_data,
  trControl = trainControl(method = "cv", number = 5),
  method = "rpart"
)

# Predict probabilities
pred_prob <- predict(model_dt, test_data, type = "prob")[, 2]

# ROC and Threshold
roc_obj <- roc(test_data$default, pred_prob)
best_threshold <- coords(roc_obj, "best", ret = "threshold") %>% as.numeric()

# Apply best threshold
pred_class <- factor(pred_class, levels = c("No", "Yes"))
reference <- factor(test_data$default, levels = c("No", "Yes"))

# Evaluate
conf_matrix <- confusionMatrix(pred_class, reference)

# Print confusion matrix
print(conf_matrix)

#6. Print individual metrics
cat("Accuracy:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(roc_obj), 4), "\n")

library(rpart.plot)
rpart.plot(model_dt$finalModel)

#7. Calculate and plot ROC curve
roc_obj <- roc(test_data[[target_column_name]], pred_prob)
auc_value <- auc(roc_obj)

plot(roc_obj, main = paste("ROC Curve - AUC:", round(auc_value, 4)))
abline(a = 0, b = 1, lty = 2, col = "gray")





