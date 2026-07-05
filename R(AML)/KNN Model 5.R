#Libraries
library(tidyverse)
library(caret)
library(pROC)
library(kknn)
library(recipes)

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

# Recipe: normalize 
rec_simple <- recipe(default ~ ., data = train_data) %>%
  step_normalize(all_predictors())

prep_simple <- prep(rec_simple)
train_simple <- bake(prep_simple, new_data = NULL)
test_simple  <- bake(prep_simple, new_data = test_data)

# Model Training with parallel processing
set.seed(123)
knn_model <- train(
  default ~ .,
  data = train_simple,
  method = "knn",
  tuneGrid = data.frame(k = 5),
  trControl = trainControl(method = "none", classProbs = TRUE),
  metric = "ROC"
)

# Predict and evaluate
prob_simple <- predict(knn_model, test_simple, type = "prob")[, "Yes"]
roc_simple <- roc(test_data$default, prob_simple)
best_thresh_simple <- coords(roc_simple, "best", ret = "threshold") %>% as.numeric()
pred_simple <- ifelse(prob_simple > best_thresh_simple, "Yes", "No")

conf_simple <- confusionMatrix(factor(pred_simple, levels = c("No", "Yes")), test_data$default)
print(conf_simple)

# Print metrics
cat("Accuracy:", round(conf_simple$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(conf_simple$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_simple$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(roc_simple), 4), "\n")

# Plot ROC
plot(roc_simple, main = paste("KNN - AUC:", round(auc(roc_simple), 4)), col = "black")
abline(a = 0, b = 1, lty = 2, col = "gray")