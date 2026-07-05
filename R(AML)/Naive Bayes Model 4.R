#libraries
library(tidyverse)
library(caret)
library(pROC)
library(e1071)

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

#training model
set.seed(123)
nb_model <- train(
  default ~ .,
  data = train_data,
  method = "naive_bayes",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "ROC"
)

# Predictions and evaluation on the test data
nb_pred <- predict(nb_model, newdata = test_data)
nb_prob <- predict(nb_model, newdata = test_data, type = "prob")

# Confusion Matrix
nb_cm <- confusionMatrix(nb_pred, test_data$default)
print(nb_cm)

# ROC and AUC
nb_roc <- roc(response = test_data$default, predictor = nb_prob[,2])

cat("Accuracy:", round(nb_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(nb_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(nb_cm$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(nb_roc), 4), "\n")

# ROC Plot
plot(nb_roc, main = paste("Naives Bayes - AUC:", round(auc(nb_roc), 4)), col = "black")
abline(a = 0, b = 1, lty = 2, col = "gray")