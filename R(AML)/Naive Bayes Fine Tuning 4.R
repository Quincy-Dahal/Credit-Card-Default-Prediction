# Load libraries
library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(naivebayes)
library(doParallel)
library(themis)
library(recipes)

# Load the dataset
data <- read.csv("C:/Users/dahal/OneDrive/school work/OneDrive/Desktop/R SEM 5/R(AML)/scaled_credit_data.csv")

# Convert 'default' to factor
data$default <- factor(data$default, levels = c(0, 1), labels = c("No", "Yes"))

# Data splitting
set.seed(123)
train_index <- createDataPartition(data$default, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

cat("Training data size:", nrow(train_data), "\n")
cat("Testing data size:", nrow(test_data), "\n")

# Preprocessing: Recipe with SMOTE
rec <- recipe(default ~ ., data = train_data) %>%
  step_smote(default)

prep_rec <- prep(rec, training = train_data)
train_data_smote <- bake(prep_rec, new_data = NULL)

# Enable parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# Define tuning grid
nb_grid <- expand.grid(
  laplace = c(0, 1),
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 2)
)

# Train the Naive Bayes model with tuning
set.seed(123)
nb_model <- train(
  default ~ .,
  data = train_data_smote,
  method = "naive_bayes",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = nb_grid
)

# Stop parallel backend
stopCluster(cl)
registerDoSEQ()

# Display model performance
print(nb_model)
plot(nb_model)

# Make predictions
nb_pred <- predict(nb_model, newdata = test_data)
nb_prob <- predict(nb_model, newdata = test_data, type = "prob")

# Evaluate performance
nb_cm <- confusionMatrix(nb_pred, test_data$default)
nb_roc <- roc(response = test_data$default, predictor = nb_prob[, "Yes"])

# Print results
print(nb_cm)
cat("AUC:", round(auc(nb_roc), 4), "\n")

# Plot ROC Curve
plot(nb_roc, main = paste("Naives Bayes - AUC:", round(auc(nb_roc), 4)), col = "black")
abline(a = 0, b = 1, lty = 2, col = "gray")