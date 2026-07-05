# Libraries
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(recipes)
library(themis)
library(doParallel)

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

# SMOTE Recipe
rf_recipe <- recipe(default ~ ., data = train_data) %>%
  step_smote(default)

prep_rf_recipe <- prep(rf_recipe, training = train_data)
train_data_smote <- bake(prep_rf_recipe, new_data = NULL)

# Enable parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# Define grid for mtry tuning
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8))

# Training control
rf_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

# Train Random Forest with tuning
set.seed(123)
rf_tuned_model <- train(
  default ~ .,
  data = train_data_smote,
  method = "rf",
  trControl = rf_ctrl,
  tuneGrid = rf_grid,
  metric = "ROC",
  ntree = 300
)

# Stop parallel backend
stopCluster(cl)
registerDoSEQ()

# Predict and evaluate
rf_tuned_pred <- predict(rf_tuned_model, newdata = test_data)
rf_tuned_prob <- predict(rf_tuned_model, newdata = test_data, type = "prob")[, "Yes"]
confusionMatrix(rf_tuned_pred, test_data$default)

rf_tuned_roc <- roc(test_data$default, rf_tuned_prob)
auc_value <- auc(rf_tuned_roc)

plot(rf_tuned_roc, 
     main = paste("ROC - Tuned RF + SMOTE | AUC:", round(auc_value, 4)), 
     col = "black", 
     lwd = 2)
abline(a = 0, b = 1, col = "gray", lty = 2)

cat("AUC:", round(auc_value, 4), "\n")

# Feature importance
print(varImp(rf_tuned_model))
