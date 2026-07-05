# Load required libraries
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(naivebayes)
library(recipes)
library(themis)
library(doParallel)

# Load data
data <- read.csv("C:/Users/dahal/OneDrive/school work/OneDrive/Desktop/R SEM 5/R(AML)/scaled_credit_data.csv")
data$default <- factor(data$default, levels = c(0, 1), labels = c("No", "Yes"))

# Data splitting
set.seed(123)
train_index <- createDataPartition(data$default, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Apply SMOTE
recipe_obj <- recipe(default ~ ., data = train_data) %>%
  step_smote(default)
prep_recipe <- prep(recipe_obj, training = train_data)
train_data_smote <- bake(prep_recipe, new_data = NULL)

# Enable parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# Train Random Forest
rf_model <- train(
  default ~ ., data = train_data_smote,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  tuneGrid = expand.grid(mtry = c(2, 4, 6)),
  metric = "ROC",
  ntree = 300
)

# Train Naive Bayes
nb_model <- train(
  default ~ ., data = train_data_smote,
  method = "naive_bayes",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  tuneGrid = expand.grid(laplace = 1, usekernel = TRUE, adjust = 1),
  metric = "ROC"
)

# Stop parallel backend
stopCluster(cl)
registerDoSEQ()

# Predict probabilities on test set
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]
nb_probs <- predict(nb_model, newdata = test_data, type = "prob")[, "Yes"]

# Combine predictions (simple average)
ensemble_probs <- (rf_probs + nb_probs) / 2

# Final class predictions
ensemble_pred <- ifelse(ensemble_probs > 0.5, "Yes", "No") %>% factor(levels = c("No", "Yes"))

# Confusion matrix
conf_matrix <- confusionMatrix(ensemble_pred, test_data$default)
print(conf_matrix)

# ROC and AUC
ensemble_roc <- roc(test_data$default, ensemble_probs)
plot(ensemble_roc, main = paste("Ensemble - AUC:", round(auc(ensemble_roc), 4)), col = "black", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
cat("Ensemble AUC:", round(auc(ensemble_roc), 4), "\n")
