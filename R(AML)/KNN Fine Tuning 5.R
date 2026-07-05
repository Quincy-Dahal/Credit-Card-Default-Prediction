#libraries
library(tidyverse)
library(caret)
library(themis)
library(recipes)
library(pROC)
library(kknn)
library(doParallel)  # To enable parallel processing

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

# Create recipe with SMOTE 
recipe_smote <- recipe(default ~ ., data = train_data) %>%
  step_smote(default)#
prep_recipe <- prep(recipe_smote, training = train_data)
train_data_smote <- bake(prep_recipe, new_data = NULL)

# Parallel backend setup
cores <- detectCores() - 1  # Use one less than the available cores to avoid overload
cl <- makeCluster(cores)
registerDoParallel(cl)  

# Tune KNN
set.seed(123)
knn_model <- train(
  default ~ .,
  data = train_data_smote,
  method = "kknn",
  trControl = trainControl(
    method = "cv", 
    number = 3, 
    classProbs = TRUE, 
    summaryFunction = twoClassSummary, 
    allowParallel = TRUE  # Enable parallel processing for cross-validation
  ),
  tuneLength = 5,  
  metric = "ROC" 
)

# Stop parallel backend
stopCluster(cl)
registerDoSEQ()

# Predict and evaluate
prob_tuned <- predict(knn_model, test_data, type = "prob")[, "Yes"]
roc_tuned <- roc(test_data$default, prob_tuned)
best_thresh_tuned <- coords(roc_tuned, "best", ret = "threshold") %>% as.numeric()
pred_tuned <- ifelse(prob_tuned > best_thresh_tuned, "Yes", "No")

conf_tuned <- confusionMatrix(factor(pred_tuned, levels = c("No", "Yes")), test_data$default)
print(conf_tuned)

# Print metrics
cat("Accuracy:", round(conf_tuned$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(conf_tuned$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_tuned$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(roc_tuned), 4), "\n")

# Plot ROC
plot(roc_tuned, main = paste("Tuned KNN - AUC:", round(auc(roc_tuned), 4)), col = "black")
abline(a = 0, b = 1, lty = 2, col = "gray")