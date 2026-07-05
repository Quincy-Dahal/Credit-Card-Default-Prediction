# Load library
library(tidyverse)
library(caret)
library(rpart)
library(pROC)
library(recipes)
library(themis)
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

# Apply SMOTE to training data
recipe_smote <- recipe(default ~ ., data = train_data) %>%
  step_smote(default)
prep_recipe <- prep(recipe_smote, training = train_data)
train_data_smote <- bake(prep_recipe, new_data = NULL)

# Set up cross-validation and hyperparameter grid
set.seed(123)
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))

# Train decision tree with tuning
model_dt <- train(
  default ~ .,
  data = train_data_smote,
  method = "rpart",
  trControl = control,
  metric = "ROC",   
  tuneGrid = grid
)

# Visualize the tree
rpart.plot(model_dt$finalModel)

# Predict probabilities and classes
prob_predictions <- predict(model_dt, newdata = test_data, type = "prob")[, "Yes"]
roc_obj <- roc(test_data$default, prob_predictions)
best_thresh <- coords(roc_obj, "best", ret = "threshold", best.method = "youden") %>% as.numeric()
class_predictions <- ifelse(prob_predictions > best_thresh, "Yes", "No")

# Evaluate the model
conf_matrix <- confusionMatrix(factor(class_predictions, levels = c("No", "Yes")), test_data$default)
print(conf_matrix)

# Print performance
cat("Best cp:", model_dt$bestTune$cp, "\n")
cat("Accuracy:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("AUC Score:", round(auc(roc_obj), 4), "\n")

# Plot ROC
plot(roc_obj, main = paste("ROC Curve - AUC:", round(auc(roc_obj), 4)))
abline(a = 0, b = 1, lty = 2, col = "gray")
