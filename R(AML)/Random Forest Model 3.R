# Libraries
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(doParallel)

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

# Enable parallel processing
cores <- detectCores()
cl <- makeCluster(cores - 1)
registerDoParallel(cl)

# Train basic Random Forest
set.seed(123)
rf_basic_model <- train(
  default ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  ntree = 300
)

# Stop parallel backend
stopCluster(cl)
registerDoSEQ()

# Predict and evaluate
rf_pred <- predict(rf_basic_model, newdata = test_data)
rf_prob <- predict(rf_basic_model, newdata = test_data, type = "prob")[, "Yes"]
confusionMatrix(rf_pred, test_data$default)

rf_roc <- roc(test_data$default, rf_prob)
auc_value <- auc(rf_roc)
plot(rf_roc, 
     main = paste("ROC Curve - RF | AUC:", round(auc_value, 4)), 
     col = "black", 
     lwd = 2)
abline(a = 0, b = 1, col = "gray", lty = 2)
cat("AUC:", round(auc_value, 4), "\n")
