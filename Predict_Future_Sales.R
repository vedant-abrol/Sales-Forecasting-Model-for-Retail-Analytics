# Install Required Packages if not already installed
if (!require("data.table")) install.packages("data.table", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("xgboost")) install.packages("xgboost", dependencies = TRUE)

# Load Required Libraries
library(data.table)
library(ggplot2)
library(caret)
library(xgboost)

# Load the datasets
sales_train <- fread('/Users/vedantabrol/Downloads/competitive-data-science-predict-future-sales/sales_train.csv')
test <- fread('/Users/vedantabrol/Downloads/competitive-data-science-predict-future-sales/test.csv')
shops <- fread('/Users/vedantabrol/Downloads/competitive-data-science-predict-future-sales/shops.csv')
items <- fread('/Users/vedantabrol/Downloads/competitive-data-science-predict-future-sales/items.csv')
item_categories <- fread('/Users/vedantabrol/Downloads/competitive-data-science-predict-future-sales/item_categories.csv')

# Check datasets
print(head(sales_train))
print(head(test))
print(head(shops))
print(head(items))
print(head(item_categories))

# Ensure the date column is correctly formatted
sales_train[, date := as.IDate(date, format = "%d.%m.%Y")]

# Check if 'date_block_num' column exists in sales_train dataset
if (!"date_block_num" %in% colnames(sales_train)) {
  print("Column 'date_block_num' not found in sales_train dataset.")
  # Additional steps to fix the issue can be added here
} else {
  # Proceed with the rest of the code
  # Aggregate daily sales to monthly sales per shop and item
  sales <- sales_train[, .(item_cnt_month = sum(item_cnt_day)), by = .(shop_id, item_id, date_block_num)]
  # Continue with the rest of your analysis
}

# Aggregate daily sales to monthly sales per shop and item
sales <- sales_train[, .(item_cnt_month = sum(item_cnt_day)), by = .(shop_id, item_id, date_block_num)]

# Prepare the data for model
sales <- merge(sales, items[, .(item_id, item_category_id)], by = "item_id")

# Prepare test data
test <- merge(test, items[, .(item_id, item_category_id)], by = "item_id")

# Splitting the data
set.seed(123)
train_rows <- createDataPartition(sales$item_cnt_month, p = 0.8, list = FALSE)
train_data <- sales[train_rows,]
test_data <- sales[-train_rows,]

# Model training using XGBoost
model <- train(item_cnt_month ~ ., data = train_data, method = "xgbTree",
               trControl = trainControl(method = "cv", number = 10),
               tuneLength = 3)

# Prediction and model evaluation
predictions <- predict(model, test_data)
results <- postResample(predictions, test_data$item_cnt_month)
print(results)

# Predicting for submission
final_predictions <- predict(model, test[, .(shop_id, item_id, item_category_id)])
sample_submission <- fread('./sample_submission.csv')
sample_submission[, item_cnt_month := final_predictions]

# Save the results
fwrite(sample_submission, file = "final_submission.csv")

# Print out a summary of the model to check variable importance
print(summary(model))
