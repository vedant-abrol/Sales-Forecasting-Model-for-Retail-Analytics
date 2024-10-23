---

# Sales Forecasting Model for Retail Analytics

This project is part of the "Predict Future Sales" competition on Kaggle, where the objective is to forecast the total number of products sold across different shops for a given month. The model uses historical daily sales data to predict future sales and is built using XGBoost, a powerful machine learning algorithm optimized for performance.

## Project Overview

The dataset includes daily historical sales data from January 2013 to October 2015. The task is to predict the total monthly sales for each shop-item pair in the test set. The model is trained using sales data and other features such as shop and item categories.

## Dataset Information

The dataset contains the following files:

- `sales_train.csv`: Historical daily sales data for training the model.
- `test.csv`: Test set for predicting the number of products sold for each shop and item.
- `items.csv`: Information about products (items).
- `item_categories.csv`: Information about item categories.
- `shops.csv`: Information about shops.
- `sample_submission.csv`: A sample submission file in the required format for Kaggle competition.

## Requirements

The project uses R programming for data analysis and model building. The following R packages are required:

- `data.table`
- `ggplot2`
- `caret`
- `xgboost`

We can install these packages using the following commands in R:

```r
if (!require("data.table")) install.packages("data.table", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("xgboost")) install.packages("xgboost", dependencies = TRUE)
```

## How to Use

### 1. Load the Dataset

First, ensure that all the datasets (`sales_train.csv`, `test.csv`, `items.csv`, `item_categories.csv`, `shops.csv`) are located in your working directory. You can load them into R using the `fread()` function from the `data.table` package.

```r
library(data.table)

sales_train <- fread('sales_train.csv')
test <- fread('test.csv')
items <- fread('items.csv')
item_categories <- fread('item_categories.csv')
shops <- fread('shops.csv')
```

### 2. Data Preprocessing

- **Formatting Dates**: Convert the date column in `sales_train` to a proper date format.
  
  ```r
  sales_train[, date := as.IDate(date, format = "%d.%m.%Y")]
  ```

- **Aggregating Sales Data**: Aggregate daily sales data into monthly sales by item and shop.
  
  ```r
  sales <- sales_train[, .(item_cnt_month = sum(item_cnt_day)), by = .(shop_id, item_id, date_block_num)]
  ```

- **Merging Datasets**: Merge the sales data with `items` to include item categories.

  ```r
  sales <- merge(sales, items[, .(item_id, item_category_id)], by = "item_id")
  test <- merge(test, items[, .(item_id, item_category_id)], by = "item_id")
  ```

### 3. Splitting the Data

Split the data into training and testing sets to evaluate the model before predicting on the test set.

```r
set.seed(123)
train_rows <- createDataPartition(sales$item_cnt_month, p = 0.8, list = FALSE)
train_data <- sales[train_rows,]
test_data <- sales[-train_rows,]
```

### 4. Model Training

Train the XGBoost model with cross-validation to find the best hyperparameters.

```r
model <- train(item_cnt_month ~ ., data = train_data, method = "xgbTree", 
               trControl = trainControl(method = "cv", number = 10), tuneLength = 3)
```

### 5. Predictions

Make predictions on the test set and evaluate the model's performance.

```r
predictions <- predict(model, test_data)
results <- postResample(predictions, test_data$item_cnt_month)
print(results)
```

For submission, predict sales for the Kaggle test set and save it in the required format.

```r
final_predictions <- predict(model, test[, .(shop_id, item_id, item_category_id)])
sample_submission <- fread('sample_submission.csv')
sample_submission[, item_cnt_month := final_predictions]
fwrite(sample_submission, 'submission.csv')
```

### 6. Hyperparameter Tuning (Optional)

We can further optimize the model by experimenting with hyperparameter tuning. Modify the parameters `nrounds`, `max_depth`, and `eta` to improve model performance.

```r
tune_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.03, 0.05)
)

model <- train(item_cnt_month ~ ., data = train_data, method = "xgbTree", 
               trControl = trainControl(method = "cv", number = 10), tuneGrid = tune_grid)
```

## Running the Project

1. Clone the repository or download the files from Kaggle.
2. Install the required R packages.
3. Load the datasets into your R environment.
4. Follow the steps outlined above to preprocess the data, train the model, and make predictions.

## Additional Information

- **Feature Engineering**: We can improve the model by adding additional features like "month" and "year" extracted from the date column.
- **Ensemble Methods**: For better performance, we can try ensemble methods like combining XGBoost with other models like Random Forest.
- **Performance Monitoring**: Use metrics like RMSE and MAE to monitor model performance and fine-tune it accordingly.

## References

- Kaggle Competition: [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)
- Coursera Course: "How to Win a Data Science Competition"  
- Documentation for XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

---
