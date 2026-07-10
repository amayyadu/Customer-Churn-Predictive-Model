# Customer-Churn-Predictive-Model
# ChurnGuard: Customer Churn Prediction Machine Learning Pipeline

ChurnGuard is a machine learning project designed to predict telecom customer churn using historic customer demographic, account, and service usage data. The repository includes an end-to-end pipeline: data cleaning, feature engineering, standardizing data, training a Logistic Regression model, and an interactive prediction tool.

## 🚀 Features
* **Data Preprocessing:** Handles missing values, strips whitespace, standardizes categorical values, and handles outliers.
* **Feature Engineering:** Implements One-Hot Encoding via dummy variables to prepare categorical features for modeling.
* **Predictive Modeling:** Trains a Logistic Regression model achieving high accuracy in determining customer retention.
* **Interactive CLI Tool:** Includes a built-in user interface to input custom customer metrics and instantly receive a Churn vs. Stay prediction.

## 📊 Dataset
The script utilizes `churnguard_data.csv`, which contains key operational metrics such as:
* `tenure` (Months the customer stayed)
* `MonthlyCharges` and `TotalCharges`
* Categorical data: `Contract` type, `InternetService`, `PaymentMethod`, etc.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
