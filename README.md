# Bank Customer Churn Prediction
This project aims to build a predictive model to identify bank customers at high risk of churning (i.e., closing their accounts). By leveraging machine learning techniques and customer profile data, we propose a data-driven approach to improve retention strategies and reduce revenue loss due to customer attrition.

Kaggle link: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

# Business Context
XYZ Bank is a retail banking institution operating in France, Spain, and Germany. It offers a range of financial products including checking and savings accounts, credit cards, and investment services. Recently, the bank has observed a rise in customer churn, with a significant proportion of attrition occurring in the 30–60 age group.

- **Overall churn rate**: 4.2%
- **High-risk age segment**: 30–60 years old (88.27% of all churn cases)
- **Economic impact**: Based on industry benchmarks, the average customer lifetime value (LTV) is estimated at €1,800, and the customer acquisition cost (CAC) at €400. This implies a net loss of €1,400 for every customer who leaves the bank.

### Project Objectives

The project has both business and technical goals:

- **Business Objective**: Deploy a predictive system within the bank’s CRM that automatically triggers proactive retention protocols. These may include personalized offers, benefit adjustments, or alerts for at-risk customers based on model-driven segmentation.

- **Analytical Objective**: Develop a machine learning model with at least **85% prediction accuracy** to classify customers by churn risk. The goal is to reduce the overall churn rate by **at least 10%** over the next six months. The final model should be deployment-ready by the end of the current fiscal quarter.

### Dataset Overview

This dataset consists of 10,000 customer records from XYZ Bank. Each row represents a client and contains demographic, financial, and behavioral features. The target variable is `Exited`, indicating whether the customer has left the bank.

**Features:**


| Feature          | Type        | Description                                                  |
|------------------|-------------|--------------------------------------------------------------|
| `RowNumber`      | Integer     | Row index (for internal reference)                          |
| `CustomerId`     | Integer     | Unique ID assigned to each customer                         |
| `Surname`        | Text        | Customer’s last name                                        |
| `CreditScore`    | Integer     | Credit score assigned by the bank                           |
| `Geography`      | Categorical | Country of residence (France, Germany, Spain)               |
| `Gender`         | Categorical | Customer’s gender                                           |
| `Age`            | Integer     | Age of the customer                                         |
| `Tenure`         | Integer     | Number of years the customer has been with the bank         |
| `Balance`        | Float       | Account balance in euros                                    |
| `NumOfProducts`  | Integer     | Number of financial products used by the customer           |
| `HasCrCard`      | Binary      | Whether the customer owns a credit card (1 = Yes, 0 = No)   |
| `IsActiveMember` | Binary      | Whether the customer is considered active (1 = Yes, 0 = No) |
| `EstimatedSalary`| Float       | Estimated yearly salary                                     |
| `Exited`         | Binary      | Target variable: 1 if the customer left the bank, 0 if not  |

### 🔍 Exploratory Data Analysis (EDA)

Before beginning the analysis, we performed a basic data quality audit to ensure the dataset was suitable for modeling. We checked for missing values and found that all columns were complete, with zero null values across the entire dataset (10,000 records and 14 variables). This allowed us to proceed without requiring imputation or row removal.

**Steps:**

```python
# Check for missing values
data.isnull().sum()

# Check for duplicated rows
data.duplicated().sum()

# Basic statistics
data.describe()
```

We validated that the dataset has no missing values across any feature:

| Feature           | Missing Values |
|------------------|----------------|
| RowNumber         | 0              |
| CustomerId        | 0              |
| Surname           | 0              |
| CreditScore       | 0              |
| Geography         | 0              |
| Gender            | 0              |
| Age               | 0              |
| Tenure            | 0              |
| Balance           | 0              |
| NumOfProducts     | 0              |
| HasCrCard         | 0              |
| IsActiveMember    | 0              |
| EstimatedSalary   | 0              |
| Exited            | 0              |


### Univariate Statistical Summary

To better understand the distribution and characteristics of the dataset’s **numerical variables**, we computed basic descriptive statistics:

| Variable         | Mean      | Median    | SD        | CV (Mean) | CV (Median) | Skewness | Kurtosis | Min     | Max        | Range      |
|------------------|-----------|-----------|-----------|------------|--------------|----------|-----------|----------|-------------|------------|
| CreditScore      | 650.53    | 652.00    | 96.65     | 0.15       | 0.15         | -0.07    | -0.43     | 350.00   | 850.00      | 500.00     |
| Age              | 38.92     | 37.00     | 10.49     | 0.27       | 0.28         | 1.01     | 1.39      | 18.00    | 92.00       | 74.00      |
| Tenure           | 5.01      | 5.00      | 2.89      | 0.58       | 0.58         | 0.01     | -1.17     | 0.00     | 10.00       | 10.00      |
| Balance          | 76,485.89 | 97,198.54 | 62,397.40 | 0.82       | 0.64         | -0.14    | -1.49     | 0.00     | 250,898.09  | 250,898.09 |
| NumOfProducts    | 1.53      | 1.00      | 0.58      | 0.38       | 0.58         | 0.74     | 0.58      | 1.00     | 4.00        | 3.00       |
| EstimatedSalary  | 100,090.24| 100,193.91| 57,510.49 | 0.57       | 0.57         | 0.00     | -1.18     | 11.58    | 199,992.48  | 199,980.90 |


### Univariate Analysis – Key Insights of numeric variables

- **Balance:**  
  Average of €76,486 and standard deviation of €62,397 indicate **high dispersion**.  
  Slightly negative skewness (-0.14) and negative kurtosis (-1.48) suggest a **slightly left-skewed distribution** with thinner tails than a normal curve.  
  Several values are zero → may represent **clients with no active funds**, which could be relevant for churn prediction.

- **Age:**  
  Mean age is 38.9 years, with **positive skewness** (1.01) and **moderate kurtosis** (1.39).  
  The distribution leans toward younger clients but includes a **long tail of older customers**.  
  Ranging from 18 to 92 → age can be **segmented into groups** (e.g., young, adults, seniors) for churn analysis.

- **NumOfProducts:**  
  Mean is 1.53 with a median of 1 → most customers hold **only 1 or 2 products**.  
  Positive skewness (0.74) shows that **few customers have 3 or more products**.  
  This feature may reflect **relationship depth** or customer loyalty.

- **CreditScore:**  
  The distribution is **quite symmetric and stable** (skew ≈ 0, CV ≈ 0.15).  
  Ranges between 350 and 850, with no evident outliers → **clean and ready for modeling**.

- **EstimatedSalary:**  
  Average around €100,000 but with a **very wide range** (from €11.5 to €199,992).  
  No significant skewness → although due to high dispersion, **scaling or normalization** may be needed when using models sensitive to feature magnitude.
