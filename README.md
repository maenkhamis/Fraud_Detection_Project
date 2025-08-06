# Fraud Detection using Machine Learning

## Project Overview

This project focuses on building a machine learning model to accurately detect fraudulent credit card transactions. Financial fraud poses a significant challenge to institutions, leading to substantial losses and erosion of customer trust. Our goal is to develop a robust predictive system that can identify suspicious transactions in real-time.

The analysis is performed on a simulated credit card transaction dataset (`fraud.csv`), which contains over 594,000 transactions. A key challenge addressed in this project is the severe **class imbalance**, where only approximately **1.21%** of transactions are fraudulent.

## Methodology

Our approach involved several key steps:

1.  **Data Loading and Exploration:** Initial analysis of the `fraud.csv` dataset to understand its structure and identify missing values.
2.  **Feature Engineering & Preprocessing:**
    * Categorical features (`age`, `gender`, `category`) were transformed using one-hot encoding.
    * A new `Hour` feature was engineered from the `step` column to capture time-based patterns.
    * The `amount` feature was scaled using `StandardScaler` to normalize its range.
3.  **Exploratory Data Analysis (EDA):** Visualizations were created to understand the distribution of fraud, transaction amounts, high-risk categories, and hourly patterns.
4.  **Addressing Class Imbalance:** Due to the severe imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data. This technique generated synthetic samples for the minority class (fraudulent transactions), balancing the training set from `Counter({0: 469954, 1: 5760})` to `Counter({0: 469954, 1: 469954})`.
5.  **Model Training & Evaluation:**
    * Multiple machine learning models were trained and evaluated: Logistic Regression, Random Forest, and XGBoost.
    * Performance was assessed using **AUC-ROC**, a suitable metric for imbalanced datasets.
    * Models were evaluated both on the original imbalanced data and on the SMOTE-resampled data to demonstrate the impact of handling imbalance.
6.  **Feature Importance Analysis:** The final model's feature importances were analyzed to identify the strongest predictors of fraud.
7.  **Model Persistence:** The trained model, the scaler, and the feature names were saved using `joblib` for future deployment and prediction on new data.

## Key Findings & Results

Our analysis yielded significant insights and a high-performing model:

* **Class Imbalance:** Fraudulent transactions are rare, making detection challenging.
* **High-Risk Indicators:** High transaction amounts, specific categories (e.g., `es_fashion`, `es_eletronic`, `es_transportation`), and certain hours of the day are strong indicators of fraud.
* **Model Performance (AUC-ROC):**
    * **Before SMOTE:**
        * Logistic Regression: 0.9917
        * Random Forest: 0.9582
        * XGBoost: 0.9789
    * **After SMOTE (on training data):**
        * Logistic Regression: 0.9911
        * Random Forest: **0.9801**
* **Best Model:** The **Random Forest Classifier**, trained on SMOTE-resampled data, emerged as the best-performing model with an **AUC-ROC score of 0.9801**.

## Business Recommendations

Based on our findings, we recommend the following actionable strategies:

1.  **Implement Real-Time Transaction Monitoring:** Integrate the trained Random Forest model into a real-time system to provide immediate fraud probability scores, enabling proactive alerts and holds on suspicious transactions.
2.  **Focus Prevention Efforts on Key Indicators:** Prioritize monitoring and rule-setting for transactions with high amounts, those occurring in identified high-risk categories (fashion, electronics, transportation), and during specific hours of the day.
3.  **Establish a Continuous Improvement Loop:** Regularly retrain the model with new, labeled transaction data to ensure it adapts to evolving fraud patterns and maintains its high accuracy over time.

## How to Run the Code

To run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourGitHubUsername]/Fraud-Detection-Project.git
    cd Fraud-Detection-Project
    ```
2.  **Install dependencies:**
    Ensure you have Python installed. Then install the required libraries:
    ```bash
    pip install pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib numpy
    ```
3.  **Place the dataset:**
    Make sure the `fraud.csv` file is in the same directory as the `task3_1.py` script.
4.  **Run the script:**
    Execute the main Python script from your terminal:
    ```bash
    python task3_1.py
    ```
    This script will perform data loading, preprocessing, model training, evaluation, saving of the model, and a final demonstration of prediction on new dummy data.
