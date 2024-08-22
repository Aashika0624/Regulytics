# Regulytics

Regulytics is a "Data-Driven Regulatory Reporting and Compliance Analytics System" machine learning-driven system that automates regulatory reporting processes and identifies potential compliance risks within financial institutions.

# Table of Contents

 1. Getting Started
 2. Overview
     - Aim
     - Objective
 3. System Design
 4. Methodology
    - Literature Review and Regulatory Research
    - Data Analysis and Preparation
    - Developing Machine Learning Models
    - Automating Regulatory Reporting and System Integration
    - System Validation
   
# 1. Getting Started

To run this project, make sure the dependencies are installed:
1. `pandas`
2. `matplotlib`
3. `imbalanced-learn`
4. `scikit-learn`
5. `networkx`
6. `xgboost`

Use the following code to install the dependencies:

```python
!pip install pandas matplotlib imbalanced-learn scikit-learn networkx xgboost
```

# 2. Overview
## 2.1 Aim
The project aims to address the challenges associated with manual, time-consuming, and error-prone regulatory reporting and compliance processes within financial institutions. The system being developed will facilitate the efficient identification of compliance risks and ensure that financial institutions can adapt quickly to regulatory changes without significant manual intervention.
## 2.2 Objective
1. To research and understand the regulatory environment and reporting requirements for financial institutions.
2. To design and develop a machine learning-driven system for automating regulatory reporting processes.
3. To create an analytics model capable of identifying potential compliance risks based on historical data.
4. To develop predictive analytics capabilities for forecasting potential compliance issues.
5. To implement an automated report generation module that adheres to the formatting and submission requirements of various regulatory bodies.
6. To validate the system's accuracy and efficiency through real-world scenario testing.
7. To document the system's development process, model methodologies, and the efficacy of the analytics in enhancing regulatory compliance.

# 3. System Design
![image](https://github.com/user-attachments/assets/f8a4fccd-6159-4167-be3f-001b88f90538)

# 4. Methodology

## 4.1 Literature Review and Regulatory Research

### 4.1.1 Understanding Regulatory Frameworks
![image](https://github.com/user-attachments/assets/90ff4391-6904-4ef5-bba1-4f6d4041daa4)

1. **Key Regulations**
   - Basel III, IFRS 9, SOX, PRA, FCA, AML
     ![image](https://github.com/user-attachments/assets/c4529899-600c-4487-b70b-39cf37d982ad)
2. **Study Regulatory Documents**
   - Obtain and study primary regulatory documents and guidelines.
   - Summarize key requirements related to reporting and compliance standards.

### 4.1.2 Reviewing Academic and Industry Literature
My literature review will focus on three main areas:

1. **Financial Distress Prediction**
   - Review recent machine learning models used for financial distress prediction.
2. **Credit Risk Assessment**
   - Explore advancements in credit risk modeling.
3. **Fraud Detection**
   - Review real-time and hybrid fraud detection models.
**Deliverable:**
- A comprehensive literature review that outlines regulatory requirements and relevant academic findings.

## 4.2 Data Analysis and Preparation (Supports Objectives 2, 3, 4)

### 4.2.1 Data Preprocessing

1. **Data Loading and Exploration**
   - Load and explore Financial Distress, Credit Risk, and Transaction datasets.
   - Perform initial data cleaning (handling missing values, outliers).
2. **Feature Engineering**
   - Create features like Debt-to-Equity Ratio, Lagged Financial Distress, etc., to enhance model predictive capabilities.
3. **Data Imbalance Handling**
   - Use techniques like SMOTE to address class imbalance in datasets.
**Deliverable:**
- Cleaned and preprocessed datasets ready for model development.

## 4.3 Developing Machine Learning Models (Supports Objectives 2, 3, 4)

### 4.3.1 Compliance Risk Identification Model

1. **Model Development**
   - Develop models (Random Forest, Gradient Boosting, XGBoost) to identify potential compliance risks.
2. **Model Evaluation**
   - Evaluate models using precision, recall, F1-score, and ROC AUC.
**Deliverable:**
- Validated compliance risk identification model.

### 4.3.2 Predictive Analytics for Compliance Issues

1. **Model Training and Testing**
   - Train and test models to forecast potential compliance issues.
   - Use techniques like time-series analysis or trend detection.
2. **Model Validation**
   - Validate predictive models with historical compliance data.
**Deliverable:**
- Predictive models capable of forecasting compliance issues.

## 4.4 Automating Regulatory Reporting and System Integration (Supports Objective 5)

### 4.4.1 Report Generation and Automation

1. **Data Preparation and Model Development**
   - Load, clean, and preprocess the data (financial distress, credit risk, transaction data).
   - Develop models that automate report generation, ensuring data is organized according to regulatory requirements.   
2. **Report Formatting and Validation**
   - Format data to meet regulatory standards.
   - Validate reports to ensure they adhere to the necessary formatting and submission requirements.

### 4.4.2 System Integration

1. **Integrate Models**
   - Combine the compliance risk, predictive analytics, and report generation models into a unified system.
2. **Automated Reporting Module**
   - Implement a module to automate the entire reporting process.
**Deliverable:**
- An integrated system with automated report generation, validated and ready for submission.

### 4.4.3 Regulatory Environment and Reporting Requirements for Financial Institutions

1. **Overview**:
   - Summarize the key regulatory frameworks (Basel III, IFRS 9, SOX, PRA, FCA, PSD2, Money Laundering Regulations) that the automated system must comply with.
2. **Compliance Reporting**:
   - Demonstrate how the system generates key regulatory compliance reports (e.g., Basel III metrics, IFRS 9 Expected Credit Loss, SOX logging) and adheres to specific requirements for PRA, FCA, PSD2, and AML regulations.
**Deliverable:**
- A concise suite of compliance reports generated through the integrated system, ensuring adherence to various regulatory requirements.

## 4.5 System Validation (Supports Objective 6)

### 4.5.1 Model Performance Evaluation
To validate the system, a Random Forest model was fine-tuned using the combined dataset, which was prepared by integrating features from financial distress, credit risk, and transaction data. The model's performance was evaluated using a separate validation set, ensuring that the results reflect the model's ability to generalize to unseen data.

The following steps were undertaken:
1. **Data Splitting**: The combined dataset was split into training (70%) and validation (30%) sets using a random seed of 42 for reproducibility.  
2. **Feature Scaling**: To ensure that all features contributed equally to the model, the data was scaled using a `StandardScaler`. This step was crucial as it brought all features onto a similar scale, improving the model's convergence during training.

Optimized Validation Accuracy: 0.96
Optimized Validation Precision: 0.94
Optimized Validation Recall: 0.99
Optimized Validation F1-Score: 0.97
Optimized Validation ROC AUC: 0.99

Optimized Validation Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.93      0.96       967
           1       0.94      0.99      0.97       946

    accuracy                           0.96      1913
   macro avg       0.97      0.96      0.96      1913
weighted avg       0.97      0.96      0.96      1913

3. **Model Fine-Tuning**: The Random Forest model was fine-tuned with the following hyperparameters:
   - `n_estimators=500`: Increased the number of trees in the forest to 500, allowing the model to capture more complex patterns in the data.
   - `max_depth=30`: Limited the maximum depth of each tree to 30, preventing overfitting while maintaining the model's ability to learn intricate relationships.
   - `min_samples_split=3`: Reduced the minimum number of samples required to split an internal node, enabling the model to create deeper trees where necessary.
   - `min_samples_leaf=1`: Set the minimum number of samples required at a leaf node to 1, allowing the model to fit as tightly as possible to the data.
   - `max_features='sqrt'`: Configured the model to consider the square root of the number of features when looking for the best split, balancing complexity and computational efficiency.
   - `class_weight='balanced'`: Automatically adjusted class weights to handle any imbalances in the dataset.

4. **Model Training**: The optimized Random Forest model was trained on the scaled training set.

5. **Model Evaluation**: The model's performance was evaluated on the validation set using the following metrics:
   - **Accuracy**: The overall accuracy of the model, representing the proportion of correctly predicted instances.
   - **Precision**: The proportion of positive identifications that were actually correct.
   - **Recall**: The proportion of actual positives that were correctly identified.
   - **F1-Score**: The harmonic mean of precision and recall, providing a single metric to evaluate the model's performance.
   - **ROC AUC**: The area under the Receiver Operating Characteristic curve, indicating the model's ability to distinguish between classes.


## The results are summarized as follows:
- **Optimized Validation Accuracy**: 0.96
- **Optimized Validation Precision**: 0.94
- **Optimized Validation Recall**: 0.99
- **Optimized Validation F1-Score**: 0.97
- **Optimized Validation ROC AUC**: 0.99

The classification report for the validation set provided additional insights into the model's performance across the different classes:

| Metric     | Class 0 | Class 1 |
|------------|---------|---------|
| Precision  | 0.99    | 0.94    |
| Recall     | 0.93    | 0.99    |
| F1-Score   | 0.96    | 0.97    |

The model demonstrated excellent performance with high accuracy, precision, recall, and F1-scores. The ROC AUC score of 0.99 indicates that the model is highly effective at distinguishing between compliant and non-compliant instances, making it well-suited for integration into the automated regulatory reporting system. These results validate the model's capability to accurately predict outcomes in a real-world setting, ensuring that the system can reliably generate compliance reports that meet regulatory standards.
