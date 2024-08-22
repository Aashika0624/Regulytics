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
    - Reporting
   
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
The first step in my project is to conduct a thorough literature review and regulatory research to ensure that the developed system aligns with the current regulatory standards and effectively addresses compliance needs. This process involves several key activities:

### 4.1.1 Understanding Regulatory Frameworks
Given the complexity of the financial sector and its focus on maintaining stability, preventing fraud, and ensuring the health of financial institutions, it is essential to adhere to various regulations. These regulations often involve time-consuming tasks prone to errors, which is what my project aims to tackle by automating the reporting and compliance processes.
![image](https://github.com/user-attachments/assets/90ff4391-6904-4ef5-bba1-4f6d4041daa4)

1. **Key Regulations**
   - Basel III, IFRS 9, SOX, PRA, FCA, AML
     ![image](https://github.com/user-attachments/assets/c4529899-600c-4487-b70b-39cf37d982ad)
2. **Study Regulatory Documents**
   - Obtain and study primary regulatory documents and guidelines.
   - Summarize key requirements related to reporting and compliance standards.

### 4.1.2 Reviewing Academic and Industry Literature
Recent advances in machine learning present significant opportunities for automating and enhancing reporting and compliance analytics. My literature review will focus on three main areas:

1. **Financial Distress Prediction:**
   - Regulations such as Basel III and guidelines from the Prudential Regulation Authority (PRA) highlight the importance of stress testing and capital reserves to mitigate distress risks. I will examine recent research showing how advanced machine learning methods, including Random Forests, Gradient Boosting Machines, and Deep Neural Networks, can significantly improve prediction accuracy. Addressing imbalanced datasets, a common issue in distress prediction, will also be a key focus, with techniques like SMOTE and ADASYN offering potential solutions (Li & Sun, 2022; He et al., 2020; Wang et al., 2021).

2. **Credit Risk Assessment:**
   - Frameworks like Basel III, IFRS 9, and FCA guidelines mandate robust credit risk management. I plan to delve into recent advancements in this area, particularly those utilizing machine learning techniques like XGBoost and deep learning architectures. These methods have proven more effective than traditional regression models in predicting credit risk. Additionally, I will explore how interpretability techniques like SHapley Additive exPlanations (SHAP) and LIME can ensure transparency in these models, which is crucial for regulatory compliance (Berg et al., 2020; Brown & Mues, 2019; Lundberg & Lee, 2017).

3. **Fraud Detection:**
   - In the context of strict regulations like the UK Money Laundering Regulations and PSD2, effective fraud detection is paramount. I will investigate how deep learning advancements, particularly Recurrent Neural Networks (RNNs), have enhanced real-time fraud detection. Additionally, I plan to explore graph-based methods for anomaly detection in transaction networks and the potential of hybrid models combining unsupervised and supervised learning techniques to detect previously unknown fraud patterns (Akoglu et al., 2015; Nguyen et al., 2019; Roy et al., 2020).
     

### **4.1.3 Identify Gaps in Current Research**

Finally, a crucial part of my literature review will be identifying gaps in current research. This will involve:

1. **Regulatory Integration:**  
   - Assessing how well current predictive models integrate regulatory requirements such as those from Basel III, IFRS 9, and SOX.
2. **Real-Time Capabilities:**  
   - Evaluating the ability of existing models to handle real-time data, an essential feature for both compliance reporting and fraud detection.
3. **Scalability and Adaptability:**  
   - Determining the scalability of current solutions to manage large datasets and adapt to evolving patterns, ensuring that the system I develop can handle increasing data demands.

**Deliverables:**
- A comprehensive literature review document summarizing regulatory requirements and recent advancements in financial distress prediction, credit risk assessment, and fraud detection.
- Identification of research gaps that I will address in the dissertation, forming the foundation for developing a more effective and compliant system.
  
## 4.2 Data Analysis and Preparation
### 4.2.1 Financial Distress Prediction Dataset
#### 1. Data Description and Initial Exploration
   - The dataset used is in the CSV `content/Financial Distress.csv`     
   - **Data Description:**
     - The Financial Distress Prediction dataset is a comprehensive collection of financial metrics for various companies over multiple time periods. The primary goal is to predict the financial distress of companies based on historical financial data. The dataset contains 3672 entries and 86 columns, each representing different financial metrics and identifiers.
   - **Dataset Structure:**
     - **Rows:** 3672
     - **Columns:** 86
- **Categorization of Columns:**
     1. **Identifiers:**
        - `Company`: Unique identifier for each company.
        - `Time`: Time period identifier.
     2. **Target Variable:**
        - `Financial Distress`: The primary target variable indicating the financial distress status of the company.
     3. **Financial Metrics:**
        - `x1` to `x83`: A series of 83 financial metrics and ratios capturing various aspects of the company's financial health.
- **Data Types:**
   The dataset contains two primary data types:
   - **Integer (int64):** 5 columns (Company, Time, x80, x82, x83)
   - **Float (float64):** 81 columns (Financial Distress, x1 to x79, x81)
- **Summary Statistics:**
  - Key statistics for the dataset include the mean, standard deviation, and ranges for each metric, providing insights into the distribution and variability of the financial data.

#### 2. Time Series Analysis
- The time series analysis involves plotting the `Financial Distress` variable over time for selected companies. This helps in understanding the temporal patterns and trends in financial distress.

![image](https://github.com/user-attachments/assets/47f32c67-3e10-4638-b7b3-2710f5010d22)

- **Plot Description:**
  - The line plot shows the financial distress levels of five selected companies over different time periods. Each company's financial distress trajectory is distinct, with some companies experiencing significant fluctuations, while others exhibit more stable patterns. This variability underscores the importance of time series analysis in predicting financial distress.

#### 3. Imbalance Handling
- Initially, the `Financial Distress` variable exhibited class imbalance, which can lead to biased model predictions. To address this, the Synthetic Minority Over-sampling Technique (SMOTE) was applied to balance the dataset by generating synthetic samples for underrepresented classes.

![image](https://github.com/user-attachments/assets/95f998e2-cf8d-4765-b40d-9693239402aa) ![image](https://github.com/user-attachments/assets/b6cf9356-41a5-41a6-9c89-f2d3264085ca)

- **Distribution of Discretized Financial Distress**
  - The initial distribution of the discretized `Financial Distress` variable was highly imbalanced, with a majority of instances falling into certain distress categories. The application of SMOTE balanced the distribution, ensuring that each category is equally represented, which is crucial for training robust and unbiased predictive models.

#### 4. Feature Engineering
Feature engineering involved creating new variables to improve the predictive power of the dataset:

- **Debt-to-Equity Ratio:**
  - Calculated as the ratio of `x1` (a financial metric) to `x2` (another financial metric), providing a crucial indicator of a company's financial leverage.
- **Lag Features:**
  - Introduced lagged versions of the `Financial Distress` variable, capturing temporal dependencies and trends.
- **Normalization:**
  -  Applied MinMax scaling to normalize the range of selected features, ensuring they are on a comparable scale for modeling.
 
#### 5. Data Quality Check
A comprehensive data quality check was performed to ensure the reliability of the dataset:

- **Missing Values:**
   - Initially, there were no missing values. However, creating lag features introduced missing values, which were handled appropriately.
- **Duplicates:**
   - No duplicate rows were found in the dataset.
- **Outliers:**
   - Outliers were identified in various numerical columns, which could influence the model’s performance if not addressed.
- **Categorical Data:**
   - Ensured categorical variables had appropriate data types and filled missing values using the mode.

### 4.2.2 Credit Risk Dataset
#### 1. Data Loading and Initial Exploration
   - The dataset used is in the CSV `content/credit_risk_dataset.csv`
   - **Feature Exploration:**
     - The Credit Risk Dataset contains information on various attributes of individuals applying for loans, including their demographic details, employment information, loan characteristics, and credit history.
   - **Data Quality Check:**
     - The dataset was checked for missing values, duplicates, and outliers, and necessary preprocessing steps were applied.
#### 2. Feature Engineering

To enhance the predictive power of the dataset, additional features were engineered:

- **Debt-to-Income Ratio:**
  - This new feature was created to provide insight into the individual's debt burden relative to their income.
- **Normalization:**
  - Applied MinMax scaling to normalize the range of selected features.

#### 3. Additional Data Quality Check

The dataset was cleaned and enhanced by creating a new feature, `Debt_to_Income_Ratio`, and normalizing the relevant columns. Outliers were identified, and categorical columns were optimized for memory usage.

### 4.2.3 Credit Card Fraud Detection Dataset

#### 1. Data Loading and Initial Exploration
   - The dataset used is in the CSV `content/cc_info.csv` and `content/transactions.csv`
   - **Understanding Transaction Data:**
     - The Credit Card Fraud Detection dataset consists of transaction records linked to credit cards. The data exploration involved checking for missing values and understanding the dataset structure.

#### 2. Analyzing Transaction Amount Distribution
This analysis shows that most transactions are small, with only a few larger transactions, indicating a skewed distribution that could be critical in detecting fraud.
![image](https://github.com/user-attachments/assets/e4565056-dd2e-435e-986e-5b594d9ed4e1)

#### 3. Analyzing Transactions Over Time
This analysis reveals trends in transaction volumes and amounts, identifying periods of high activity that could correlate with increased fraud risk.
![image](https://github.com/user-attachments/assets/565e86a7-117f-4679-a370-862bcaa9a07a)
![image](https://github.com/user-attachments/assets/ecd0f43b-7d69-4486-97e0-e64ac0d8c65e)

#### 4. Graph-Based Techniques

By constructing a transaction network graph, we explored relationships between credit cards, identifying key players and potential anomalies based on centrality measures.
![image](https://github.com/user-attachments/assets/cdcec3fc-c4f7-401c-b896-c632e74b9c94)

  - **Summary**
    - This section integrates detailed data analysis and preprocessing for each of the three datasets (Financial Distress Prediction, Credit Risk, and Credit Card Fraud Detection). It covers the entire workflow from data loading and initial exploration, through feature engineering and data quality checks, to advanced analysis techniques such as time series analysis and graph-based anomaly detection. This comprehensive approach ensures the datasets are well-prepared for subsequent machine learning modeling and analysis, fulfilling the project's goals of predicting financial distress, assessing credit risk, and detecting fraud.

## 4.3 Developing Machine Learning Models
## 4.4 Automating Regulatory Reporting and System Integration
## 4.5 System Validation
## 4.6 Reporting



   
  


