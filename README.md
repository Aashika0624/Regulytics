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

Finally, outliers and missing data values are addressed, and a final check is conducted to ensure there are no duplicates and that the data types are correct, making the dataset ready for modeling.

## 4.3 Developing Machine Learning Models
### 4.3.1 Compliance Risk Identification
In this section, we aim to develop machine learning models that can identify potential compliance risks by combining data from the Financial Distress, Credit Risk, and Credit Card Fraud datasets. The goal is to leverage the combined insights from these datasets to build robust models that can predict compliance-related risks, such as loan defaults, financial distress, and fraudulent activities.

#### Step 1: Data Preparation and Integration
In this step, we first load the three datasets: Financial Distress, Credit Risk, and Transactions. We then engineer new features relevant to compliance risks, such as the `Debt_to_Equity` ratio in the Financial Distress dataset and the `Debt_to_Income_Ratio` in the Credit Risk dataset. After feature engineering, we combine these datasets into a single DataFrame by aligning common features and ensuring all necessary columns are included. The combined dataset is then used for model development.

#### Step 2: Model Development
In this step, developed three different machine learning models: Random Forest, Gradient Boosting, and XGBoost. Each model is trained on the combined dataset, and their performance is evaluated using the testing set. The evaluation is based on the classification report, which includes precision, recall, F1-score, and the ROC AUC score.

### 4.3.2 Predictive Analytics for Compliance Issues
In this section, I developed and validated multiple machine learning models to predict potential compliance risks using the Financial Distress dataset. The objective was to identify the most effective model for forecasting compliance-related issues, a critical component in maintaining regulatory adherence within financial institutions.

#### Data Preprocessing and Feature Engineering
The Financial Distress dataset was first subjected to feature engineering to enhance its predictive capability:
 - **Lagged_Financial_Distress**:
   - This feature captures the financial distress level of the previous period for each company. By incorporating this temporal context, the model can leverage historical trends to predict future compliance risks.
 - **Debt_to_Equity**:
   - This ratio, calculated as the ratio of `x1` (a financial metric) to `x2` (another financial metric), is a key indicator of a company’s financial leverage. Higher leverage may indicate increased financial distress, thus representing a potential compliance risk.

After feature engineering, the dataset was cleaned by removing rows with missing values. The target variable was defined as a binary indicator, where a value of `1` denotes financial distress (indicating a compliance risk), and `0` denotes no distress.

To address the issue of class imbalance—a common problem in financial distress datasets—I applied the Synthetic Minority Over-sampling Technique (SMOTE). This technique generated synthetic samples for the minority class, ensuring the model is not biased towards the majority class.

#### Model Training
Four different machine learning models were trained to predict compliance risks:

 1. **Random Forest Classifier**:
    - This ensemble model aggregates the predictions of multiple decision trees to improve predictive accuracy and robustness. It is well-suited to handle complex datasets with intricate feature interactions.
 2. **Gradient Boosting Classifier**:
    - A powerful boosting algorithm that builds models sequentially, each correcting errors from the previous one. It tends to deliver high accuracy, especially on structured data, but at the cost of longer training times. 
 3. **XGBoost Classifier**:
    - An optimized implementation of gradient boosting, XGBoost is known for its high performance and scalability, often outperforming other algorithms in structured data tasks.
 4. **Support Vector Machine (SVM)**:
    - SVM is a robust model for classification tasks, particularly effective in high-dimensional spaces. It aims to find the optimal hyperplane that separates classes with maximum margin.

#### Model Evaluation and Comparison
To evaluate the performance of these models, I used key metrics such as precision, recall, F1-score, and ROC AUC score. These metrics provide a comprehensive understanding of each model’s ability to accurately predict compliance risks.
 - **Results:**
   - **Random Forest Classifier** achieved the highest accuracy of 96%, with a precision of 0.99 and a recall of 0.93 for the no-distress class. The ROC AUC score was 0.964, indicating strong discriminatory power in distinguishing between distress and no-distress cases.
   - **Gradient Boosting Classifier** demonstrated a slightly lower accuracy of 93%, with an ROC AUC score of 0.926. Although it performed well, particularly in identifying distressed companies (recall of 0.97), it was outperformed by the Random Forest in overall accuracy and ROC AUC.
   - **XGBoost Classifier** delivered results very close to the Random Forest, with an accuracy of 96% and an ROC AUC score of 0.957. Its efficiency and ability to handle large datasets make it a highly competitive model for compliance risk prediction.
   - **Support Vector Machine (SVM)**, while effective, lagged behind the ensemble methods with an accuracy of 86% and an ROC AUC score of 0.859. It showed good precision and recall but was not as effective in capturing the complexities of the dataset compared to the tree-based models.

 - **Conclusion:**
   - The **Random Forest Classifier** emerged as the best-performing model, offering the highest accuracy and ROC AUC score. This model's ability to handle complex interactions within the data and its robust performance make it the preferred choice for predictive analytics in compliance risk management. The results obtained provide a strong foundation for integrating these predictive models into compliance frameworks, ensuring proactive identification and management of potential risks, in alignment with regulatory standards.

## 4.4 Automating Regulatory Reporting and System Integration
The objective of this section was to automate the generation of regulatory reports using machine learning models, ensuring the final reports are compliant with regulatory standards and ready for submission.
### 4.4.1 Report Generation and Automation
#### 1. Report Generation Models
 - To automate the generation of regulatory reports, I designed and trained machine learning models capable of extracting, organizing, and formatting data according to the specific requirements of regulatory bodies. The datasets used include financial distress data, credit risk data, and transaction data.
 - The following steps were carried out:
   1. **Data Preparation and Integration:**  
        - Data from various sources, including financial distress, credit risk, and transaction datasets, were merged into a single report structure. The combined dataset was cleaned by removing any rows with missing values to ensure the integrity of the data used in the reports.
  2. **Model Development:**  
        - A machine learning model, specifically a RandomForestRegressor, was developed and trained using the integrated dataset. The model was optimized to predict values where necessary and to ensure the accuracy of the final report.
  3. **Model Evaluation:**  
        - The model’s performance was evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to ensure it met the required accuracy standards.
    
#### 2. Integration with the Reporting Module
After developing the model, the next step was to integrate it into an automated reporting module that formats the data according to regulatory guidelines and generates the final report.

- **Data Formatting:**  
  - The data in the report was formatted to meet regulatory standards, including rounding financial metrics and ensuring all numerical fields were within acceptable ranges. Categorical data, such as loan status, was labeled correctly according to regulatory requirements.
- **Handling Outliers:**  
  - Specific steps were taken to handle outliers in critical fields like `Debt_to_Equity` and `Financial Distress`. This included capping values within a predefined range and replacing extreme outliers with median values to maintain the integrity of the report.
- **Validation and Finalization:**  
  - A validation step was implemented to ensure the report met all required formatting and submission standards. This included checks for missing values, ensuring numerical values were within the specified range, and verifying the correctness of categorical data.
- **Report Generation and Storage:**  
  - The final validated report was generated and saved in a CSV format, making it ready for submission to the relevant regulatory bodies.

### 4.4.2 System Integration: Model Integration and Automated Reporting
#### Overview
This section describes the process of integrating models for compliance risk, predictive analytics, and automated report generation into a unified system. The goal is to streamline the entire regulatory reporting process by automating data processing, analysis, and report generation.

#### Data Preparation Module
In this module, simulated financial and transactional data is generated. The data undergoes feature engineering to ensure it is structured appropriately for subsequent modeling tasks. The key steps include the creation of features like `Debt_to_Equity` and `Lag_Financial_Distress`, followed by handling missing values.

#### Compliance Risk Analysis Module
The compliance risk module addresses the need to assess the potential for loan default (compliance risk). This is done by training a RandomForest model on resampled data to handle class imbalance, ensuring robust predictions.

#### Predictive Analytics Module
This module focuses on forecasting potential loan defaults using an XGBoost model. The model is trained and tested using a portion of the dataset.

#### Report Generation Module
The report generation module automates the creation of a compliance report, which includes predictions on loan status. The report undergoes validation to ensure it meets the required standards before being saved as a CSV file.

#### System Integration Module
This module integrates all the above components into a unified system. It automates the entire process from data preparation to report generation, ensuring that the system is ready for deployment.


### 4.4.3 Regulatory Environment and Reporting Requirements for Financial Institutions
#### Overview
The regulatory environment for financial institutions is complex and constantly evolving. Key regulations include Basel III, IFRS 9, the Sarbanes-Oxley Act (SOX), and UK-specific regulations enforced by the Prudential Regulation Authority (PRA) and the Financial Conduct Authority (FCA). This section demonstrates how the automated regulatory reporting system can be tailored to meet these specific requirements by generating various compliance reports and providing insights into the data.

#### 1. Basel III Compliance
Basel III is a global regulatory framework that strengthens the regulation, supervision, and risk management within the banking sector.
- **Automated Compliance Reporting:**
  - The following example calculates and reports on key Basel III metrics such as the Capital Adequacy Ratio (CAR), Liquidity Coverage Ratio (LCR), and Net Stable Funding Ratio (NSFR). These metrics help ensure that the institution meets the required capital and liquidity thresholds.
  
   Company  Capital_Adequacy_Ratio  Liquidity_Coverage_Ratio  \
0        1               55.855934                  0.718958   
1        1              196.771095                  0.816668   
2        1              -17.731858                  1.264871   
3        1              -73.090814                  1.060686   
4        2                9.926182                  0.974484   

   Net_Stable_Funding_Ratio  
0                  0.323684  
1                 -0.077773  
2                  0.456391  
3                  0.835269  
4                  4.183264   


- **Explanation:**
The Basel III Compliance Report includes three key metrics:
  - **Capital Adequacy Ratio (CAR):** Measures the capital buffer of a financial institution relative to its risk-weighted assets.
  - **Liquidity Coverage Ratio (LCR):** Ensures that the institution has sufficient liquid assets to cover short-term liabilities.
  - **Net Stable Funding Ratio (NSFR):** Ensures that the institution maintains a stable funding profile in relation to the composition of its assets.
By analyzing these metrics, regulators can assess whether a financial institution maintains adequate capital and liquidity to withstand financial stress.


#### 2. IFRS 9 Compliance
The International Financial Reporting Standard 9 (IFRS 9) requires institutions to implement an Expected Credit Loss (ECL) model for financial instruments, recognizing credit impairments proactively.
- **Expected Credit Loss (ECL) Modeling:**
  - In this example, the system calculates the Expected Credit Loss (ECL) based on the Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). The report is generated to ensure compliance with IFRS 9.
    
   person_age  person_income  loan_amnt      PD   LGD    EAD  \
0          22          59000      35000  0.1602  0.45  35000   
1          21           9600       1000  0.1114  0.45   1000   
2          25           9600       5500  0.1287  0.45   5500   
3          23          65500      35000  0.1523  0.45  35000   
4          24          54400      35000  0.1427  0.45  35000   

   Expected_Credit_Loss  
0             2523.1500  
1               50.1300  
2              318.5325  
3             2398.7250  
4             2247.5250  

**Explanation:**
The IFRS 9 Compliance Report details the Expected Credit Loss (ECL) for each company. The ECL is a critical measure used to predict potential credit losses, helping institutions set aside appropriate provisions for impaired loans.


#### 3. Sarbanes-Oxley Act (SOX) Compliance
The Sarbanes-Oxley Act (SOX) mandates stringent internal controls for financial reporting to protect investors from fraudulent activities.
**Internal Controls and Reporting:**
  - The following example logs data processing activities to ensure compliance with SOX. This log helps auditors trace the steps taken during data processing, ensuring transparency and accountability.
Log Entries:
                   Timestamp  \
0 2024-08-22 14:52:38.085875   
1 2024-08-22 14:52:38.552909   
2 2024-08-22 14:52:38.558642   
3 2024-08-22 14:52:39.443827   

                                            Activity  
0  Loaded Financial Distress and Credit Risk data...  
1  Merged Financial Distress and Credit Risk data...  
2         Calculated Risk Ratio for risk assessment.  
3             Generated Basel III Compliance Report.  

Basel III Compliance Report:
   Company  Risk_Ratio
0       20    5.493001
1       20    5.493001
2       20    5.493001
3       20    5.493001
4       20    5.493001

**Explanation:**
The SOX Compliance Log records every significant data processing activity, providing a clear audit trail. This log is essential for demonstrating that the financial institution maintains rigorous internal controls and adheres to regulatory standards. The Basel III Compliance Report generated includes the calculated `Risk_Ratio` for each company, demonstrating compliance with the Basel III framework.

This process ensures that all steps taken during data processing are documented and traceable, supporting transparency and accountability in financial reporting as required by the Sarbanes-Oxley Act.

#### 4. UK-Specific Regulatory Requirements
In the UK, regulations enforced by the Prudential Regulation Authority (PRA) and the Financial Conduct Authority (FCA) focus on financial resilience and consumer protection.
**Prudential Regulation Authority (PRA) Compliance:**
 - This example calculates the Leverage Ratio, a key metric for assessing financial stability, and generates a PRA compliance report.
   Company  Leverage_Ratio
0        1       27.850854
1        1       96.976491
2        1       -5.316925
3        1      -35.915351
4        2        5.795517

**Explanation:**
The PRA Compliance Report includes the Leverage Ratio for each company. This ratio assesses the extent to which the company’s capital base is leveraged, providing insights into its financial stability.

#### 5. Financial Conduct Authority (FCA) Compliance:
In this example, the system monitors retail loan performance to ensure compliance with FCA guidelines, which emphasize consumer protection.
   loan_status  count
0            0   4423
1            1   1098

**Explanation:**
The FCA Compliance Report provides a summary of the performance of retail loans, categorized by their status (e.g., defaulted, current). This report is essential for monitoring consumer-related financial activities and ensuring compliance with consumer protection regulations.

#### 6. Payment Services Directive 2 (PSD2) Compliance
The Payment Services Directive 2 (PSD2) aims to enhance the security of electronic payments and foster innovation in the financial sector. It mandates strong customer authentication and requires detailed reporting of payment transactions.

- **Transaction Monitoring and Reporting:**
  - To comply with PSD2 reporting requirements, financial institutions must monitor large transactions and ensure that they are properly flagged and reported. The following example demonstrates how to identify and report transactions that exceed a specified threshold (e.g., $100).
         credit_card                 date  transaction_dollar_amount  \
1   1003715054175576  2015-10-24 22:23:08                     103.15   
3   1003715054175576  2015-10-22 19:41:10                     136.18   
5   1003715054175576  2015-10-17 21:28:57                     121.60   
6   1003715054175576  2015-08-29 18:34:04                     122.65   
11  1003715054175576  2015-08-24 19:29:27                     101.17   

         Long        Lat  
1  -80.194240  40.180114  
3  -80.174138  40.290895  
5  -80.243565  40.260887  
6  -80.238186  40.245928  
11 -80.278162  40.263657  

**Explanation:**
- The transactions listed above exceed the $100 threshold, making them significant for PSD2 compliance reporting. The threshold was adjusted based on the observed transaction amounts to ensure relevant data is captured.
- Each transaction includes details such as the credit card number, date, transaction amount, and the geographical coordinates (longitude and latitude).
- By flagging these large transactions, the financial institution ensures that it meets the stringent reporting requirements mandated by PSD2. This is essential for enhancing the security of electronic payments and protecting consumers.

This process is part of the institution's broader efforts to maintain compliance with PSD2, ensuring that all significant transactions are monitored, documented, and reported in accordance with regulatory standards.

#### 7. Money Laundering Regulations 2017 Compliance
The UK Money Laundering Regulations 2017 require robust measures for detecting and reporting suspicious activities.
- **Anti-Money Laundering (AML) Reporting:**
  - In this example, the system detects anomalies in transaction amounts that may indicate money laundering activities and generates an AML compliance report.
          credit_card                 date  transaction_dollar_amount  \
84   1003715054175576  2015-09-11 19:50:02                     995.35   
246  1003715054175576  2015-09-24 22:10:07                     925.78   
255  1003715054175576  2015-09-02 20:54:00                     905.54   
269  1013870087888817  2015-09-01 18:23:13                     898.00   
429  1013870087888817  2015-08-19 19:12:24                     972.63   

          Long        Lat  Anomaly  
84  -80.126760  40.225626     True  
246 -80.164139  40.237733     True  
255 -80.237126  40.290891     True  
269 -72.140018  43.220782     True  
429 -72.067729  43.119872     True  

**Explanation:**
The AML Compliance Report identifies transactions flagged as anomalous, which could indicate suspicious or fraudulent activities. This report is vital for meeting regulatory requirements related to anti-money laundering efforts.

### **4.4.4 Regulatory Environment and Reporting Requirements for Financial Institutions**

#### **Overview**
The regulatory landscape for financial institutions is inherently complex and continuously evolving, driven by the necessity to ensure financial stability, protect consumers, and prevent financial crimes. Key regulations such as Basel III, IFRS 9, the Sarbanes-Oxley Act (SOX), and various UK-specific regulations enforced by the Prudential Regulation Authority (PRA) and the Financial Conduct Authority (FCA) play crucial roles in this landscape. To comply with these regulatory requirements, financial institutions must employ robust reporting systems capable of generating accurate and timely compliance reports. This section illustrates how an automated regulatory reporting system can be tailored to meet specific regulatory requirements, with a focus on various compliance aspects.

#### 1. Basel III Compliance

**Basel III Framework:** Basel III is a global regulatory framework aimed at strengthening the regulation, supervision, and risk management of banks. It requires financial institutions to maintain sufficient capital reserves and liquidity to mitigate risks.

**Automated Compliance Reporting:**
To comply with Basel III, the reporting system calculates and reports on key metrics such as the Capital Adequacy Ratio (CAR), Liquidity Coverage Ratio (LCR), and Net Stable Funding Ratio (NSFR). These metrics are essential for ensuring that the institution meets the required capital and liquidity thresholds. The system automates the calculation of these ratios, ensuring accuracy and consistency in the reporting process. The generated Basel III Compliance Report provides regulators with a clear view of the institution's financial health, enabling them to assess whether the institution is adequately prepared to withstand financial stress.

**Example Output:**
The Basel III Compliance Report includes calculated values for CAR, LCR, and NSFR, providing insights into the institution's capital buffer, liquidity position, and funding stability. These metrics are crucial for regulatory oversight, ensuring that financial institutions operate within safe and sound practices.

#### 2. IFRS 9 Compliance

**IFRS 9 Framework:** IFRS 9 mandates financial institutions to recognize credit impairments proactively by implementing an Expected Credit Loss (ECL) model. This forward-looking approach requires institutions to estimate potential credit losses over the life of financial instruments.

**ECL Modeling:**
The automated system calculates the Expected Credit Loss (ECL) based on the Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). By integrating these metrics, the system generates an IFRS 9 Compliance Report that helps the institution assess the potential impact of credit losses on its financial statements. This proactive approach ensures that the institution maintains adequate provisions for impaired loans, aligning with regulatory expectations.

**Example Output:**
The IFRS 9 Compliance Report details the ECL for various financial instruments, enabling the institution to recognize credit impairments before they materialize. This approach enhances the transparency and accuracy of financial reporting, ensuring that the institution meets IFRS 9 requirements.

#### 3. Sarbanes-Oxley Act (SOX) Compliance

**SOX Framework:** The Sarbanes-Oxley Act (SOX) imposes stringent internal controls on financial reporting to protect investors from fraudulent activities. It requires institutions to establish processes that ensure the accuracy and integrity of financial data.

**Internal Controls and Reporting:**
The automated system logs all data processing activities, creating a transparent and traceable record of steps taken during data processing. This log serves as an audit trail, demonstrating compliance with SOX requirements and providing auditors with the information needed to verify the integrity of financial reporting.

**Example Output:**
The SOX Compliance Log records significant data processing activities, such as data loading, merging, and report generation. This log is essential for maintaining transparency and accountability in financial reporting, ensuring that the institution adheres to SOX regulations.

#### 4. UK-Specific Regulatory Requirements

**PRA Compliance:**
The Prudential Regulation Authority (PRA) focuses on ensuring the resilience of financial institutions by enforcing capital adequacy and stress testing requirements. The system calculates the Leverage Ratio, a key metric for assessing the extent to which a company’s capital base is leveraged, and generates a PRA Compliance Report.

**Example Output:**
The PRA Compliance Report includes the Leverage Ratio for each company, providing insights into the institution’s financial stability. This ratio is critical for assessing the risk exposure of the institution and ensuring that it operates within the PRA's regulatory framework.

**FCA Compliance:**
The Financial Conduct Authority (FCA) emphasizes consumer protection and market integrity. The system monitors retail loan performance to ensure compliance with FCA guidelines, generating a report that summarizes the performance of these loans.

**Example Output:**
The FCA Compliance Report categorizes loans by their status, helping the institution monitor and manage consumer-related risks effectively. This report is crucial for meeting FCA's regulatory standards and protecting consumers' financial interests.

#### 5. Payment Services Directive 2 (PSD2) Compliance

**PSD2 Framework:** PSD2 aims to enhance the security of electronic payments and promote transparency in payment services. It requires financial institutions to implement strong customer authentication and report on significant payment transactions.

**Transaction Monitoring and Reporting:**
The system monitors large transactions that exceed a predefined threshold, generating a PSD2 Compliance Report. This report ensures that all significant transactions are documented and reported in compliance with PSD2 regulations.

**Example Output:**
The PSD2 Compliance Report lists transactions that exceed the $100 threshold, including details such as transaction amount, date, and location. This report is vital for meeting PSD2 requirements and safeguarding the integrity of payment services.

#### 6. Money Laundering Regulations 2017 Compliance

**AML Framework:** The UK Money Laundering Regulations 2017 require financial institutions to implement robust anti-money laundering (AML) measures. These regulations mandate the detection and reporting of suspicious activities that could indicate money laundering.

**AML Reporting:**
The system detects anomalies in transaction amounts that may suggest suspicious activity, generating an AML Compliance Report. This report is critical for identifying and mitigating the risk of money laundering, ensuring that the institution complies with AML regulations.

**Example Output:**
The AML Compliance Report highlights transactions flagged as anomalous, providing the institution with a tool to identify potential money laundering activities. This report supports the institution’s efforts to comply with money laundering regulations and protect against financial crime.



## 4.5 System Validation
This section focuses on validating the effectiveness of the developed machine learning models within the integrated system. The primary goal is to ensure that the models provide accurate and reliable predictions, which are critical for the automated regulatory reporting system.

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



   
  


