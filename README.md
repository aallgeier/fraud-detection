## Project Name 
Credit Card Fraud Detection (Group Project)

## Summary
The problem of credit card fraud detection gives us a unique opportunity to analyze a highly skewed real-world dataset. We accomplished an extensive pre-processing of the data including resampling and dimensionality reduction, and applied supervised learning for detecting fraudulent transactions. We performed a comparative analysis of the performance of machine learning methods including Naive Bayes, Logistic Regression, kNN, Extreme Gradient Boosting, Neural Nets, and Support Vector Machines. The “winner” among these algorithms based on accuracy, precision, recall and the f1-score is described in detail in subsequent sections.

<p float="left">
  <img src="images/fig4.png" width="800" />
</p>

# Dataset
IEEE-CIS Fraud Detection: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203

# Some highlights
## 1. Data Preprocessing (numerical data)
### Dropping data
-Remove features not apprpriate for in our process such as Transaction ID and TransactionDT (delta time)<br>
-Remove columns with more than a threashold-percent of NaNs 
-Fill NaN in columns with column means<br>

### Dimensionality reduction 
-After min-max normalization, remove features with variance lower than threshold since features essentially constant do not help the fitting.<br>
-Remove features with high correlation to other features since it can be redundant information when fitting<br>
-Perform PCA and standardize data beforehand (if a feature shows high variance, this may dominate the principal components).<br>

### PCA 
-Standardize data (mean 0 variance 1)<br>
-Perform PCA with scikit-learn<br>

## 2. Resampling of training data with SMOTE (numerical data)
The givien data is skewed towards transactions which are not fradulent, we will use SMOTE to sinthesize more fradulent transation data points to have a 1:1 ratio of the classes. <br>

## 3. kNN, NBayes, Linear Regression
Code provided

## 4. SVM, Neural Networks (FFN, CNN), Random Forest, XG Boost






