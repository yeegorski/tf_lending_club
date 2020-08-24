# Lending Club: Predicting Loan Defaulters
LendingClub is the world's largest peer-to-peer lending platform.

The project is a part of the Jose Portilla's Data Science Bootcamp [course](https://www.udemy.com/share/101WaUB0MdcFdXRXQ=/). The project has been performed using deep learning library TensorFlow.

* Performed the exploratory data analysis
* Performed data preprocessing (coping with missing data, transforming categorical data, train-test split, and data normalizing)
* Engineered features from the existing variables
* Created and evaluated the model built using TensorFlow
* Checked an exising entry from the dataset against the model 

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, tensorflow  
**Data set:** https://www.kaggle.com/wordsforthewise/lending-club.

## The Goal
Based on the historical data provided in the dataset, a model is to be built that will predict if the borrower will fully pay the loan or will default on it (so the loan will be charged-off). The model will allow to predict if a new potential customer is likely to pay back the loan.

## EDA
Due to the nature of the problem, the dataset in regards to the labels is quite imbalanced - most of the loans get paid. This pattern can be seen in the visualizations below, along with other insights (highly correlated features and distributions of loan statuses per grade):

![alt text](https://github.com/yeegorski/tf_lending_club/blob/master/loan_status_countplot.png "Loan status - Imbalanced target")

Loan status - Imbalanced target

![alt text](https://github.com/yeegorski/tf_lending_club/blob/master/corr_matrix.png "Numeric variable correlation - High correlation between loan_amnt and installment")

Numeric variable correlation - High correlation between loan_amnt and installment

![alt text](https://github.com/yeegorski/tf_lending_club/blob/master/sub_grade_distribution.png "Distribution of loans per subgrade - Paid ones peak at B3, while defaulted - at C4")

Distribution of loans per subgrade - Paid ones peak at B3, while defaulted - at C4

## Data Preprocessing

### Missing data
Six variables have missing data points. I started with the ones that had the most null values.
* emp_title: the job title of the borrower
    ** since there were around 173K unique titles (around 50% of the whole dataset), I decided to drop the variable.

