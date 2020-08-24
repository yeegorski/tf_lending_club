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

![alt text](https://github.com/yeegorski/tf_lending_club/blob/master/target_corr.png "Correlation of target with other numeric features")

Correlation of target with other numeric features

![alt text](https://github.com/yeegorski/tf_lending_club/blob/master/sub_grade_distribution.png "Distribution of loans per subgrade - Paid ones peak at B3, while defaulted - at C4")

Distribution of loans per subgrade - Paid ones peak at B3, while defaulted - at C4

## Data Preprocessing

### Missing data
Six variables have missing data points. I started with the ones that had the most null values.
* *emp_title*: the job title of the borrower (5.79% missing)
    * since there were around 173K unique titles (around 50% of the whole dataset), I decided to drop the variable.
* *emp_length*: employment length in years (4.62% missing)
    * after analysis of the ratios fully paid to defaulted loans per each category (per length), it turned out there was no much of a difference among them. Around 19% of all loans have been defaulted in each category. This variable has been dropped as well.
    
| emp_length  | loan_status   | %        |
| :---        |    :----:     |     ---: |
| 1 year	     | Charged Off	| 0.199135 |
| 10+ years	  | Charged Off	| 0.184186 |
| 2 years	  | Charged Off	| 0.193262 |
| 3 years	  | Charged Off	| 0.195231 |
| 4 years	  | Charged Off	| 0.192385 |
| 5 years	  | Charged Off	| 0.192187 |
| 6 years	  | Charged Off   | 0.189194 |
| 7 years	  | Charged Off	| 0.194774 |
| 8 years	  | Charged Off	| 0.199760 |
| 9 years	  | Charged Off	| 0.200470 |
| < 1 year	  | Charged Off	| 0.206872 |
   
* *title*: the loan title provided by the borrower (0.44% missing)
    * the variable *purpose* provided essentially same information in a more systematic fashion. *title* has been dropped
* *mort_acc*: number of mortgage accounts (9.54% missing)
    * the variable has the highest correlation with the target, so it was worth to keep it, and impute the missing data
    * for imputation I used another feature - *total_acc* (number of total accounts) - which was somewhat correlated to *mort_acc*
    * the data were grouped by *total_acc*, and an average of *mort_acc* was calcualted for each group
    * based on such averages, the missing data in *mort_acc* was imputed
* *revol_util* (revolving line utilization rate) and *pub_rec_bankruptcies* (number of public record bankruptcies): 0.07% and 0.14% missng respectively
    * the missing data for those features constitute a very small percent of the whole dataset length, so I decided to drop the rows with the null values 
    
### Categorical data
There are eleven categorical variables (excluding the target) in the dataset. Each of them was analyzed separately, and transformations were made according to their nature.
* *term*: he number of payments on the loan. Values are in months and can be either 36 or 60.
    * this is essentially a binary variable. I decided to keep the numbers to reflect the difference in period lengths.
* *grade* and *sub_grade*: assigned by the LendingClub
    * *grade* is duplicating the information already available in the *sub_grade* variable, so it was dropped
    * *sub_grade* was converted to dummy variables
* *verification_status*, *purpose*, *initial_list_status*, *application_type* were all appropriate for one-hot encoding as well.
* *home_ownership*: the home ownership status provided by the borrower during registration.
    * I converted very rare instances into one category, and applied one-hot encoding as well
* *address*: provided by the borrower in the application
    * I extracted zip-codes from the addresses, and since there were only 10 unique ones, I converted the feature into dummy variables
* *issue_d*: date of loan issue. Since the goal of the model is to predict, if the borrower will pay or default on the loan, we cannot know the issue date. Therefore, there is no point of keeping such feature for the model.
* *earliest_cr_feature*: the month the borrower's earliest reported credit line was opened.
    * the data are provided in mmm-yyyy format, I decided to keep only the year for the model

