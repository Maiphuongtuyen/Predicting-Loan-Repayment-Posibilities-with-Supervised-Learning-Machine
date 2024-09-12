# Predicting-Loan-Repayment-Posibilities-with-Supervised-Learning-Machine

### Credit scoring methods have been utilized to assess borrower creditworthiness. However, these methods often rely on a limited set of factors, potentially overlooking crucial aspects that influence loan performance. This can lead to inaccurate assessments and increase the risk of defaults for lenders. Therefore, in this study, i used supervised model to predict borrowers' repayment ability at Lending Club, and evaluate the best-performing model on the dataset. An exploratory data analysis (EDA) was also conducted to understand the dataset’s characteristics

### About dataset
LendingClub is the leading digital marketplace bank in the U.S., connecting borrowers with investors since 2007. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. 

### Dataset Download Link:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

The data provided by LendingClub includes detailed records of loan applications, borrower profiles, loan terms, credit scores, employment information, and loan performance metrics. The dataset extracted from LendingClub.com spans from the third quarter of 2016 to the fourth quarter of 2018.

In this study, i will implement 5 supervised machine learning models such as: Decision Tree Regression, Random Forest Regression, Logistic Regression Classifier, eXtreme Gradient Boosting, Adaptive Boosting Classifier to predict the customer loan repayment  and then assess which model performs best on the dataset. The figure below is a part of the dataset after removing the less important variables. However, to be able to run the predictive model, it’s necessary to remove additional variables to reduce noise in the model. I have noted these steps in my Python code.

### Method Research
In the process of setting up a supervised machine learning model, several key steps are involved to ensure its effectiveness and accuracy.
Initially, we start with raw data, which may contain various features and target variables. The first step is pre-processing, where we handle null values and drop variables that don't significantly impact our study, ensuring the dataset's cleanliness and relevance.

Next, we encode categorical variables into numerical representations, enabling the model to understand and process them effectively. Following this, we standardize the data, rescaling features to have a mean of 0 and a standard deviation of 1, to ensure consistent scaling across all features.
Subsequently, we split the data into training and testing sets, typically using an 80-20 ratio, with 80% of the data allocated for training the model and 20% for evaluating its performance.

Once the model is trained, we test its efficacy by making predictions on the test data, assessing its ability to generalize to unseen instances. Moreover, to thoroughly evaluate the model's performance and generalizability, we incorporate k-fold cross-validation. Here, the dataset is partitioned into 5 equal-sized folds, with each fold serving as both a training and validation set iteratively. This iterative process allows us to assess the model's performance across different subsets of the data, providing valuable insights into its stability and robustness.

Finally, we evaluate the model using performance metrics such as accuracy, precision, recall, F1-score, or mean squared error, to quantify its effectiveness and identify areas for improvement.

### Data Dictionary
The dataset containing detailed information on loans from LendingClub was collected from July 2016 to December 2018. The original dataset comprises a total of 151 variables. However, for the purpose of this study, a reduced dataset consisting of only 27 variables has been created, containing over 215,982 observations. This reduced dataset includes 14 numerical variables and 12 categorical variables. In this research, the target variable will be 'loan_status', which indicates whether a loan has been fully paid or poses a risk of default. A value of 1 for 'loan_status' signifies that the customer is at risk of default, also known as Charged-Off, while a value of 0 indicates that the customer has fully paid off the loan, also known as Fully-Paid.

***Dependent variable***
<br>loan_status - Current status of the loan, 0: Fully-Paid, 1: Charged-Off

***Independent variable***
<br>issue_d - The month which the loan was funded

sub_grade - LC assigned loan subgrade

term - The number of payments on the loan. Values are in months and can be either 36 months or 60 months

home_ownership - The home ownership status provided by the borrower during registration or obtained from the credit report. Values are: RENT, OWN, MORTGAGE, OTHER

fico_range_low - The lower boundary range the borrower’s FICO at loan origination belongs to.

total_acc - The total number of credit lines currently in the borrower's credit file

pub_rec - Number of derogatory public records

revol_util - Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

annual_inc - The self-reported annual income provided by the borrower during registration

int_rate - Interest Rate on the loan

dti - A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.

purpose - A category provided by the borrower for the loan request.

mort_acc - Number of mortgage accounts

loan_amnt - The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.

application_type - Indicates whether the loan is an individual application or a joint application with two co-borrowers

installment - The monthly payment owed by the borrower if the loan originates

verification_status - Indicates if income was verified by LC, not verified, or if the income source was verified

pub_rec_bankruptcies - Number of public record bankruptcies

addr_state - The state provided by the borrower in the loan application

initial_list_status - The initial listing status of the loan. Possible values are – W, F

fico_range_high - The upper boundary range the borrower’s FICO at loan origination belongs to.

revol_bal - Total credit revolving balance

id - A unique LC assigned ID for the loan listing

open_acc - The number of open credit lines in the borrower's credit file

emp_length - Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.

### Data Preprocessing
--- Handle missing value

For missing values, we have 2 variables with missing values: emp_length with 16,239 missing values, dti with 162 missing values and revol_util with 222 missing values. Since the number of missing values is very small compared to the total of 215,982 data points, we will delete the rows containing these missing values.

--- Feature selection

In this feature selection step, with the primary purpose of avoiding overfitting for the model, we will consider dropping some variables that are not significant for the model, to avoid biasing the model. Therefore, I will drop the issue_d, id, addr_state, and total_acc columns. The reason for deleting the variable total_acc is that when I implemented the feature importance of the XGBoost model, the result showed that total_acc was the least important variable in the dataset

--- Feature engineering

Firstly, we need to classify the data into two types.

The first type is numerical, consisting of the following variables: 'fico_range_low', 'total_acc', 'pub_rec', 'pub_rec_bankruptcies', 'revol_util', 'annual_inc', 'int_rate', 'dti', 'mort_acc', 'loan_amnt', 'installment', 'fico_range_high', 'revol_bal', 'open_acc', 'emp_length'.

The second type is categorical, including the remaining variables: 'sub_grade', 'term', 'home_ownership', 'application_type', 'initial_list_status', 'verification_status', , 'addr_state', 'emp_length'.

Afterwards, the categorical variables will be transformed into numerical data types, corresponding to their respective meanings. For example, in the variable 'term' which has 2 types of values: "36 months" and "60 months", it will be transformed into numbers 1 and 2 according to the increasing time duration to match its meaning.

--- Standardize data

Before training models, I utilize the StandardScaler method to preprocess the data. This technique is essential for ensuring uniformity and comparability across features in the dataset. By standardizing and scaling the features to have a mean of zero and a standard deviation of one, StandardScaler enhances the performance and accuracy of many machine learning algorithms. The process involves subtracting the mean value of each feature and then dividing by its standard deviation, effectively placing all features on the same scale and ensuring their equal importance in the model

### Research Results

***--- Exploratory Data Analysis - Descriptive statistics of the numerical variables***

The FICO score in this dataset ranges from 660 to 850, with higher scores indicating better creditworthiness. In this dataset, 25% of the FICO scores in the dataset are lower than or equal to 674, the median FICO scores for "fico_range_low" and "fico_range_high" are 690 and 694, respectively, indicating an average range within the FICO scale. P2P Lending platforms cater to those facing credit score challenges and limited access to traditional loans. Individuals with scores below 620 may find it challenging to secure loans, especially in the subprime market. The median is lower than the mean, indicating that the FICO score follows a right-skewed distribution

***--- Exploratory Data Analysis - Descriptive statistics of the categorical variables***

The analysis of categorical variables reveals insightful patterns about the borrowers. Firstly, it's notable that the most common sub-grade is 'B5', suggesting that a considerable portion of borrowers fall within this moderate credit risk category. Additionally, the prevalence of 'MORTGAGE' as the primary home ownership status indicates that a significant proportion of borrowers own homes with mortgages, reflecting the stability and asset ownership among borrowers. Moreover, the majority of loan purposes are geared towards 'debt consolidation', indicating a prevalent need among borrowers to consolidate their debts, showcasing their financial management behavior and the utility of loans in managing debt effectively.

In terms of loan characteristics, the dominance of '36 months' loan terms implies that borrowers generally opt for shorter loan durations, potentially driven by preferences for quicker repayment and lower overall interest costs. Moreover, the high frequency of 'Individual' application types suggests that most borrowers prefer to apply for loans individually rather than jointly. This insight sheds light on the typical borrower profile and application process on the lending platform, providing valuable information for assessing borrower behavior and tailoring loan offerings accordingly.

***--- Exploratory Data Analysis - Visualization with Python***

Since the loan purpose has 12 values, to make it easier to observe, I have filtered out the top 3 loan purposes with the highest number of charged-off cases and the lowest 3, the results are shown in the following chart. Small business loans have the highest proportion of charged-off cases at 20%, followed by house and debt consolidation. On the other hand, reasons for borrowing money such as buying a car, credit card, and home improvement have lower risks. This is understandable because small business loans are often perceived as riskier in credit assessment due to various factors. For example, studies have found that new businesses heavily rely on credit from informal sources such as business contacts and family, and that bank loans to small businesses tend to be personally guaranteed.

Fico_Low_Range and Fico_High_Range represent the lowest and highest credit score ranges borrowers can achieve. Therefore, I only need to use 1 chart to represent the highest credit score ranges borrowers can achieve to visualize the dispersion of data in Fico scores. Fico_High_Range exhibits a wider dispersion of credit scores for borrowers who are able to repay their debts. This indicates a greater diversity in their credit profiles compared to those with lower scores. Additionally, Fico_High_Range tends to have more outliers. This suggests a higher number of borrowers with very different credit histories within the high score range. We can explain this phenomenon in two ways. Firstly, borrowers with high Fico scores are more likely to have a variety of credit backgrounds. This includes individuals with long and spotless credit histories alongside those with shorter but still good credit journeys. Secondly, borrowers who struggle to repay debts typically share more similar credit profiles, often marked by late or missed payments.

There is a significant difference in housing situation rates between different loan repayment statuses. According to the chart, individuals who are able to repay their loans often end up in mortgage status, accounting for 52%, which is higher compared to those who are able to pay at 10%. Serrano-Cinca et al. (2015) also found that borrowers who own their homes or rent have different risk ratios compared to those with mortgages or other housing situations.

The fully-paid group typically has loans under $15,000, while the charged-off group tends to borrow under $25,000. Additionally, the percentage of borrowers in the $30,000-$40,000 range is usually higher in the charged-off group compared to the fully-paid group. This suggests that borrowers who are able to repay their loans often consider their capacity more carefully and therefore tend to borrow larger amounts than those in the charged-off group.

Polena & Regner (2018) mention that annual income is negatively correlated with loan default, suggesting that higher annual incomes may lower the risk of default. Pan & Liu (2019) further support this by indicating that variables like the term of the loan, credit rating, annual income, and line recycling rate act as protective factors, reducing the likelihood of loan default as they increase. However, based on the histogram of annual income, it is evident that the Fully-Paid group and the Charged-off group in terms of income has no significant differences.

The FICO score exhibits a negative correlation with the revol_util variable at 0.48 and with the interest_rate variable at 0.36. A higher revol_util indicates that borrowers are utilizing a significant portion of their available credit, which can increase financial risks. Since the revolving line utilization rate represents the percentage of available credit currently being used, a higher ratio adversely impacts the borrower's credit score, as indicated by research. This phenomenon is similarly explained by the relationship between credit scores and interest rates, suggesting that higher interest rates correlate with increased credit risk.

We further observe that the interest_rate variable has a positive correlation with loan_status at 0.25 and with revol_util at 0.24. This explains why individuals facing difficulties accessing capital are more likely to accept offers with higher interest rates, which in turn increases their risk of default. Although credit scores in this study do not significantly affect repayment likelihood, when combined with other factors, we can infer that they still indirectly influence an individual's credit status.

***--- Training and Testing results after upsampling***

Due to the significant disparity in the number of default and non-default borrowers in the current dataset, as depicted in the chart below, it is experiencing an imbalance issue. This imbalance phenomenon could potentially impact the model's performance. When datasets are imbalanced, meaning that one class is significantly more prevalent than another, models trained on such data tend to exhibit biases towards the majority class. Moreover, imbalanced data can also lead to model overfitting, where the model learns noise from the majority class rather than capturing meaningful patterns from the data. As a result, the evaluation of models trained on imbalanced data using standard metrics like accuracy can be misleading, as these metrics do not effectively capture the model's performance on the minority class

After upsampling the data, we have to implemented K-Fold Cross-Validation to evaluate the performance of our models. With K-Fold Cross-Validation, I divided the dataset into five subsets, trained the model on four of them, and evaluated it on the fifth. This process was repeated five times, each time with a different subset presented for evaluation. In each fold, we trained a model on the training data and then made predictions on the testing data. We calculated AUC metric to evaluate the model's performance per fold.

The results show that before applying the upsampling technique, Random Forest and XGBoost have the highest accuracy and AUC. However, after upsampling, Random Forest and Decision Tree performed better than XGBoost on these indicators. Precision and Recall of all models are also significantly improved after upsampling. The explanation for this may be that upsampling helps balance data, minimizing the impact of skewed data sets, and Random Forest and Decision Tree are better able to handle noisy data than XGBoost.

Upsampling enables the model to learn more about the less common class (Charged off - value 1), thereby improving its ability to accurately predict this class. The model has performed well in improving predictions for Charged-Off cases. This results in increased Precision and Recall for Charged-Off cases, from 0.1-0.5 to 0.6-0.8. This indicates that the model is more effective in detecting credit risk cases. Although Precision and Recall for Charged off values are improved, upsampling can also lead to a decrease in Accuracy. This may occur when the model's predictions for the common class Fully-Paid are no longer as accurate due to the changed balance between classes. Despite the decrease in Accuracy, since our goal is to improve predictions for the less common class, this reduction may be acceptable.

After upsampling the data, we have to implemented K-Fold Cross-Validation to evaluate the performance of our models. With K-Fold Cross-Validation, I divided the dataset into five subsets, trained the model on four of them, and evaluated it on the fifth. This process was repeated five times, each time with a different subset presented for evaluation. In each fold, we trained a model on the training data and then made predictions on the testing data. We calculated AUC metric to evaluate the model's performance per fold. By averaging the AUC values ​​across all folds, we obtained a more robust assessment of the model's performance. This approach allows us to effectively estimate how well our model generalizes to unseen data and provides us with valuable insights into the overall performance of the models.

***=> The best-performing model for this dataset is Random Forest, with an average AUC of 0.909, indicating that the model can explain 90% of the dataset. Following is Decision Tree with an average AUC of 0.839. Next, there are Light GBM, AdaBoost, and finally, the Logistic Regression model with an average AUC of 0.628.***


### Thanks for your time! If you have any questions or just want to chat, don't hesitate to get in touch. I'm always excited to talk about data and discover new possibilities!
