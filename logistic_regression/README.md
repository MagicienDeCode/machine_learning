# [Logistic Regression](https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial/notebook) 

- It is a **supervised learning classification** algorithm which is used to predict observations to a discrete set of classes. 
- In statistics, the Logistic Regression model is a widely used statistical model which is primarily used for classification purposes. It means that given a set of observations, Logistic Regression algorithm helps us to classify these observations into two or more discrete classes. So, the target variable is discrete in nature.

## Sigmoid Function
- This predicted response value, denoted by z is then converted into a probability value that lie between 0 and 1. We use the sigmoid function in order to map predicted values to probability values. This sigmoid function then maps any real value into a probability value between 0 and 1.

## Decision boundary
- The sigmoid function returns a probability value between 0 and 1. This probability value is then mapped to a discrete class which is either “0” or “1”. In order to map this probability value to a discrete class (pass/fail, yes/no, true/false), we select a threshold value. This threshold value is called Decision boundary.

## Assumptions of Logistic Regression 

1. Logistic Regression model requires the dependent variable to be binary, multinomial or ordinal in nature.
2. It requires the observations to be independent of each other. So, the observations should not come from repeated measurements.
3. Logistic Regression algorithm requires little or no multicollinearity among the independent variables. It means that the independent variables should not be too highly correlated with each other.
4. Logistic Regression model assumes linearity of independent variables and log odds.
5. The success of Logistic Regression model depends on the sample sizes. Typically, it requires a large sample size to achieve the high accuracy.

1. 逻辑回归模型要求因变量是二元、多项或有序的。
2. 它要求观测值之间相互独立。因此，观测值不应来自重复测量。
3. 逻辑回归算法要求自变量之间几乎没有或没有多重共线性。这意味着自变量之间不应该有太高的相关性。
4. 逻辑回归模型假设自变量与对数几率之间具有线性关系。
5. 逻辑回归模型的成功取决于样本量。通常，它需要较大的样本量才能达到高准确性。

## Types of Logistic Regression

1. Binary Logistic Regression
In Binary Logistic Regression, the target variable has two possible categories. The common examples of categories are yes or no, good or bad, true or false, spam or no spam and pass or fail.

2. Multinomial Logistic Regression
In Multinomial Logistic Regression, the target variable has three or more categories which are not in any particular order. So, there are three or more nominal categories. The examples include the type of categories of fruits - apple, mango, orange and banana.

3. Ordinal Logistic Regression
In Ordinal Logistic Regression, the target variable has three or more ordinal categories. So, there is intrinsic order involved with the categories. For example, the student performance can be categorized as poor, average, good and excellent.