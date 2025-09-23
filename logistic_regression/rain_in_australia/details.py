# https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial/notebook#Explore-problems-within-categorical-variables
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""

data = './weatherAUS.csv'

df = pd.read_csv(data)
#print(df.shape) # (145460, 23)

#print(df.head())

col_names = df.columns
#print(col_names)
"""
['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',   
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',       
       'Temp3pm', 'RainToday', 'RainTomorrow']
"""

# It is given in the dataset description, that we should drop the RISK_MM feature variable from the dataset description. So, we should drop it as follows
# df.drop(['RISK_MM'], axis=1, inplace=True)

#print(df.info())
"""
 #   Column         Non-Null Count   Dtype
---  ------         --------------   -----
 0   Date           145460 non-null  object
 1   Location       145460 non-null  object
 2   MinTemp        143975 non-null  float64
 3   MaxTemp        144199 non-null  float64
 4   Rainfall       142199 non-null  float64
 5   Evaporation    82670 non-null   float64
 6   Sunshine       75625 non-null   float64
 7   WindGustDir    135134 non-null  object
 8   WindGustSpeed  135197 non-null  float64
 9   WindDir9am     134894 non-null  object
 10  WindDir3pm     141232 non-null  object
 11  WindSpeed9am   143693 non-null  float64
 12  WindSpeed3pm   142398 non-null  float64
 13  Humidity9am    142806 non-null  float64
 14  Humidity3pm    140953 non-null  float64
 15  Pressure9am    130395 non-null  float64
 16  Pressure3pm    130432 non-null  float64
 17  Cloud9am       89572 non-null   float64
 18  Cloud3pm       86102 non-null   float64
 19  Temp9am        143693 non-null  float64
 20  Temp3pm        141851 non-null  float64
 21  RainToday      142199 non-null  object
 22  RainTomorrow   142193 non-null  object
dtypes: float64(16), object(7)
"""

# In this section, I segregate the dataset into categorical and numerical variables. There are a mixture of categorical and numerical variables in the dataset. Categorical variables have data type object. Numerical variables have data type float64.
categorical = [var for var in df.columns if df[var].dtype=='O']
# In pandas, columns with dtype 'O' usually contain strings or mixed types, so this line is used to identify all categorical (non-numeric) variables in the dataset.
#print('There are {} categorical variables\n'.format(len(categorical)))
#print('The categorical variables are :', categorical)
"""
There are 7 categorical variables
The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
"""

#print(df[categorical].isnull().sum())
"""
Date                0
Location            0
WindGustDir     10326
WindDir9am      10566
WindDir3pm       4228
RainToday        3261
RainTomorrow     3267
dtype: int64
"""

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]
#print(df[cat1].isnull().sum())
"""
WindGustDir     10326
WindDir9am      10566
WindDir3pm       4228
RainToday        3261
RainTomorrow     3267
dtype: int64
"""

# view frequency of categorical variables
#for var in categorical: print(df[var].value_counts())

# view frequency distribution of categorical variables
#for var in categorical:  print(df[var].value_counts()/float(len(df)))

# check for cardinality in categorical variables
#for var in categorical: print(var, ' contains ', len(df[var].unique()), ' labels')
"""
Date  contains  3436  labels
Location  contains  49  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  3  labels
"""

#print(df['Date'].dtypes) #object

df['Date'] = pd.to_datetime(df['Date'])

#print(df['Date'].head())
"""
0   2008-12-01
1   2008-12-02
2   2008-12-03
3   2008-12-04
4   2008-12-05
"""
# extract year from date

df['Year'] = df['Date'].dt.year
#print(df['Year'].head())
"""
0    2008
1    2008
2    2008
3    2008
4    2008
"""

df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace = True)
#print(df.info())

# check labels in location variable
#print(df.Location.unique())
#print(df.Location.value_counts())

# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method
pd.get_dummies(df.Location, drop_first=True).head()

pd_dummies_location = pd.get_dummies(df.Location, drop_first=True)

# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
pd_dummies_windgustdir = pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True)

#print(df['WindDir9am'].unique())
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
pd_dummies_winddir9m = pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True)
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
pd_dummies_winddir3pm = pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True)

#pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()

#print(pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head())

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
#print(pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0))

numerical = [var for var in df.columns if df[var].dtype!='O']
#print('There are {} numerical variables\n'.format(len(numerical)))
#print('The numerical variables are :', numerical)
"""
There are 19 numerical variables
The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
"""
#print(df[numerical].isnull().sum())
"""
MinTemp           1485
MaxTemp           1261
Rainfall          3261
Evaporation      62790
Sunshine         69835
WindGustSpeed    10263
WindSpeed9am      1767
WindSpeed3pm      3062
Humidity9am       2654
Humidity3pm       4507
Pressure9am      15065
Pressure3pm      15028
Cloud9am         55888
Cloud3pm         59358
Temp9am           1767
Temp3pm           3609
Year                 0
Month                0
Day                  0
"""
# view summary statistics in numerical variables
#print(round(df[numerical].describe()),2)
# On closer inspection, we can see that the Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm 
# columns may contain outliers.
# draw boxplots to visualize outliers

"""
plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')
plt.show()

plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')
plt.show()

plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')
plt.show()

plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
plt.show()
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')
plt.show()

plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')
plt.show()

plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')
plt.show()

plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
plt.show()
"""

# find outliers for Rainfall variable
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
#print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
# Rainfall outliers are values < -2.4000000000000004 or > 3.2

# find outliers for Evaporation variable
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
#print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
# Evaporation outliers are values < -11.800000000000002 or > 21.800000000000004

# find outliers for WindSpeed9am variable
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
#print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
# WindSpeed9am outliers are values < -29.0 or > 55.0

# find outliers for WindSpeed3pm variable
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
#print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
# WindSpeed3pm outliers are values < -20.0 or > 57.0

def max_value(mydf, col_name, max_v):
    return np.where(mydf[col_name] > max_v, max_v, mydf[col_name])

df['Rainfall'] = max_value(df, 'Rainfall', 3.2)
df['Evaporation'] = max_value(df, 'Evaporation', 21.8)
df['WindSpeed9am'] = max_value(df, 'WindSpeed9am', 55.0)
df['WindSpeed3pm'] = max_value(df, 'WindSpeed3pm', 57.0)

#print(df['Rainfall'].describe())
#print(df['Evaporation'].describe())
#print(df['WindSpeed9am'].describe())
#print(df['WindSpeed3pm'].describe())



categorical = [var for var in df.columns if df[var].dtype=='O']
#print(categorical)
#['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
numerical = [var for var in df.columns if df[var].dtype!='O']
#print(numerical)
#['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']

"""
print(df[numerical].isnull().sum())

MinTemp           1485
MaxTemp           1261
Rainfall          3261
Evaporation      62790
Sunshine         69835
WindGustSpeed    10263
WindSpeed9am      1767
WindSpeed3pm      3062
Humidity9am       2654
Humidity3pm       4507
Pressure9am      15065
Pressure3pm      15028
Cloud9am         55888
Cloud3pm         59358
Temp9am           1767
Temp3pm           3609
Year                 0
Month                0
Day                  0
"""
"""
# print the percentage of missing values in numerical variables
for col in numerical:
    if df[col].isnull().sum() > 0:
        print(col,round(df[col].isnull().mean(),4))

MinTemp 0.0102
MaxTemp 0.0087
Rainfall 0.0224
Evaporation 0.4317
Sunshine 0.4801
WindGustSpeed 0.0706
WindSpeed9am 0.0121
WindSpeed3pm 0.0211
Humidity9am 0.0182
Humidity3pm 0.031
Pressure9am 0.1036
Pressure3pm 0.1033
Cloud9am 0.3842
Cloud3pm 0.4081
Temp9am 0.0121
Temp3pm 0.0248
"""

# impute missing values in numerical variables with mean
for col in numerical:
    col_mean = df[col].mean()
    df[col] = df[col].fillna(col_mean)

#print(df[numerical].isnull().sum())

#print(df[categorical].isnull().mean())
"""
for col in categorical:
    if df[col].isnull().mean() > 0:
        print(col, round(df[col].isnull().mean(), 4))
WindGustDir 0.071
WindDir9am 0.0726
WindDir3pm 0.0291
RainToday 0.0224
RainTomorrow 0.0225
"""
for col in categorical:
    df[col] = df[col].fillna(df[col].mode()[0])

#print(df[categorical].isnull().sum())

#print(df.isnull().sum())

#print(df[categorical].head())

import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['RainToday'])
df = encoder.fit_transform(df)

categorical = [var for var in df.columns if df[var].dtype=='O']
#print(categorical)

df = pd.concat([df,pd_dummies_location,pd_dummies_windgustdir,pd_dummies_winddir9m,pd_dummies_winddir3pm ], axis=1)

y = df['RainTomorrow']
df.drop(categorical, axis=1, inplace=True)
# Convert all boolean columns to integers (1/0) using .map



duplicates = df.columns[df.columns.duplicated()].unique()

for col in duplicates:
    # Get all columns with the same name
    cols = [c for c in df.columns if c == col]
    # Convert to int and calculate row-wise average
    df[str(col) + '_avg'] = df[cols].astype(int).mean(axis=1)
    # Optionally, drop the original duplicate columns
    df.drop(columns=cols, inplace=True)
    df[col] = df.pop(str(col) + '_avg')

df = df.loc[:, df.columns.notna()]
duplicates = df.columns[df.columns.duplicated()]
#print("Duplicate columns:", duplicates)
#df.head().to_csv('weatherAUS_cleaned.csv', index=False)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

#print(df.shape) #(145460, 117)

from sklearn.model_selection import train_test_split
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.linear_model import LogisticRegression
# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)
# fit the model
logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)
# print(y_pred_test) # ['No' 'No' 'No' ... 'Yes' 'No' 'No']
# # probability of getting output as 0 - no rain
#print(logreg.predict_proba(X_test)[:,0])
# [0.86534973 0.74991153 0.79312423 ... 0.47766642 0.62890666 0.96848226]
# probability of getting output as 1 - rain
#print(logreg.predict_proba(X_test)[:,1])
# [0.13465027 0.25008847 0.20687577 ... 0.52233358 0.37109334 0.03151774]

from sklearn.metrics import accuracy_score

#print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
# Model accuracy score: 0.8485

y_pred_train = logreg.predict(X_train)
y_pred_train
#print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
# Training-set accuracy score: 0.8487

#print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
#print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
#Training set score: 0.8487
#Test set score: 0.8485

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

#print('Confusion matrix\n\n', cm)
#[[21537  1189]
#[ 3217  3149]]
#print('\nTrue Positives(TP) = ', cm[0,0])
# True Positives(TP) =  21537  (Actual Positive:1 and Predict Positive:1) 
#print('\nTrue Negatives(TN) = ', cm[1,1])
# True Negatives(TN) =  3149   (Actual Negative:0 and Predict Negative:0) 
#print('\nFalse Positives(FP) = ', cm[0,1])
# False Positives(FP) =  1189  (Actual Negative:0 but Predict Positive:1)
#print('\nFalse Negatives(FN) = ', cm[1,0])
# False Negatives(FN) =  3217  (Actual Positive:1 but Predict Negative:0) 

import matplotlib.pyplot as plt
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#plt.title("Confusion Matrix")
#plt.ylabel("Predicted label")
#plt.xlabel("Actual label")

#plt.show()

from sklearn.metrics import classification_report

#print(classification_report(y_test, y_pred_test))
"""
              precision    recall  f1-score   support

          No       0.87      0.95      0.91     22726
         Yes       0.73      0.49      0.59      6366

    accuracy                           0.85     29092
   macro avg       0.80      0.72      0.75     29092
weighted avg       0.84      0.85      0.84     29092
"""

# 真正例 模型预测为 正类，实际也是 正类。例子：肿瘤检测中，模型预测有肿瘤，实际上病人确实有肿瘤。
TP = cm[0,0]

# 真反例 模型预测为 负类，实际也是 负类。例子：肿瘤检测中，模型预测无肿瘤，实际上病人确实无肿瘤。
TN = cm[1,1]

# 假正例 / I 型错误 模型预测为 正类，实际是 负类。例子：肿瘤检测中，模型预测有肿瘤，实际上病人无肿瘤。
FP = cm[0,1]

# 假反例 / II 型错误 模型预测为 负类，实际是 正类。例子：肿瘤检测中，模型预测无肿瘤，实际上病人有肿瘤。
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
#print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)
#print('Classification error : {0:0.4f}'.format(classification_error))

# Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. 
# It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
# 在模型预测为正的样本里，有多少是真的正的。
precision = TP / float(TP + FP)
#print('Precision : {0:0.4f}'.format(precision))

# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. 
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.
# 在实际为正的样本里，有多少被模型找出来了。 True Positive
recall = TP / float(TP + FN)
#print('Recall or Sensitivity : {0:0.4f}'.format(recall))

# F1-score 是 Precision 和 Recall 的调和平均数

f1_score = 2 * (precision * recall) / (precision + recall)
#print('F1-score : {0:0.4f}'.format(f1_score))

# threshold
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]
#print(y_pred_prob)
"""
[[0.86534973 0.13465027]
 [0.74991153 0.25008847]
 [0.79312423 0.20687577]
 [0.55824255 0.44175745]
 [0.93058262 0.06941738]
 [0.96765262 0.03234738]
 [0.60057727 0.39942273]
 [0.53642722 0.46357278]
 [0.77244898 0.22755102]
 [0.7866863  0.2133137 ]]
"""
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])
#print(y_pred_prob)
# print the first 10 predicted probabilities for class 1 - Probability of rain
#y_pred1 = logreg.predict_proba(X_test)[0:10, 1]
#print(y_pred1)
y_pred1 = logreg.predict_proba(X_test)[:, 1]

# clean previous photos
plt.clf()
# adjust the font size 
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
#plt.show()

# ROC ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')
plt.clf()
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for RainTomorrow classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
#plt.show()

# ROC AUC stands for Receiver Operating Characteristic - Area Under Curve.
#  It is a technique to compare classifier performance. In this technique,
#  we measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, 
# whereas a purely random classifier will have a ROC AUC equal to 0.5.
# So, ROC AUC is the percentage of the ROC plot that is underneath the curve.

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)
#print('ROC AUC : {:.4f}'.format(ROC_AUC))

# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()
#print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

# k-Fold Cross Validation 

# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')
# print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.84854344 0.84927387 0.8496176  0.84535728 0.84883771]
# print('Average cross-validation score: {:.4f}'.format(scores.mean()))
# Average cross-validation score: 0.8483

from sklearn.model_selection import GridSearchCV
parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)

grid_search.fit(X_train, y_train)

# best score achieved during the GridSearchCV
#print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
# GridSearch CV best score : 0.8483
# print parameters that give the best results
# print('Parameters that give the best results :','\n\n', (grid_search.best_params_))
# print estimator that was chosen by the GridSearch
# print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
# LogisticRegression(penalty='l1', random_state=0, solver='liblinear')

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
# GridSearch CV score on test set: 0.8484

# result & conclusion
# 1. The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.

# 2. Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.

# 3. The model shows no signs of overfitting.

# 4. Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.

# 5. Increasing the threshold level results in increased accuracy.

# 6. ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.

# 7. Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with reduced set of features.

# 8. In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slighly higher false negatives.

# 9. ur, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.

# 10. Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507. We can see that GridSearch CV improve the performance for this particular model.