import pandas as pd # data processing, CSV 
import matplotlib.pyplot as plt # phtos

csv_file = "./weatherAUS.csv"
df = pd.read_csv(csv_file)

# print(df.shape) # (145460, 23)

col_names = df.columns.tolist()
#print(col_names)
# ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']
# print(df.info())
"""
Data columns (total 23 columns):
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
# numerical -> float64, int64
# categorical -> object

###################################################### categorical ######################################################################
categorical = [var for var in df.columns if df[var].dtype=='O']
# print(len(categorical),categorical)
# 7 ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

numerical = [var for var in df.columns if df[var].dtype!='O']
# print(len(numerical),numerical)
# 16 ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

# null values in categorical
null_cat = [c for c in categorical if df[c].isnull().sum()>0]
# print(df[null_cat].isnull().sum())
"""
WindGustDir     10326
WindDir9am      10566
WindDir3pm       4228
RainToday        3261
RainTomorrow     3267
"""

for col in categorical: df[col] = df[col].fillna(df[col].mode()[0])
# ull_cat = [c for c in categorical if df[c].isnull().sum()>0]
# print(df[null_cat].isnull().sum())

# for c in categorical: print(f"{c}: {df[c].value_counts()}")
# for c in categorical: print(df[c].value_counts()/len(df[c]))

# unique values in categorical
# for c in categorical: print(c, ' contains ', len(df[c].unique()), ' unique values')
"""
Date  contains  3436  unique values
Location  contains  49  unique values
WindGustDir  contains  17  unique values
WindDir9am  contains  17  unique values
WindDir3pm  contains  17  unique values
RainToday  contains  3  unique values
RainTomorrow  contains  3  unique values
"""


# print(df["Date"].value_counts())
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace = True)

# print(df.Location.unique())
# print(df.Location.value_counts())

# it converts categorical data (like strings or categories) into numerical indicator (0/1) columns. 
# one-hot encoding categorical variables
# drop_first=True in pd.get_dummies() means drop the first category for each categorical variable and only keep the rest.
# This is mostly used to avoid dummy variable trap / multicollinearity in regression models (because if you keep all dummy columns, one of them is redundant — it can always be inferred from the others).
# pd.get_dummies(df, columns=['Location'], drop_first=True)
pd_dummies_location = pd.get_dummies(df.Location, drop_first=True)
#print(pd_dummies_location.head())

# create an extra column that indicates if the value was NaN (missing) in the original data.
# pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True)
pd_dummies_windgustdir = pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True)
# pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True)
pd_dummies_winddir9m = pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True)
# pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True)
pd_dummies_winddir3pm = pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True)

"""
# RainToday_0, RainToday_1
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['RainToday'])
df = encoder.fit_transform(df)
"""

df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

y = df['RainTomorrow']
categorical = [var for var in df.columns if df[var].dtype=='O']
# print(categorical)
df.drop(categorical, axis=1, inplace=True)

categorical = [var for var in df.columns if df[var].dtype=='O']
# print(categorical)

# TO DO this before train model
# df = pd.concat([df,pd_dummies_location,pd_dummies_windgustdir,pd_dummies_winddir9m,pd_dummies_winddir3pm ], axis=1)

###################################################### numerical ######################################################################

numerical = [var for var in df.columns if df[var].dtype!='O']
# print(len(numerical),numerical)
# print(df[numerical].isnull().sum())
"""
MinTemp            1485
MaxTemp            1261
Rainfall           3261
Evaporation       62790
Sunshine          69835
WindGustSpeed     10263
WindSpeed9am       1767
WindSpeed3pm       3062
Humidity9am        2654
Humidity3pm        4507
Pressure9am       15065
Pressure3pm       15028
Cloud9am          55888
Cloud3pm          59358
Temp9am            1767
Temp3pm            3609
RainToday        145460
Year                  0
Month                 0
Day                   0
"""

# print(df['Rainfall'].describe())

# check outliers, min max, we can see 4 columns have outliers: Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm 
"""
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')
plt.show()
"""

"""
plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
plt.show()
"""
# Interquartile Range (IQR) 四分位距
# normal range = [Q1 - 1.5*IQR, Q3 + 1.5*IQR], extreme outliers = [Q1 - 3*IQR, Q3 + 3*IQR]
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

import numpy as np
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

for col in numerical:
    col_mean = df[col].mean()
    df[col] = df[col].fillna(col_mean)

# print(df.isnull().sum()) # all 0, good to go


###################################################### before train model ######################################################################

# append dummies, then drop duplicates
df = pd.concat([df,pd_dummies_location,pd_dummies_windgustdir,pd_dummies_winddir9m,pd_dummies_winddir3pm ], axis=1)
duplicates = df.columns[df.columns.duplicated()].unique()
# print(duplicates)

for col in duplicates:
    # Get all columns with the same name
    cols = [c for c in df.columns if c == col]
    # Convert to int and calculate row-wise average
    df[str(col) + '_avg'] = df[cols].astype(int).mean(axis=1)
    # Optionally, drop the original duplicate columns
    df.drop(columns=cols, inplace=True)
    df[col] = df.pop(str(col) + '_avg')

# drop columns with NaN column names
df = df.loc[:, df.columns.notna()]

duplicates = df.columns[df.columns.duplicated()].unique()
# print(duplicates)

bool_cols = df.select_dtypes(include='bool').columns
# convert boolean columns to int (0 and 1)
df[bool_cols] = df[bool_cols].astype(int)

#print(df.info())

###################################################### train model ######################################################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)
X = pd.DataFrame(X_scaled, columns=df.columns)
# y line 120 y = df['RainTomorrow']
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

###################################################### Model Evaluation ######################################################################

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)
# print(cm)
# [[21539  1187]
#  [ 3227  3139]]
#print('\nTrue Positives(TP) = ', cm[0,0])
# True Positives(TP) =  21537  (Actual Positive:1 and Predict Positive:1) 
#print('\nTrue Negatives(TN) = ', cm[1,1])
# True Negatives(TN) =  3149   (Actual Negative:0 and Predict Negative:0) 
#print('\nFalse Positives(FP) = ', cm[0,1])
# False Positives(FP) =  1189  (Actual Negative:0 but Predict Positive:1)
#print('\nFalse Negatives(FN) = ', cm[1,0])
# False Negatives(FN) =  3217  (Actual Positive:1 but Predict Negative:0) 

"""
import seaborn as sns
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# clean previous photos

plt.title("Confusion Matrix")
plt.ylabel("Predicted label")
plt.xlabel("Actual label")

plt.show()
"""

# 真正例 模型预测为 正类，实际也是 正类。例子： 模型预测下雨， 实际确实下雨。
TP = cm[0,0]

# 真反例 模型预测为 负类，实际也是 负类。例子： 模型预测无雨， 实际确实无雨。
TN = cm[1,1]

# 假正例 / I 型错误 模型预测为 正类，实际是 负类。例子：模型预测下雨， 实际无雨。
FP = cm[0,1]

# 假反例 / II 型错误 模型预测为 负类，实际是 正类。例子：模型预测无雨， 实际下雨。
FN = cm[1,0]


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
# print('Classification accuracy : {0:0.4f}'.format(classification_accuracy)) # Classification accuracy : 0.8483

classification_error = (FP + FN) / float(TP + TN + FP + FN)
# print('Classification error : {0:0.4f}'.format(classification_error)) # Classification error : 0.1517

# Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. 
# It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
# 在模型预测为正的样本里，有多少是真的正的。
precision = TP / float(TP + FP)
# print('Precision : {0:0.4f}'.format(precision)) # Precision : 0.9478

# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. 
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.
# 在实际为正的样本里，有多少被模型找出来了。 True Positive
recall = TP / float(TP + FN)
# print('Recall or Sensitivity : {0:0.4f}'.format(recall)) Recall or Sensitivity : 0.8697

# F1-score 是 Precision 和 Recall 的调和平均数

f1_score = 2 * (precision * recall) / (precision + recall)
# print('F1-score : {0:0.4f}'.format(f1_score)) # F1-score : 0.9071

y_pred1 = logreg.predict_proba(X_test)[:, 1]

# result & conclusion
# 1. The logistic regression model accuracy score is 0.8483. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.
# 2. Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.
# 3. The model shows no signs of overfitting.
