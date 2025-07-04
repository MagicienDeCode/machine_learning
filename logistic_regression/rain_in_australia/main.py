# https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial/notebook#Explore-problems-within-categorical-variables
# import numpy as np # linear algebra
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

# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()

#print(df['WindDir9am'].unique())

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)

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

