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
df['RainToday'] = df['RainToday'].map({'YES': 1, 'NO': 0})

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




###################################################### train model ######################################################################

df = pd.concat([df,pd_dummies_location,pd_dummies_windgustdir,pd_dummies_winddir9m,pd_dummies_winddir3pm ], axis=1)
