import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

# Display the first few rows of the DataFrame
#print(df.head())

# Display the summary information of the DataFrame, how many rows and columns it has, and the data types of each column
#print(df.info())

#print(df.describe())
# Display the summary statistics of the 'Price' column
#print(df['Price'].describe())

# Display the unique values in the 'Price' column
#print(df['Price'].value_counts())

# drop one column, axis=1 apply column
#df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop(['Name','Unnamed: 0'],axis=1,inplace=True)
#print(df.info())
#print(df.isnull().sum())

#df['company'] = df['company'].str.lower()
df['company'] = df['company'].apply(lambda x: x.lower())
#print(df['company'].value_counts())

#print(df.info())
#print(df['External_Memory'].value_counts())

# Android verion fix
external_memory_contains_android = df['External_Memory'].str.contains('Android', na=False)
android_version_in_external_memory = df['External_Memory'].where(external_memory_contains_android)
android_version_in_external_memory = android_version_in_external_memory.str.extract(r'(\d+)')[0]
df['Android_version'] = df['Android_version'].combine_first(android_version_in_external_memory)
df['Android_version'] = df['Android_version'].str.extract(r'(\d+)').astype(float)
#print(df['Android_version'].value_counts())

external_memory_contains_harmonyos = df['External_Memory'].str.contains('HarmonyOS', na=False)
harmonyos_in_external_memory = df['External_Memory'].where(external_memory_contains_harmonyos)
harmonyos_in_external_memory = harmonyos_in_external_memory.str.extract(r'(\d+)')[0]
df['HarmonyOS_version'] = harmonyos_in_external_memory

#print(df['HarmonyOS_version'].value_counts(dropna=False))
#print(df['Android_version'].value_counts(dropna=False))

# calculate the mean Android version for each company
comany_android_version = df.groupby('company')['Android_version'].mean().items()
comany_android_version_mapping = {k:v for k,v in comany_android_version}
#print(comany_android_version_mapping)
condition = df['Android_version'].isna() & df['HarmonyOS_version'].isna()
df.loc[condition, 'Android_version'] = df.loc[condition, 'company'].map(comany_android_version_mapping)
df.fillna({'Android_version':0},inplace=True)
df.fillna({'HarmonyOS_version':0},inplace=True)
#print(df['HarmonyOS_version'].value_counts(dropna=False))
df['Android_version'] = df['Android_version'].round(1)
df['HarmonyOS_version'] = df['HarmonyOS_version'].astype(int)
#print(df['Android_version'].value_counts(dropna=False))

#print(df.isnull().sum())

# Inbuild_memory
#print(df['Inbuilt_memory'].value_counts(dropna=False))
condition = (df['Ram'].str.contains('inbuilt', na=False)) & (df['Inbuilt_memory'].isna())
df.loc[condition, 'Inbuilt_memory'] = df.loc[condition, 'Ram']
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.replace("1 TB","1024 GB")
df = df.dropna(subset=['Inbuilt_memory'])
#print(df['Inbuilt_memory'].value_counts(dropna=False))
df = df[df['Inbuilt_memory'].str.strip() != 'Octa Core'] #df = df[df['Inbuilt_memory'].str.contains("inbuilt")]
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)').astype(int)
#print(df['Inbuilt_memory'].value_counts(dropna=False))

#print(df.isnull().sum())

# Ram
ram_patterns = {
    '256 GB inbuilt': '12 GB RAM',
    '512 GB inbuilt': '12 GB RAM',
    '128 GB inbuilt': '8 GB RAM'
}
df['Ram'] = df['Ram'].str.strip().replace(ram_patterns)
#print(df['Ram'].value_counts(dropna=False))
df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(float) #df['Ram'] = df['Ram'].str.extract(r'(\d+)').astype(int)
#print(df['Ram'].value_counts(dropna=False))

df['Battery'] = df['Battery'].str.split(' ', expand=True)[0].astype(int)
df['Display'] = df['Display'].str.split(' ', expand=True)[0].astype(float)

# Fast Charging
#print(df['fast_charging'].value_counts(dropna=False))
df['fast_charging'] = df['fast_charging'].str.extract(r'(\d+)').astype(float)
comany_fast_chargings = df.groupby('company')['fast_charging'].mean().items()
comany_fast_charging_mapping = {k:v for k,v in comany_fast_chargings}
condition = df['fast_charging'].isna()
df.loc[condition, 'fast_charging'] = df.loc[condition, 'company'].map(comany_fast_charging_mapping)
df['fast_charging'] = df['fast_charging'].astype(int)
#print(df['fast_charging'].value_counts(dropna=False))

#print(df.isnull().sum())

# Processor
#print(df['Processor'].value_counts(dropna=False))
# x.mode()[0] returns the first mode value (in case there are multiple modes).
comany_processor = df.groupby('company')['Processor'].apply(lambda x: x.mode()[0]).items()
comany_processor_mapping = {k:v for k,v in comany_processor}
condition = df['Processor'].isna() 
df.loc[condition, 'Processor'] = df.loc[condition, 'company'].map(comany_processor_mapping)
#print(df['Processor'].value_counts(dropna=False))
processor_mapping = {
    'Octa Core': 'Octa Core',
    'Octa Core Processor': 'Octa Core',
    'Quad Core': 'Quad Core',
    'Deca Core': 'Deca Core',
    'Deca Core Processor': 'Deca Core',
    'Nine-Cores': 'Nine Core',
    'Nine Core': 'Nine Core',
    'Nine Cores': 'Nine Core',
    '1.6 GHz Processor': '1.6 GHz Processor',
    '2 GHz Processor': '2 GHz Processor',
    '1.8 GHz Processor': '1.8 GHz Processor',
    '1.3 GHz Processor': '1.3 GHz Processor',
    '2.3 GHz Processor': '2.3 GHz Processor'
}
df['Processor'] = df['Processor'].str.strip().replace(processor_mapping)
processor_mapping = {
    'Octa Core': '8',
    'Nine Core':'9',
    'Deca Core':'10',
    'Quad Core':'4',
    '2 GHz Processor':'1',
    '1.8 GHz Processor':'1',
    '1.6 GHz Processor':'1',
    '1.3 GHz Processor':'1',
    '2.3 GHz Processor':'1'
}
df['Processor'] = df['Processor'].replace(processor_mapping).astype(int)
#print(df['Processor'].value_counts(dropna=False))

import re
# Screen resolution
#print(df['Screen_resolution'].value_counts(dropna=False))
def extract_resolution_pixels(resolution_str):
    match = re.search(r'(\d+) x (\d+)', resolution_str)
    if match:
        width, height = map(int, match.groups())
        return width * height / 1000
    return 0

df['Screen_resolution'] = df['Screen_resolution'].apply(extract_resolution_pixels)
df['Screen_resolution'] = df['Screen_resolution'].astype(int)
df = df[df['Screen_resolution'] != 0]
#print(df['Screen_resolution'].value_counts(dropna=False))

#print(df.isnull().sum())
#print(df.info())
#print(df['No_of_sim'].value_counts(dropna=False))
sim_columns = ['Dual Sim', 'Single Sim', '3G', '4G', '5G', 'VoLTE', 'Vo5G']
for scol in sim_columns:
    df['Sim_'+scol.lower().replace(" ","_")] = df['No_of_sim'].apply(lambda x: 1 if scol in x else 0)
df.drop('No_of_sim',axis=1,inplace=True)
#print(df.info())


df['Camera'] = df['Camera'].str.extract(r'(\d+)').astype(str)

comany_first_camera = df.groupby('company')['Camera'].apply(lambda x: x.mode()[0]).items()
comany_first_camera_mapping = {k:v for k,v in comany_first_camera}
condition = df['Camera'] == 'nan'
df.loc[condition, 'Camera'] = df.loc[condition, 'company'].map(comany_first_camera_mapping)
#print(df['Camera'].value_counts(dropna=False))
df['Camera'] = df['Camera'].astype(int)
#print(df.info())

df.drop(['Processor_name','company','External_Memory'],axis=1,inplace=True)

"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['company'] = le.fit_transform(df['company'])
print(df['company'].value_counts(dropna=False))
"""

df['Price'] = df['Price'].str.replace(',', '')
#df['Price'] = df['Price'].str[:-2] + '.' + df['Price'].str[-2:]
df['Price'] = df['Price'].astype(int)

#print(df['Price'].value_counts(dropna=False))


############################################

#df.to_csv("clean_data.csv")

from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)

#data.to_csv("clean_data2.csv", index=False)

y = data['Price']
X = data.drop(columns=['Price'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lazyReg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazyReg.fit(X_train, X_test, y_train, y_test)

print(models)

