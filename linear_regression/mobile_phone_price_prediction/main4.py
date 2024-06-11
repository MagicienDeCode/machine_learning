import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

df.drop(['Name','Unnamed: 0'],axis=1,inplace=True)
df['company'] = df['company'].str.lower()

# Android_version
fix_android_version_condition = df['External_Memory'].str.contains('Android', na=False)
temp_fix_android_version = df['External_Memory'].where(fix_android_version_condition)
temp_fix_android_version = temp_fix_android_version.str.extract(r'(\d+)')[0]
df['Android_version'] = df['Android_version'].combine_first(temp_fix_android_version)
df['Android_version'] = df['Android_version'].str.extract(r'(\d+)').astype(float)


fix_huawei_version_condition = df['External_Memory'].str.contains('HarmonyOS', na=False)
temp_fix_huawei_version = df['External_Memory'].where(fix_huawei_version_condition)
temp_fix_huawei_version = temp_fix_huawei_version.str.extract(r'(\d+)')[0]
df['HarmonyOS_version'] = temp_fix_huawei_version


comany_android_version = df.groupby('company')['Android_version'].mean().items()
comany_android_version_mapping = {k:v for k,v in comany_android_version}

#filtered_df = df[df['Android_version'].isna() & df['HarmonyOS_version'].isna()]
#print(filtered_df['company'].value_counts())
condition = df['Android_version'].isna() & df['HarmonyOS_version'].isna()
df.loc[condition, 'Android_version'] = df.loc[condition, 'company'].map(comany_android_version_mapping)

#filtered_df = df[df['Android_version'].isna() & df['HarmonyOS_version'].isna()]
#df = df.dropna(subset=['Android_version', 'HarmonyOS_version'], how='all')
df.fillna({'Android_version':0},inplace=True)
df.fillna({'HarmonyOS_version':0},inplace=True)

#print(df.isnull().sum())

# Inbuild_memory
#print(df['Inbuilt_memory'].value_counts(dropna=False))
condition = (df['Ram'].str.contains('inbuilt', na=False)) & (df['Inbuilt_memory'].isna())
df.loc[condition, 'Inbuilt_memory'] = df.loc[condition, 'Ram']
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.replace("1 TB","1024 GB")
df = df.dropna(subset=['Inbuilt_memory'])
df = df[df['Inbuilt_memory'].str.contains("inbuilt")]
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)').astype(int)
#print(df['Inbuilt_memory'].value_counts())

# Ram
"""
print(df['Ram'].value_counts())
filtered_df = df[df['Inbuilt_memory'] == 256]
print(filtered_df['Ram'].value_counts().idxmax()) # 12 GB RAM
filtered_df = df[df['Inbuilt_memory'] == 128]
print(filtered_df['Ram'].value_counts().idxmax()) # 8 GB RAM
filtered_df = df[df['Inbuilt_memory'] == 512]
print(filtered_df['Ram'].value_counts().idxmax()) # 12 GB RAM
"""

ram_patterns = {
    '256 GB inbuilt': '12 GB RAM',
    '512 GB inbuilt': '12 GB RAM',
    '128 GB inbuilt': '8 GB RAM'
}
df['Ram'] = df['Ram'].str.strip().replace(ram_patterns)
df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(float)


df['Battery'] = df['Battery'].str.split(' ', expand=True)[0].astype(int)
df['Display'] = df['Display'].str.split(' ', expand=True)[0].astype(float)


# Fast Charging

df['fast_charging'] = df['fast_charging'].str.extract(r'(\d+)').astype(float)
#print(df['fast_charging'].value_counts())
#filtered_df = df[df['fast_charging'].isna()]

comany_fast_chargings = df.groupby('company')['fast_charging'].mean().items()
comany_fast_charging_mapping = {k:v for k,v in comany_fast_chargings}

condition = df['fast_charging'].isna()  # 判断列 'B' 是否为空
df.loc[condition, 'fast_charging'] = df.loc[condition, 'company'].map(comany_fast_charging_mapping)

# Processor
#filtered_df = df[df['Processor'].isna()]
#print(filtered_df)
comany_processor = df.groupby('company')['Processor'].apply(lambda x: x.mode()[0]).items()
comany_processor_mapping = {k:v for k,v in comany_processor}
condition = df['Processor'].isna() 
df.loc[condition, 'Processor'] = df.loc[condition, 'company'].map(comany_processor_mapping)

#print(df.isnull().sum())
#print(df.shape)


# fix camera data
#print(df.sample(10))
#test = df[~df['Camera'].str.contains("MP")]
#print(test['Camera'].value_counts())

condition = (df['External_Memory'].str.contains('MP')) & (~df['Camera'].str.contains("MP"))
df.loc[condition, 'Camera'] = df.loc[condition, 'Camera'] + df.loc[condition, 'External_Memory']
display_columns = ['Foldable Display', 'Dual Display']
for dc in display_columns:
    df[dc] = df['Camera'].apply(lambda x: 1 if dc in x else 0)

#print(df['Camera'].value_counts())

import re
import numpy as np
def extract_back_camera(x):
    try:
        return x.split("&")[0]
    except IndexError:
        return np.nan
    except AttributeError:  # if x is np.nan, it will raise AttributeError
        return np.nan

def extract_front_camera(x):
    try:
        return x.split("&")[1]
    except IndexError:
        return np.nan
    except AttributeError:  # if x is np.nan, it will raise AttributeError
        return np.nan


df['Front_Camera'] = df['Camera'].apply(extract_front_camera)
df['Back_Camera'] = df['Camera'].apply(extract_back_camera)

#print(df['Front_Camera'].value_counts())

def extract_c1_numbers(camera_str):
    try:
        res= re.findall(r'(\d+)\s*MP', camera_str)
        return float(res[0])
    except TypeError:
        return 0
def extract_c2_numbers(camera_str):
    try:
        res= re.findall(r'(\d+)\s*MP', camera_str)
        return float(res[1])
    except IndexError:
        return 0
    except TypeError:
        return 0
def extract_c3_numbers(camera_str):
    try:
        res= re.findall(r'(\d+)\s*MP', camera_str)
        return float(res[2])
    except IndexError:
        return 0
    except TypeError:
        return 0

df['Front_Camera_1'] = df['Front_Camera'].apply(extract_c1_numbers)
df['Front_Camera_2'] = df['Front_Camera'].apply(extract_c2_numbers)
df.fillna({'Front_Camera_1':0},inplace=True)
df.fillna({'Front_Camera_2':0},inplace=True)


df['Back_Camera_1'] = df['Back_Camera'].apply(extract_c1_numbers)
df['Back_Camera_2'] = df['Back_Camera'].apply(extract_c2_numbers)
df['Back_Camera_3'] = df['Back_Camera'].apply(extract_c3_numbers)
df.fillna({'Back_Camera_1':0},inplace=True)
df.fillna({'Back_Camera_2':0},inplace=True)
df.fillna({'Back_Camera_3':0},inplace=True)


df.drop(['Front_Camera','Back_Camera','Camera'],axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Processor_name'] = le.fit_transform(df['Processor_name'])

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

def extract_resolution_pixels(resolution_str):
    match = re.search(r'(\d+) x (\d+)', resolution_str)
    if match:
        width, height = map(int, match.groups())
        return width * height / 1000
    return 0

df['Screen_resolution'] = df['Screen_resolution'].apply(extract_resolution_pixels)


tb_gb_pattern= {'1 TB':'1024 GB','2 TB':'2048 GB'}
df['External_Memory'] = df['External_Memory'].astype(str).str.strip().replace(processor_mapping)
df['External_Memory'] = df['External_Memory'].str.extract(r'(\d+)\s*GB')
df.fillna({'External_Memory':0},inplace=True)
df['External_Memory'] = df['External_Memory'].astype(int)


sim_columns = ['Dual Sim', 'Single Sim', '3G', '4G', '5G', 'VoLTE', 'Vo5G']
for scol in sim_columns:
    df['Sim_'+scol.lower().replace(" ","_")] = df['No_of_sim'].apply(lambda x: 1 if scol in x else 0)
df.drop('No_of_sim',axis=1,inplace=True)
df['Processor_name'] = le.fit_transform(df['Processor_name'])

df['company'] = le.fit_transform(df['company'])

df['Price'] = df['Price'].str.replace(',', '')
df['Price'] = df['Price'].str[:-2] + '.' + df['Price'].str[-2:]
df['Price'] = df['Price'].astype(float)


############################################

import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)

X = data.drop(columns=['Price'])
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(data)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)