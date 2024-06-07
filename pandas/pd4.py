import pandas as pd

df = pd.read_csv('mobile_phone_price_prediction.csv')

# drop colunm with name 'Unnamed: 0', axis=1 (foreach line)
df.drop('Unnamed: 0',axis = 1,inplace=True)

# row, col
print(df.shape)

# random pick 3 lines
print(df.sample(3))

# group by a column
print(df['Ram'].value_counts())

# define patterns, for one column, drop line that patterns contain its value
patterns = ['Helio G90T', '6000 mAh Battery with 22.5W Fast Charging']
df = df[~df['Ram'].str.contains('|'.join(patterns))]
print(df['Ram'].value_counts())

# split column Ram, expand=True (return result as a dataFrame), take first element (index=0) in format float
df['Ram'] = df['Ram'].str.split(' ', expand=True)[0].astype(float)
print(df['Ram'].value_counts())

print(df['Battery'].value_counts())
print(df['Display'].value_counts())
df['Battery'] = df['Battery'].str.split(' ', expand=True)[0].astype(float)
df['Display'] = df['Display'].str.split(' ', expand=True)[0].astype(float)
print(df['Battery'].value_counts())
print(df['Display'].value_counts())


# display count with null value
print(df['Android_version'].value_counts(dropna=False))

df.drop('Android_version',axis=1,inplace=True)
df.drop(['Name'],axis=1,inplace=True)



print(df['Inbuilt_memory'].value_counts(dropna=False))
print(df['fast_charging'].value_counts(dropna=False))
print(df['Processor'].value_counts(dropna=False))

# given columns, drop null value lines
print(df.shape)
df = df.dropna(subset=['Inbuilt_memory','fast_charging','Processor'])
#print(df['Inbuilt_memory'].value_counts(dropna=False))
#print(df['fast_charging'].value_counts(dropna=False))
#print(df['Processor'].value_counts(dropna=False))

print("======================")
print(df.isnull().sum())
print(df.shape)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='most_frequent')
"""
SimpleImputer 是 scikit-learn 库中的一个类，用于处理缺失值。通过指定 strategy 参数，你可以决定用什么值来替换缺失值。
strategy='most_frequent' 表示使用每列中最频繁出现的值来替换缺失值。
imputer.fit_transform(df[['Inbuilt_memory']]): 对 Inbuilt_memory 列进行拟合并转换，返回一个二维数组。
.ravel(): 将二维数组展平成一维数组，以便可以直接赋值给 df['Inbuilt_memory'] 列。
# Fit and transform the data, then flatten back to 1D
df['Inbuilt_memory'] = imputer.fit_transform(df[['Inbuilt_memory']]).ravel()
df['fast_charging']=imputer.fit_transform(df[['fast_charging']]).ravel()
df['Processor']=imputer.fit_transform(df[['Processor']]).ravel()
"""

# str.extract(r'(\d+)') 使用正则表达式 r'(\d+)' 从字符串中提取数值部分。正则表达式 \d+ 匹配一个或多个数字，括号 () 表示提取的部分。
df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract(r'(\d+)').astype(float)
print(df['fast_charging'].value_counts())
df['fast_charging'] = df['fast_charging'].str.extract(r'(\d+)').astype(float)
print(df['fast_charging'].value_counts())
print("=======================================================================")
print(df.isnull().sum())
print(df['Inbuilt_memory'].value_counts())
print(df['fast_charging'].value_counts())

# count unique value
print(df['Processor_name'].nunique())
df.drop(['Processor_name'],axis=1,inplace=True)


df['Processor'] = df['Processor'].str.strip()
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

# Replace the values in the 'Processor' column based on the mapping
df['Processor'] = df['Processor'].replace(processor_mapping)

# Show the processed DataFrame and the value counts
print(df['Processor'].value_counts())

print(df['No_of_sim'].value_counts())

df['No_of_sim'] = df['No_of_sim'].str.strip()
sim_importance_mapping = {
    'No Sim Supported,': '0',
    'Single Sim, 3G, 4G,': '1',
    'Dual Sim, 3G, VoLTE,': '2',
    'Single Sim, 3G, 4G, VoLTE,': '3',
    'Dual Sim, 3G, 4G,': '4',
    'Dual Sim, 3G, 4G, VoLTE,': '5',
    'Single Sim, 3G, 4G, 5G, VoLTE,': '6',
    'Dual Sim, 3G, 4G, 5G, VoLTE,': '7',
    'Single Sim, 3G, 4G, 5G, VoLTE, Vo5G,': '8',
    'Dual Sim, 3G, 4G, 5G, VoLTE, Vo5G,': '9'
}

# Replace the values in the column based on the mapping
df['Sim_encoded'] = df['No_of_sim'].replace(sim_importance_mapping).astype(int)
df.drop(['No_of_sim'],axis=1,inplace=True)

print(df['Sim_encoded'].value_counts())
"""
7    788
5    329
9     87
6     19
4      7
3      5
0      1
8      1
"""



patterns = ['Foldable Display, Dual Display']
df = df[~df['Camera'].str.contains('|'.join(patterns))]

print(df['Camera'].value_counts())

import re
def extract_back_camera_mp(camera_str):
    back_camera_part = camera_str.split('&')[0]
    if back_camera_part:
        back_cameras = re.findall(r'(\d+) MP', back_camera_part)
        return sum(map(int, back_cameras))
    return 0

def extract_front_camera_mp(camera_str):
    front_camera_part = re.search(r'(\d+) MP Front', camera_str)
    if front_camera_part:
        return int(front_camera_part.group(1))
    return 0

df['Front_Camera_MP'] = df['Camera'].apply(extract_front_camera_mp)
df['Back_Camera_MP'] = df['Camera'].apply(extract_back_camera_mp)

print(df['Front_Camera_MP'].value_counts())
print(df['Back_Camera_MP'].value_counts())

"""
print(df['Processor'].value_counts())
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
df['Processor_Encoded'] = encoder.fit_transform(df[['Processor']])

print(df['Processor_Encoded'].value_counts())
"""

processor_mapping = {
    'Octa Core': '8',
    'Nine Core':'9',
    'Deca Core':'10',
    'Quad Core':'4',
    '2 GHz Processor':'1',
    '1.8 GHz Processor':'1',
    '2.3 GHz Processor':'1'
}

# Replace the values in the column based on the mapping
df['Processor_Encoded'] = df['Processor'].replace(processor_mapping).astype(int)

print(df['Processor_Encoded'].value_counts())


print(df.isnull().sum())
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='most_frequent')
df['fast_charging']=imputer.fit_transform(df[['fast_charging']]).ravel()
print(df.isnull().sum())

def extract_resolution_pixels(resolution_str):
    match = re.search(r'(\d+) x (\d+)', resolution_str)
    if match:
        width, height = map(int, match.groups())
        return width * height
    return 0

# Apply the function to create a new column for total pixels
df['Total_Pixels'] = df['Screen_resolution'].apply(extract_resolution_pixels)

print(df['Total_Pixels'].value_counts())
df.drop(['Screen_resolution'],axis=1,inplace=True)


df.drop(columns=['Camera','External_Memory','Processor'],inplace=True)
print(df.isnull().sum())

print(df.head())
df.drop(['company'],axis=1,inplace=True)
print(df.info())

df['Price'] = df['Price'].str.replace(',', '')
df['Price'] = df['Price'].str[:-2] + '.' + df['Price'].str[-2:]
df['Price'] = df['Price'].astype(float)
print(df['Price'].value_counts())

print(df.info())